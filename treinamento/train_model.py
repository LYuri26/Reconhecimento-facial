# train_model.py - VERSÃO MELHORADA COM AUMENTO DE DADOS AGRESSIVO
import os
import cv2
import numpy as np
from deepface import DeepFace
import mysql.connector
from mysql.connector import Error
import pickle
import logging
import sys
import json
from datetime import datetime
from sklearn.preprocessing import Normalizer
import tensorflow as tf
import random


class DeepFaceTrainer:
    def __init__(self):
        self.MIN_IMAGES_PER_USER = 1
        self.MAX_IMAGES_PER_USER = 5
        self.IMAGE_SIZE = (160, 160)
        self.MIN_FACE_SIZE = 80
        self.EMBEDDING_MODEL = "Facenet512"
        self.DETECTOR = "mtcnn"
        self.THRESHOLD_BLUR = 25
        self.ENFORCE_DETECTION = False
        self.AUGMENTATION_MULTIPLIER = 30  # Número de variações por imagem original

        self.SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
        self.PROJECT_ROOT = os.path.dirname(self.SCRIPT_DIR)
        self.MODEL_DIR = os.path.join(self.PROJECT_ROOT, "model")
        self.MODEL_PATH = os.path.join(self.MODEL_DIR, "deepface_model.pkl")
        self.UPLOADS_DIR = os.path.join(self.PROJECT_ROOT, "uploads")
        self.REPORTS_DIR = os.path.join(self.PROJECT_ROOT, "training_reports")

        os.makedirs(self.MODEL_DIR, exist_ok=True)
        os.makedirs(self.REPORTS_DIR, exist_ok=True)

        # Otimizações do TensorFlow
        tf.config.threading.set_intra_op_parallelism_threads(2)
        tf.config.threading.set_inter_op_parallelism_threads(2)

        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=[
                logging.FileHandler(os.path.join(self.REPORTS_DIR, "training.log")),
                logging.StreamHandler(),
            ],
        )

        self.DB_CONFIG = {
            "host": "localhost",
            "user": "root",
            "password": "",
            "database": "reconhecimento_facial",
        }

        self.user_images = {}
        self.embeddings_db = {}
        self.label_map = {}
        self.reverse_label_map = {}
        self.training_stats = {
            "start_time": None,
            "end_time": None,
            "total_users": 0,
            "processed_users": 0,
            "total_images": 0,
            "valid_images": 0,
            "failed_images": 0,
        }

        # Normalizador para embeddings
        self.normalizer = Normalizer(norm="l2")

    def create_connection(self):
        try:
            conn = mysql.connector.connect(**self.DB_CONFIG)
            if conn.is_connected():
                return conn
        except Error as e:
            logging.error(f"Erro ao conectar ao MySQL: {e}")
        return None

    def get_user_images(self):
        conn = self.create_connection()
        if conn is None:
            return None
        try:
            cursor = conn.cursor(dictionary=True)
            cursor.execute(
                """
                SELECT c.id, c.nome, c.sobrenome 
                FROM cadastros c 
                WHERE EXISTS (SELECT 1 FROM imagens_cadastro ic WHERE ic.cadastro_id = c.id)
            """
            )
            users = cursor.fetchall()
            if not users:
                return None

            user_images = {}
            for user in users:
                cursor.execute(
                    "SELECT caminho_imagem FROM imagens_cadastro WHERE cadastro_id = %s LIMIT %s",
                    (user["id"], self.MAX_IMAGES_PER_USER),
                )
                images = [img["caminho_imagem"] for img in cursor.fetchall()]
                if images:
                    user_images[user["id"]] = {
                        "nome": user["nome"],
                        "sobrenome": user["sobrenome"],
                        "images": images,
                    }
                    self.training_stats["total_images"] += len(images)
            return user_images
        except Error as e:
            logging.error(f"Erro ao buscar imagens: {e}")
            return None
        finally:
            if conn and conn.is_connected():
                conn.close()

    def validate_image(self, img_path):
        try:
            img = cv2.imread(img_path)
            if img is None:
                return False, "Não foi possível ler a imagem"

            if img.shape[0] < self.MIN_FACE_SIZE or img.shape[1] < self.MIN_FACE_SIZE:
                return False, "Imagem muito pequena"

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            if np.mean(gray) < 15:
                return False, "Imagem muito escura"

            return True, "OK"
        except Exception as e:
            return False, f"Erro na validação: {str(e)}"

    def augment_image(self, img):
        """
        Gera múltiplas variações da imagem para aumentar o conjunto de treino.
        Retorna uma lista de imagens aumentadas (incluindo a original).
        """
        augmented = []
        try:
            # Original
            augmented.append(img)

            # Flip horizontal
            flipped = cv2.flip(img, 1)
            augmented.append(flipped)

            # Rotações
            rows, cols = img.shape[:2]
            for angle in [-10, -5, 5, 10]:
                M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
                rotated = cv2.warpAffine(img, M, (cols, rows))
                augmented.append(rotated)

            # Zoom
            for scale in [0.9, 0.95, 1.05, 1.1]:
                M = cv2.getRotationMatrix2D((cols / 2, rows / 2), 0, scale)
                zoomed = cv2.warpAffine(img, M, (cols, rows))
                augmented.append(zoomed)

            # Translação
            for tx, ty in [(-10, 0), (10, 0), (0, -10), (0, 10)]:
                M = np.float32([[1, 0, tx], [0, 1, ty]])
                translated = cv2.warpAffine(img, M, (cols, rows))
                augmented.append(translated)

            # Brilho e contraste
            for alpha in [0.8, 0.9, 1.1, 1.2]:
                for beta in [-20, -10, 10, 20]:
                    adjusted = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
                    augmented.append(adjusted)

            # Ruído gaussiano
            for sigma in [5, 10, 15]:
                noise = np.random.normal(0, sigma, img.shape).astype(np.uint8)
                noisy = cv2.add(img, noise)
                augmented.append(noisy)

            # Equalização de histograma
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            eq_gray = cv2.equalizeHist(gray)
            eq_bgr = cv2.cvtColor(eq_gray, cv2.COLOR_GRAY2BGR)
            augmented.append(eq_bgr)

        except Exception as e:
            logging.debug(f"Erro no aumento de dados: {str(e)}")
            augmented = [img]  # fallback

        # Limita ao número máximo desejado (embaralha e pega os primeiros)
        if len(augmented) > self.AUGMENTATION_MULTIPLIER:
            random.shuffle(augmented)
            augmented = augmented[: self.AUGMENTATION_MULTIPLIER]

        return augmented

    def generate_embedding(self, img_input):
        """
        Gera embedding a partir de um caminho de arquivo ou array numpy.
        """
        try:
            # Se for string (caminho), valida a imagem
            if isinstance(img_input, str):
                if not os.path.exists(img_input):
                    logging.warning(
                        f"Arquivo não encontrado: {os.path.basename(img_input)}"
                    )
                    return None
                is_valid, msg = self.validate_image(img_input)
                if not is_valid:
                    logging.warning(f"Imagem inválida: {msg}")
                    return None

            embedding_objs = DeepFace.represent(
                img_path=img_input,
                model_name=self.EMBEDDING_MODEL,
                detector_backend=self.DETECTOR,
                enforce_detection=self.ENFORCE_DETECTION,
                align=True,
                normalization="base",
            )

            if embedding_objs and len(embedding_objs) > 0:
                embedding = np.array(embedding_objs[0]["embedding"]).flatten()
                # Normalização L2
                embedding = self.normalizer.transform([embedding])[0]
                return embedding
            else:
                return None

        except Exception as e:
            logging.warning(f"Erro ao gerar embedding: {str(e)}")
            return None

    def generate_embeddings_for_user(self, user_id, user_data):
        embeddings = []
        user_name = f"{user_data['nome']} {user_data['sobrenome']}"

        logging.info(f"Processando usuário: {user_name}")
        logging.info(f"Total de imagens: {len(user_data['images'])}")

        for i, img_path in enumerate(user_data["images"]):
            full_path = os.path.join(self.UPLOADS_DIR, img_path.replace("\\", "/"))
            logging.info(
                f"Processando imagem {i+1}/{len(user_data['images'])}: {full_path}"
            )

            # Carrega a imagem original
            img = cv2.imread(full_path)
            if img is None:
                self.training_stats["failed_images"] += 1
                logging.warning(f"✗ Imagem {i+1}: Não foi possível ler")
                continue

            # Gera embedding da imagem original
            emb_original = self.generate_embedding(full_path)
            if emb_original is not None:
                embeddings.append(emb_original)
                self.training_stats["valid_images"] += 1
                logging.info(f"✓ Imagem {i+1}: Embedding original gerado")
            else:
                self.training_stats["failed_images"] += 1
                logging.warning(f"✗ Imagem {i+1}: Falha no embedding original")
                continue  # se a original falhar, não adianta aumentar

            # Gera variações aumentadas
            augmented_images = self.augment_image(img)

            # Gera embeddings para cada variação
            for j, aug_img in enumerate(augmented_images):
                # Pula a original (já processada)
                if j == 0 and np.array_equal(aug_img, img):
                    continue
                emb_aug = self.generate_embedding(aug_img)  # passa array numpy
                if emb_aug is not None:
                    embeddings.append(emb_aug)
                    self.training_stats["valid_images"] += 1
                    logging.debug(f"✓ Imagem {i+1}: Embedding aumentado {j} gerado")
                else:
                    logging.debug(f"✗ Imagem {i+1}: Embedding aumentado {j} falhou")

        if embeddings:
            # Calcula embedding médio normalizado (opcional, mas útil)
            avg_embedding = np.mean(embeddings, axis=0)
            avg_embedding = self.normalizer.transform([avg_embedding])[0]

            logging.info(f"✓ {user_name}: {len(embeddings)} embeddings gerados")

            return {
                "nome": user_data["nome"],
                "sobrenome": user_data["sobrenome"],
                "embeddings": [e.tolist() for e in embeddings],
                "embedding": avg_embedding.tolist(),
                "embedding_count": len(embeddings),
            }
        else:
            logging.warning(f"✗ {user_name}: Nenhum embedding válido")
            return None

    def save_model(self):
        if not self.embeddings_db:
            logging.error("Nenhum dado para salvar")
            return False

        os.makedirs(self.MODEL_DIR, exist_ok=True)

        model_data = {
            "embeddings_db": self.embeddings_db,
            "label_map": self.label_map,
            "reverse_label_map": self.reverse_label_map,
            "training_stats": self.training_stats,
            "model_info": {
                "version": "5.0",  # versão com aumento agressivo
                "training_date": datetime.now().isoformat(),
                "embedding_model": self.EMBEDDING_MODEL,
                "detector": self.DETECTOR,
                "normalization": "l2",
                "augmentation_multiplier": self.AUGMENTATION_MULTIPLIER,
            },
        }

        try:
            with open(self.MODEL_PATH, "wb") as f:
                pickle.dump(model_data, f)

            if os.path.exists(self.MODEL_PATH):
                file_size = os.path.getsize(self.MODEL_PATH)
                logging.info(f"✓ Modelo salvo: {self.MODEL_PATH}")
                logging.info(f"Tamanho: {file_size} bytes")
                return True
            else:
                logging.error("✗ Arquivo do modelo não foi criado")
                return False

        except Exception as e:
            logging.error(f"✗ Erro ao salvar modelo: {str(e)}")
            return False

    def create_training_report(self):
        try:
            report_data = {
                "start_time": (
                    self.training_stats["start_time"].isoformat()
                    if self.training_stats["start_time"]
                    else None
                ),
                "end_time": (
                    self.training_stats["end_time"].isoformat()
                    if self.training_stats["end_time"]
                    else None
                ),
                "total_users": self.training_stats["total_users"],
                "processed_users": self.training_stats["processed_users"],
                "total_images": self.training_stats["total_images"],
                "valid_images": self.training_stats["valid_images"],
                "failed_images": self.training_stats["failed_images"],
                "success_rate": (
                    (
                        self.training_stats["valid_images"]
                        / self.training_stats["total_images"]
                        * 100
                    )
                    if self.training_stats["total_images"] > 0
                    else 0
                ),
            }

            report_path = os.path.join(
                self.REPORTS_DIR,
                f"training_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            )

            with open(report_path, "w", encoding="utf-8") as f:
                json.dump(report_data, f, indent=2, ensure_ascii=False)

            logging.info(f"✓ Relatório salvo: {report_path}")
            return True

        except Exception as e:
            logging.error(f"✗ Erro ao criar relatório: {str(e)}")
            return False

    def train(self):
        self.training_stats["start_time"] = datetime.now()
        logging.info("\n=== INICIANDO TREINAMENTO COM AUMENTO DE DADOS AGRESSIVO ===")

        if not os.path.exists(self.UPLOADS_DIR) or not os.listdir(self.UPLOADS_DIR):
            logging.error("Pasta uploads não encontrada ou vazia")
            return False

        self.user_images = self.get_user_images()
        if not self.user_images:
            logging.error("Nenhuma imagem encontrada")
            return False

        logging.info(f"Usuários para processar: {len(self.user_images)}")
        self.training_stats["total_users"] = len(self.user_images)

        processed_users = 0
        for user_id, user_data in self.user_images.items():
            self.label_map[user_id] = processed_users
            self.reverse_label_map[processed_users] = user_id

            user_result = self.generate_embeddings_for_user(user_id, user_data)
            if user_result:
                self.embeddings_db[user_id] = user_result
                processed_users += 1
                logging.info(f"✓ {user_data['nome']} processado")

        self.training_stats["processed_users"] = processed_users
        self.training_stats["end_time"] = datetime.now()

        if not self.embeddings_db:
            logging.error("Nenhum embedding gerado")
            return False

        self.create_training_report()
        return self.save_model()

    def print_summary(self):
        """
        Exibe um relatório completo do treinamento no terminal.
        """

        if not self.training_stats["start_time"] or not self.training_stats["end_time"]:
            print("❌ Treinamento não finalizado corretamente.")
            return

        total_time = (
            self.training_stats["end_time"] - self.training_stats["start_time"]
        ).total_seconds()

        print("\n" + "=" * 75)
        print("📊 RELATÓRIO FINAL DO TREINAMENTO DE RECONHECIMENTO FACIAL")
        print("=" * 75)

        # ------------------------------------------------------------------
        # Informações gerais
        # ------------------------------------------------------------------
        print("\n🔹 INFORMAÇÕES GERAIS")

        print(f"   ⏰ Tempo total de execução : {total_time:.2f} segundos")
        print(f"   📅 Início do treinamento  : {self.training_stats['start_time']}")
        print(f"   📅 Fim do treinamento     : {self.training_stats['end_time']}")

        print(f"   🧠 Modelo de Embedding    : {self.EMBEDDING_MODEL}")
        print(f"   🔍 Detector Facial        : {self.DETECTOR}")
        print(f"   📐 Normalização           : L2 (sklearn)")
        print(
            f"   🔁 Aumento de Dados        : {self.AUGMENTATION_MULTIPLIER} variações/imagem"
        )

        # ------------------------------------------------------------------
        # Estatísticas globais
        # ------------------------------------------------------------------
        print("\n🔹 ESTATÍSTICAS GERAIS")

        total_users = self.training_stats["total_users"]
        processed_users = self.training_stats["processed_users"]
        total_images = self.training_stats["total_images"]
        valid_images = self.training_stats["valid_images"]
        failed_images = self.training_stats["failed_images"]

        print(f"   👥 Usuários encontrados   : {total_users}")
        print(f"   ✅ Usuários processados   : {processed_users}")

        print(f"   📷 Imagens originais      : {total_images}")
        print(f"   ✔️  Embeddings válidos     : {valid_images}")
        print(f"   ❌ Falhas de processamento: {failed_images}")

        if total_images > 0:
            success_rate = (valid_images / total_images) * 100
            print(f"   🎯 Taxa de sucesso         : {success_rate:.2f}%")

        # ------------------------------------------------------------------
        # Estatísticas por usuário
        # ------------------------------------------------------------------
        print("\n🔹 DESEMPENHO POR USUÁRIO")

        print("-" * 75)

        for user_id, data in self.embeddings_db.items():

            nome = f"{data['nome']} {data['sobrenome']}"
            total_imgs_user = len(self.user_images[user_id]["images"])
            total_embs = data["embedding_count"]

            avg_embs = total_embs / total_imgs_user if total_imgs_user > 0 else 0

            print(f"👤 Usuário: {nome}")
            print(f"   🆔 ID                : {user_id}")
            print(f"   📁 Imagens originais : {total_imgs_user}")
            print(f"   🧬 Embeddings        : {total_embs}")
            print(f"   📊 Média por imagem  : {avg_embs:.1f}")
            print("-" * 75)

        # ------------------------------------------------------------------
        # Diagnóstico do treinamento
        # ------------------------------------------------------------------
        print("\n🔹 DIAGNÓSTICO AUTOMÁTICO")

        if processed_users == 0:
            print("   ❌ Nenhum usuário foi treinado.")
            print("   👉 Verifique banco de dados e diretório uploads.")

        elif processed_users < total_users:
            print("   ⚠️  Nem todos os usuários foram processados.")
            print("   👉 Algumas imagens podem estar inválidas.")

        else:
            print("   ✅ Todos os usuários foram processados com sucesso.")

        if failed_images > valid_images:
            print("   ⚠️  Alto índice de falhas detectado.")
            print("   👉 Verifique iluminação, enquadramento e resolução.")

        if valid_images < 100:
            print("   ⚠️  Base de dados pequena.")
            print("   👉 Recomenda-se mais imagens por usuário.")

        # ------------------------------------------------------------------
        # Status final
        # ------------------------------------------------------------------
        print("\n🔹 STATUS FINAL")

        model_size = 0
        if os.path.exists(self.MODEL_PATH):
            model_size = os.path.getsize(self.MODEL_PATH) / (1024 * 1024)

        print(f"   💾 Modelo salvo em : {self.MODEL_PATH}")
        print(f"   📦 Tamanho        : {model_size:.2f} MB")

        print("\n" + "=" * 75)
        print("✅ TREINAMENTO FINALIZADO COM SUCESSO")
        print("=" * 75)


if __name__ == "__main__":
    try:
        print("=" * 60)
        print("🤖 INICIANDO TREINAMENTO FACIAL COM AUMENTO DE DADOS")
        print("=" * 60)

        trainer = DeepFaceTrainer()

        if trainer.train():
            trainer.print_summary()
            print("=" * 60)
            print("✅ TREINAMENTO CONCLUÍDO!")
            print("=" * 60)
            sys.exit(0)
        else:
            print("=" * 60)
            print("❌ FALHA NO TREINAMENTO")
            print("=" * 60)
            sys.exit(1)

    except Exception as e:
        print("=" * 60)
        print("❌ ERRO CRÍTICO")
        print(f"   Erro: {str(e)}")
        import traceback

        traceback.print_exc()
        print("=" * 60)
        sys.exit(1)
