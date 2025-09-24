# train_model.py - ATUALIZADO
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


class DeepFaceTrainer:
    def __init__(self):
        self.MIN_IMAGES_PER_USER = 1
        self.MAX_IMAGES_PER_USER = 5  # Reduzido para refletir a realidade
        self.IMAGE_SIZE = (160, 160)
        self.MIN_FACE_SIZE = 80
        self.EMBEDDING_MODEL = "Facenet512"  # Mantido por ser um dos melhores
        self.DETECTOR = "mtcnn"  # mais estável
        self.THRESHOLD_BLUR = 25
        self.ENFORCE_DETECTION = False

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

            # Verifica tamanho mínimo
            if img.shape[0] < self.MIN_FACE_SIZE or img.shape[1] < self.MIN_FACE_SIZE:
                return False, "Imagem muito pequena"

            # Verifica se a imagem está muito escura
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            if np.mean(gray) < 15:
                return False, "Imagem muito escura"

            return True, "OK"
        except Exception as e:
            return False, f"Erro na validação: {str(e)}"

    def augment_image(self, img):
        """Aumento de dados para melhorar reconhecimento com poucas imagens"""
        augmented = []

        try:
            # Imagem original
            augmented.append(img)

            # Flip horizontal
            augmented.append(cv2.flip(img, 1))

            # Pequenas rotações
            rows, cols = img.shape[:2]
            for angle in [-5, 5]:
                M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
                rotated = cv2.warpAffine(img, M, (cols, rows))
                augmented.append(rotated)

            # Pequeno zoom
            scale = 1.1
            M = cv2.getRotationMatrix2D((cols / 2, rows / 2), 0, scale)
            zoomed = cv2.warpAffine(img, M, (cols, rows))
            augmented.append(zoomed)

        except Exception as e:
            logging.debug(f"Aumento de dados parcialmente falhou: {str(e)}")
            augmented = [img]  # Fallback para imagem original

        return augmented

    def generate_embedding(self, img_path):
        try:
            full_path = os.path.join(self.UPLOADS_DIR, img_path.replace("\\", "/"))

            if not os.path.exists(full_path):
                logging.warning(
                    f"Arquivo não encontrado: {os.path.basename(full_path)}"
                )
                return None

            is_valid, msg = self.validate_image(full_path)
            if not is_valid:
                logging.warning(f"Imagem inválida: {msg}")
                return None

            # Usa detector mais preciso para treinamento
            try:
                embedding_objs = DeepFace.represent(
                    img_path=full_path,
                    model_name=self.EMBEDDING_MODEL,
                    detector_backend=self.DETECTOR,
                    enforce_detection=self.ENFORCE_DETECTION,
                    align=True,
                    normalization="base",
                )

                if embedding_objs and len(embedding_objs) > 0:
                    embedding = np.array(embedding_objs[0]["embedding"]).flatten()
                    # Normalização L2 consistente
                    embedding = self.normalizer.transform([embedding])[0]
                    logging.info("✓ Embedding gerado")
                    return embedding

            except Exception as e:
                logging.warning(f"Detector {self.DETECTOR} falhou: {str(e)}")
                return None

        except Exception as e:
            logging.error(f"Erro ao processar {img_path}: {str(e)}")
            return None

    def generate_embeddings_for_user(self, user_id, user_data):
        embeddings = []
        user_name = f"{user_data['nome']} {user_data['sobrenome']}"

        logging.info(f"Processando usuário: {user_name}")
        logging.info(f"Total de imagens: {len(user_data['images'])}")

        for i, img_path in enumerate(user_data["images"]):
            logging.info(f"Processando imagem {i+1}/{len(user_data['images'])}")

            embedding = self.generate_embedding(img_path)

            if embedding is not None:
                # Adiciona embedding original
                embeddings.append(embedding)
                self.training_stats["valid_images"] += 1
                logging.info(f"✓ Imagem {i+1}: Embedding gerado")

                # Aumento de dados para imagens únicas
                if len(user_data["images"]) == 1:  # Apenas para casos de 1 imagem
                    try:
                        full_path = os.path.join(
                            self.UPLOADS_DIR, img_path.replace("\\", "/")
                        )
                        img = cv2.imread(full_path)
                        if img is not None:
                            augmented_images = self.augment_image(img)

                            # Gera embeddings para imagens aumentadas
                            for j, aug_img in enumerate(
                                augmented_images[1:], 1
                            ):  # Pula a original
                                temp_path = (
                                    f"/tmp/aug_{j}_{os.path.basename(full_path)}"
                                )
                                cv2.imwrite(temp_path, aug_img)

                                aug_embedding = self.generate_embedding(temp_path)
                                if aug_embedding is not None:
                                    embeddings.append(aug_embedding)
                                    self.training_stats["valid_images"] += 1
                                    logging.info(
                                        f"✓ Imagem {i+1}: Embedding aumentado {j} gerado"
                                    )

                                # Remove arquivo temporário
                                if os.path.exists(temp_path):
                                    os.remove(temp_path)
                    except Exception as e:
                        logging.debug(f"Aumento de dados falhou: {str(e)}")
            else:
                self.training_stats["failed_images"] += 1
                logging.warning(f"✗ Imagem {i+1}: Falha")

        if embeddings:
            # Calcula embedding médio normalizado
            avg_embedding = np.mean(embeddings, axis=0)
            avg_embedding = self.normalizer.transform([avg_embedding])[0]

            logging.info(f"✓ {user_name}: {len(embeddings)} embeddings gerados")

            return {
                "nome": user_data["nome"],
                "sobrenome": user_data["sobrenome"],
                "embeddings": [e.tolist() for e in embeddings],
                "embedding": avg_embedding.tolist(),
                "embedding_count": len(embeddings),  # Para debug
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
                "version": "4.0",  # Atualizado
                "training_date": datetime.now().isoformat(),
                "embedding_model": self.EMBEDDING_MODEL,
                "detector": self.DETECTOR,
                "normalization": "l2",
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
        logging.info("\n=== INICIANDO TREINAMENTO OTIMIZADO PARA POUCAS IMAGENS ===")

        if not os.path.exists(self.UPLOADS_DIR) or not os.listdir(self.UPLOADS_DIR):
            logging.error("Pasta uploads não encontrada ou vazia")
            return False

        user_images = self.get_user_images()
        if not user_images:
            logging.error("Nenhuma imagem encontrada")
            return False

        logging.info(f"Usuários para processar: {len(user_images)}")
        self.training_stats["total_users"] = len(user_images)

        # Processar usuários
        processed_users = 0
        for user_id, user_data in user_images.items():
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
        total_time = (
            self.training_stats["end_time"] - self.training_stats["start_time"]
        ).total_seconds()

        print("\n" + "=" * 60)
        print("📊 RESUMO DO TREINAMENTO OTIMIZADO")
        print("=" * 60)
        print(f"   ⏰ Tempo total: {total_time:.1f} segundos")
        print(
            f"   👥 Usuários: {self.training_stats['processed_users']}/{self.training_stats['total_users']}"
        )
        print(f"   📷 Imagens válidas: {self.training_stats['valid_images']}")
        print(f"   ❌ Imagens falhas: {self.training_stats['failed_images']}")

        if self.training_stats["total_images"] > 0:
            success_rate = (
                self.training_stats["valid_images"]
                / self.training_stats["total_images"]
            ) * 100
            print(f"   🎯 Taxa de sucesso: {success_rate:.1f}%")

        print("=" * 60)

        for user_id, data in self.embeddings_db.items():
            print(
                f"   ✅ {data['nome']}: {data['embedding_count']} embeddings (com aumento)"
            )


if __name__ == "__main__":
    try:
        print("=" * 60)
        print("🤖 INICIANDO TREINAMENTO FACIAL OTIMIZADO")
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
