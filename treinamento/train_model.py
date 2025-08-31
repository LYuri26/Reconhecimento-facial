import os
import cv2
import numpy as np
from deepface import DeepFace
import mysql.connector
from mysql.connector import Error
import pickle
import time
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import sys


class DeepFaceTrainer:
    def __init__(self):
        # Configurações
        self.MIN_IMAGES_PER_USER = 1  # aceitar 1 imagem
        self.MAX_IMAGES_PER_USER = 20
        self.IMAGE_SIZE = (160, 160)
        self.MIN_FACE_SIZE = 100
        self.EMBEDDING_MODEL = "ArcFace"  # robusto para poucas imagens
        self.DETECTORS = ["retinaface", "ssd", "opencv"]
        self.THRESHOLD_BLUR = 50

        # Caminhos absolutos CORRIGIDOS
        self.SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
        self.PROJECT_ROOT = os.path.dirname(self.SCRIPT_DIR)  # Diretório do projeto
        self.MODEL_DIR = os.path.join(self.PROJECT_ROOT, "model")
        self.MODEL_PATH = os.path.join(self.MODEL_DIR, "deepface_model.pkl")
        self.UPLOADS_DIR = os.path.join(self.PROJECT_ROOT, "uploads")

        # Configuração de logging
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=[
                logging.FileHandler(os.path.join(self.PROJECT_ROOT, "training.log")),
                logging.StreamHandler(),
            ],
        )

        # Banco de dados
        self.DB_CONFIG = {
            "host": "localhost",
            "user": "root",
            "password": "",
            "database": "reconhecimento_facial",
        }

        self.embeddings_db = {}
        self.label_map = {}
        self.reverse_label_map = {}

        # Log dos caminhos para debug
        logging.info(f"SCRIPT_DIR: {self.SCRIPT_DIR}")
        logging.info(f"PROJECT_ROOT: {self.PROJECT_ROOT}")
        logging.info(f"MODEL_DIR: {self.MODEL_DIR}")
        logging.info(f"MODEL_PATH: {self.MODEL_PATH}")
        logging.info(f"UPLOADS_DIR: {self.UPLOADS_DIR}")

    # ------------------- Banco de Dados -------------------
    def create_connection(self):
        try:
            conn = mysql.connector.connect(**self.DB_CONFIG)
            if conn.is_connected():
                logging.info("Conexão com o banco de dados estabelecida com sucesso")
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
            cursor.execute("SHOW TABLES LIKE 'cadastros'")
            if not cursor.fetchone():
                logging.error("Tabela 'cadastros' não encontrada")
                return None
            cursor.execute("SHOW TABLES LIKE 'imagens_cadastro'")
            if not cursor.fetchone():
                logging.error("Tabela 'imagens_cadastro' não encontrada")
                return None

            cursor.execute("SELECT id, nome, sobrenome FROM cadastros")
            users = cursor.fetchall()
            if not users:
                logging.warning("Nenhum usuário encontrado")
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
                    logging.info(f"Usuário {user['nome']}: {len(images)} imagens")
                else:
                    logging.warning(f"Usuário {user['nome']}: sem imagens")

            return user_images
        except Error as e:
            logging.error(f"Erro ao buscar imagens: {e}")
            return None
        finally:
            if conn and conn.is_connected():
                conn.close()

    # ------------------- Validação de Imagens -------------------
    def validate_image(self, img_path):
        try:
            img = cv2.imread(img_path)
            if img is None:
                return False, "Não foi possível ler a imagem"
            if img.shape[0] < self.MIN_FACE_SIZE or img.shape[1] < self.MIN_FACE_SIZE:
                return False, "Imagem muito pequena"

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            fm = cv2.Laplacian(gray, cv2.CV_64F).var()
            if fm < self.THRESHOLD_BLUR:
                return False, f"Imagem borrada (score {fm:.1f})"

            face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            )
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)
            if len(faces) == 0:
                return False, "Nenhum rosto detectado"

            x, y, w, h = faces[0]
            if w * h / (img.shape[0] * img.shape[1]) < 0.1:
                return False, "Rosto muito pequeno"
            return True, "OK"
        except Exception as e:
            return False, f"Erro na validação: {str(e)}"

    # ------------------- Pré-processamento -------------------
    def preprocess_image(self, img_path):
        try:
            img = cv2.imread(img_path)
            if img is None:
                return None
            lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            limg = cv2.merge([clahe.apply(l), a, b])
            img = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
            img = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)
            img = cv2.resize(img, self.IMAGE_SIZE)
            return img
        except Exception as e:
            logging.error(f"Erro no pré-processamento de {img_path}: {str(e)}")
            return None

    # ------------------- Aumento de Dados -------------------
    def augment_image(self, img):
        augmented = [img]
        try:
            augmented.append(cv2.flip(img, 1))  # Flip horizontal
            rows, cols = img.shape[:2]
            for angle in [10, -10]:
                M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
                rotated = cv2.warpAffine(img, M, (cols, rows))
                augmented.append(rotated)
            for alpha in [0.8, 1.2]:
                adjusted = cv2.convertScaleAbs(img, alpha=alpha, beta=0)
                augmented.append(adjusted)
        except Exception as e:
            logging.warning(f"Erro no aumento de dados: {e}")
        return augmented

    # ------------------- Geração de Embeddings -------------------
    def generate_embedding(self, img_path):
        try:
            full_path = os.path.join(self.UPLOADS_DIR, img_path.replace("\\", "/"))
            print(f"Processando imagem: {os.path.basename(full_path)}")

            if not os.path.exists(full_path):
                print(f"✗ Arquivo não encontrado: {os.path.basename(full_path)}")
                return []

            is_valid, msg = self.validate_image(full_path)
            if not is_valid:
                print(f"✗ Imagem inválida ({msg}): {os.path.basename(full_path)}")
                return []

            img = self.preprocess_image(full_path)
            if img is None:
                return []

            # Gera embedding com detector mais preciso
            try:
                emb_obj = DeepFace.represent(
                    img_path=img,
                    model_name=self.EMBEDDING_MODEL,
                    enforce_detection=True,  # Mais rigoroso
                    detector_backend="retinaface",  # Mais preciso
                    align=True,
                    normalization="base",
                )
                if emb_obj:
                    e = np.array(emb_obj[0]["embedding"]).flatten()
                    e = e / np.linalg.norm(e)

                    # Verifica qualidade do embedding
                    if np.std(e) > 0.1:  # Embedding com boa variação
                        return [e]
                    else:
                        print(
                            f"✗ Embedding de baixa qualidade: {os.path.basename(full_path)}"
                        )
                        return []
            except Exception as e:
                print(f"✗ Erro no embedding: {str(e)}")
                return []

            return []
        except Exception as e:
            print(f"✗ Erro ao processar imagem: {str(e)}")
            return []

    def generate_embeddings_for_user(self, user_id, user_data):
        embeddings = []
        print(f"\nProcessando usuário: {user_data['nome']} {user_data['sobrenome']}")
        print(f"Total de imagens: {len(user_data['images'])}")

        if len(user_data["images"]) < self.MIN_IMAGES_PER_USER:
            print(f"⚠ Poucas imagens ({len(user_data['images'])})")

        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = [
                executor.submit(self.generate_embedding, img)
                for img in user_data["images"]
            ]

            for i, future in enumerate(as_completed(futures)):
                e_list = future.result()
                if e_list:
                    embeddings.extend(e_list)
                    print(
                        f"✓ Processada imagem {i+1}/{len(user_data['images'])} - {len(e_list)} embeddings"
                    )
                else:
                    print(f"✗ Falha na imagem {i+1}/{len(user_data['images'])}")

        if embeddings:
            avg_embedding = np.median(np.array(embeddings), axis=0)
            print(f"✓ Usuário processado - Total embeddings: {len(embeddings)}")
            return {
                "nome": user_data["nome"],
                "sobrenome": user_data["sobrenome"],
                "embeddings": [e.tolist() for e in embeddings],
                "embedding": avg_embedding.tolist(),
            }
        else:
            print(f"✗ Nenhum embedding válido para {user_data['nome']}")
            return None

    def generate_embeddings(self, user_images):
        logging.info("\nIniciando geração de embeddings...")
        start_time = time.time()

        embeddings_db = {}
        label_counter = 0
        processed_users = 0

        for user_id, user_data in user_images.items():
            self.label_map[user_id] = label_counter
            self.reverse_label_map[label_counter] = user_id
            label_counter += 1

            user_result = self.generate_embeddings_for_user(user_id, user_data)
            if user_result:
                embeddings_db[user_id] = user_result
                processed_users += 1
                logging.info(f"✓ Usuário {user_data['nome']} processado com sucesso")

        elapsed_time = time.time() - start_time
        logging.info(f"Tempo total de processamento: {elapsed_time:.2f} segundos")
        logging.info(f"Usuários processados: {processed_users}/{len(user_images)}")

        return embeddings_db

    # ------------------- Salvar Modelo -------------------
    def save_model(self):
        if not self.embeddings_db:
            logging.error("Nenhum dado para salvar")
            return False

        # Criar diretório do modelo se não existir
        os.makedirs(self.MODEL_DIR, exist_ok=True)

        model_data = {
            "embeddings_db": self.embeddings_db,
            "label_map": self.label_map,
            "reverse_label_map": self.reverse_label_map,
        }

        try:
            with open(self.MODEL_PATH, "wb") as f:
                pickle.dump(model_data, f)
            logging.info(f"✓ Modelo salvo com sucesso em: {self.MODEL_PATH}")

            # Verificar se o arquivo foi criado
            if os.path.exists(self.MODEL_PATH):
                file_size = os.path.getsize(self.MODEL_PATH)
                logging.info(f"Tamanho do arquivo: {file_size} bytes")
                return True
            else:
                logging.error("✗ Arquivo do modelo não foi criado")
                return False

        except Exception as e:
            logging.error(f"✗ Erro ao salvar modelo: {str(e)}")
            return False

    # ------------------- Treinamento -------------------
    def train(self):
        logging.info("\n=== Iniciando Treinamento DeepFace ===")

        # Verificar se a pasta uploads existe
        if not os.path.exists(self.UPLOADS_DIR):
            logging.error(f"✗ Pasta uploads não encontrada: {self.UPLOADS_DIR}")
            return False

        if not os.listdir(self.UPLOADS_DIR):
            logging.error("✗ Pasta uploads está vazia")
            return False

        user_images = self.get_user_images()
        if not user_images:
            logging.error("✗ Nenhuma imagem encontrada no banco de dados")
            return False

        logging.info(f"Usuários com imagens: {len(user_images)}")

        self.embeddings_db = self.generate_embeddings(user_images)
        if not self.embeddings_db:
            logging.error("✗ Falha na geração de embeddings")
            return False

        return self.save_model()


# ------------------- MAIN -------------------
if __name__ == "__main__":
    try:
        print("=" * 60)
        print("🤖 INICIANDO TREINAMENTO DO SISTEMA DE RECONHECIMENTO FACIAL")
        print("=" * 60)

        trainer = DeepFaceTrainer()

        if trainer.train():
            print("=" * 60)
            print("✅ TREINAMENTO CONCLUÍDO COM SUCESSO!")
            print("📊 Resumo:")
            print(f"   - Usuários processados: {len(trainer.embeddings_db)}")
            for user_id, data in trainer.embeddings_db.items():
                print(f"   - {data['nome']}: {len(data['embeddings'])} embeddings")
            print("=" * 60)
            # Forçar flush para garantir que a mensagem seja exibida
            sys.stdout.flush()
            exit(0)
        else:
            print("=" * 60)
            print("❌ FALHA NO TREINAMENTO")
            print("=" * 60)
            sys.stdout.flush()
            exit(1)

    except Exception as e:
        print("=" * 60)
        print("❌ ERRO CRÍTICO DURANTE O TREINAMENTO")
        print(f"   Erro: {str(e)}")
        print("=" * 60)
        sys.stdout.flush()
        exit(1)
