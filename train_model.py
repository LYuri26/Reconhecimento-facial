import os
import cv2
import numpy as np
from deepface import DeepFace
import mysql.connector
from mysql.connector import Error
import pickle
import time
import logging
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed


class DeepFaceTrainer:
    def __init__(self):
        # Configurações aprimoradas
        self.MAX_IMAGES_PER_USER = 20
        self.MIN_IMAGES_PER_USER = 5  # Mínimo recomendado
        self.IMAGE_SIZE = (160, 160)
        self.MODEL_PATH = "model/deepface_model.pkl"
        self.DB_CONFIG = {
            "host": "localhost",
            "user": "root",
            "password": "",
            "database": "reconhecimento_facial",
        }
        self.MIN_FACE_SIZE = 150  # Tamanho mínimo do rosto em pixels
        self.EMBEDDING_MODEL = "Facenet512"  # Modelo mais robusto
        self.DETECTORS = ["opencv", "ssd", "retinaface"]  # Backends para tentativa
        self.THRESHOLD_BLUR = 100  # Limiar para detecção de imagens borradas

        # Configuração de logging
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=[logging.FileHandler("training.log"), logging.StreamHandler()],
        )

        # Banco de dados de embeddings
        self.embeddings_db = {}
        self.label_map = {}
        self.reverse_label_map = {}

    def create_connection(self):
        """Cria conexão com o banco de dados com tratamento de erro aprimorado"""
        try:
            conn = mysql.connector.connect(**self.DB_CONFIG)
            if conn.is_connected():
                logging.info("Conexão com o banco de dados estabelecida com sucesso")
                return conn
        except Error as e:
            logging.error(f"Erro ao conectar ao MySQL: {e}")
            return None

    def get_user_images(self):
        """Obtém imagens dos usuários do banco de dados com verificação"""
        conn = self.create_connection()
        if conn is None:
            return None

        try:
            cursor = conn.cursor(dictionary=True)

            # Verificar se as tabelas existem
            cursor.execute("SHOW TABLES LIKE 'cadastros'")
            if not cursor.fetchone():
                logging.error("Tabela 'cadastros' não encontrada no banco de dados")
                return None

            cursor.execute("SHOW TABLES LIKE 'imagens_cadastro'")
            if not cursor.fetchone():
                logging.error(
                    "Tabela 'imagens_cadastro' não encontrada no banco de dados"
                )
                return None

            cursor.execute("SELECT id, nome, sobrenome FROM cadastros")
            users = cursor.fetchall()

            if not users:
                logging.warning("Nenhum usuário encontrado na tabela 'cadastros'")
                return None

            user_images = {}
            users_without_images = []

            for user in users:
                cursor.execute(
                    "SELECT caminho_imagem FROM imagens_cadastro WHERE cadastro_id = %s LIMIT %s",
                    (user["id"], self.MAX_IMAGES_PER_USER),
                )
                images = [img["caminho_imagem"] for img in cursor.fetchall()]

                if not images:
                    users_without_images.append(user["id"])
                    continue

                user_images[user["id"]] = {
                    "nome": user["nome"],
                    "sobrenome": user["sobrenome"],
                    "images": images,
                }

            if users_without_images:
                logging.warning(f"Usuários sem imagens: {users_without_images}")

            return user_images

        except Error as e:
            logging.error(f"Erro ao buscar imagens: {e}")
            return None
        finally:
            if conn and conn.is_connected():
                conn.close()

    def validate_image(self, img_path):
        """Validação rigorosa da imagem para garantir qualidade"""
        try:
            img = cv2.imread(img_path)
            if img is None:
                return False, "Não foi possível ler a imagem"

            # Verificar tamanho mínimo absoluto
            if img.shape[0] < self.MIN_FACE_SIZE or img.shape[1] < self.MIN_FACE_SIZE:
                return (
                    False,
                    f"Imagem muito pequena (min {self.MIN_FACE_SIZE}x{self.MIN_FACE_SIZE})",
                )

            # Verificar qualidade da imagem (nível de borrão)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            fm = cv2.Laplacian(gray, cv2.CV_64F).var()
            if fm < self.THRESHOLD_BLUR:
                return False, f"Imagem muito borrada (score {fm:.1f})"

            # Detecção de rosto mais precisa
            face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            )
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)

            if len(faces) == 0:
                return False, "Nenhum rosto detectado"

            # Verificar se o rosto ocupa pelo menos 15% da imagem
            x, y, w, h = faces[0]
            face_area = w * h
            img_area = img.shape[0] * img.shape[1]
            if face_area / img_area < 0.15:
                return False, "Rosto muito pequeno na imagem"

            return True, "OK"
        except Exception as e:
            return False, f"Erro na validação: {str(e)}"

    def preprocess_image(self, img_path):
        """Pré-processamento avançado para melhorar a qualidade da imagem"""
        try:
            img = cv2.imread(img_path)
            if img is None:
                return None

            # Equalização de histograma para melhorar contraste
            lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            limg = cv2.merge([clahe.apply(l), a, b])
            img = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

            # Redução de ruído
            img = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)

            # Redimensionamento padrão
            img = cv2.resize(img, self.IMAGE_SIZE)

            return img
        except Exception as e:
            logging.error(f"Erro no pré-processamento de {img_path}: {str(e)}")
            return None

    def generate_embedding(self, img_path, user_id):
        """Gera embedding para uma única imagem com tratamento de erro"""
        try:
            full_path = os.path.join("uploads", img_path.replace("\\", "/"))

            # Validação rigorosa
            is_valid, msg = self.validate_image(full_path)
            if not is_valid:
                logging.warning(f"Imagem inválida ({msg}): {full_path}")
                return None

            # Pré-processamento
            preprocessed_img = self.preprocess_image(full_path)
            if preprocessed_img is None:
                return None

            # Tentar com diferentes backends de detecção
            embedding = None

            for detector in self.DETECTORS:
                try:
                    embedding_obj = DeepFace.represent(
                        img_path=preprocessed_img,
                        model_name=self.EMBEDDING_MODEL,
                        enforce_detection=True,
                        detector_backend=detector,
                        align=True,
                    )
                    if embedding_obj:
                        embedding = np.array(embedding_obj[0]["embedding"]).flatten()
                        break
                except Exception as e:
                    continue

            if embedding is not None:
                logging.info(f"Embedding gerado com sucesso para {full_path}")
                return embedding
            else:
                logging.warning(f"Falha ao gerar embedding para: {full_path}")
                return None

        except Exception as e:
            logging.error(f"ERRO ao processar {img_path}: {str(e)}")
            return None

    def generate_embeddings_for_user(self, user_id, user_data):
        """Gera embeddings para um único usuário"""
        embeddings = []
        valid_images = 0

        logging.info(
            f"\nProcessando usuário ID: {user_id} - {user_data['nome']} {user_data['sobrenome']}"
        )
        logging.info(f"Total de imagens: {len(user_data['images'])}")

        if len(user_data["images"]) < self.MIN_IMAGES_PER_USER:
            logging.warning(
                f"Usuário {user_id} tem apenas {len(user_data['images'])} imagens (mínimo recomendado: {self.MIN_IMAGES_PER_USER})"
            )

        # Usando ThreadPool para processamento paralelo
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = []
            for img_path in user_data["images"]:
                futures.append(
                    executor.submit(self.generate_embedding, img_path, user_id)
                )

            for future in as_completed(futures):
                embedding = future.result()
                if embedding is not None:
                    embeddings.append(embedding)
                    valid_images += 1

        if embeddings:
            # Usar mediana para reduzir impacto de outliers
            avg_embedding = np.median(embeddings, axis=0)

            # Armazenar todos os embeddings individuais também
            return {
                "nome": user_data["nome"],
                "sobrenome": user_data["sobrenome"],
                "embeddings": [e.tolist() for e in embeddings],
                "embedding": avg_embedding.tolist(),
            }
        else:
            logging.warning(f"Nenhum embedding válido gerado para {user_data['nome']}")
            return None

    def generate_embeddings(self, user_images):
        """Gera embeddings faciais para todas as imagens com paralelismo"""
        logging.info("\nIniciando geração de embeddings faciais...")
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

                logging.info(f"\nResumo para {user_data['nome']}:")
                logging.info(
                    f"- Imagens processadas com sucesso: {len(user_result['embeddings'])}/{len(user_data['images'])}"
                )
                logging.info(
                    f"- Embedding médio shape: {len(user_result['embedding'])}"
                )

        logging.info(f"\nEmbeddings gerados em {time.time()-start_time:.2f} segundos")
        logging.info(
            f"Total de usuários processados: {processed_users}/{len(user_images)}"
        )

        if processed_users == 0:
            logging.error("Nenhum usuário válido foi processado")
            return None

        return embeddings_db

    def save_model(self):
        """Salva o modelo e os mapeamentos com verificação"""
        if not self.embeddings_db:
            logging.error("Nenhum dado para salvar - embeddings_db está vazio")
            return False

        model_data = {
            "embeddings_db": self.embeddings_db,
            "label_map": self.label_map,
            "reverse_label_map": self.reverse_label_map,
        }

        try:
            os.makedirs("model", exist_ok=True)
            with open(self.MODEL_PATH, "wb") as f:
                pickle.dump(model_data, f)

            logging.info(f"\nModelo DeepFace salvo em: {self.MODEL_PATH}")
            return True
        except Exception as e:
            logging.error(f"Erro ao salvar modelo: {str(e)}")
            return False

    def train(self):
        """Executa o treinamento completo com tratamento de erro robusto"""
        logging.info("\n=== Sistema de Treinamento com DeepFace ===")

        logging.info("Obtendo imagens do banco de dados...")
        user_images = self.get_user_images()
        if not user_images:
            logging.error("Nenhuma imagem encontrada para treinamento")
            return False

        logging.info("Gerando embeddings faciais...")
        self.embeddings_db = self.generate_embeddings(user_images)
        if not self.embeddings_db:
            logging.error("Nenhum embedding facial gerado")
            return False

        if not self.save_model():
            return False

        return True


if __name__ == "__main__":
    try:
        # Verificar se a pasta uploads existe
        if not os.path.exists("uploads"):
            logging.error("ERRO: Pasta 'uploads' não encontrada")
            exit(1)

        # Verificar se há subpastas/arquivos em uploads
        if not os.listdir("uploads"):
            logging.error("ERRO: Pasta 'uploads' está vazia")
            exit(1)

        trainer = DeepFaceTrainer()

        if trainer.train():
            logging.info("\nTreinamento realizado com sucesso!")
            exit(0)
        else:
            logging.error("\nFalha no treinamento")
            exit(1)

    except Exception as e:
        logging.error(f"\nERRO CRÍTICO: {str(e)}")
        exit(1)
