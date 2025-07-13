import os
import cv2
import numpy as np
from deepface import DeepFace
import mysql.connector
from mysql.connector import Error
import pickle
import time


class DeepFaceTrainer:
    def __init__(self):
        # Configurações
        self.MAX_IMAGES_PER_USER = 20
        self.IMAGE_SIZE = (160, 160)
        self.MODEL_PATH = "model/deepface_model.pkl"
        self.DB_CONFIG = {
            "host": "localhost",
            "user": "root",
            "password": "",
            "database": "reconhecimento_facial",
        }

        # Banco de dados de embeddings
        self.embeddings_db = {}
        self.label_map = {}
        self.reverse_label_map = {}

    def create_connection(self):
        """Cria conexão com o banco de dados"""
        try:
            return mysql.connector.connect(**self.DB_CONFIG)
        except Error as e:
            print(f"Erro ao conectar ao MySQL: {e}")
            return None

    def get_user_images(self):
        """Obtém imagens dos usuários do banco de dados"""
        conn = self.create_connection()
        if conn is None:
            return None

        try:
            cursor = conn.cursor(dictionary=True)
            cursor.execute("SELECT id, nome, sobrenome FROM cadastros")
            users = cursor.fetchall()

            user_images = {}

            for user in users:
                cursor.execute(
                    "SELECT caminho_imagem FROM imagens_cadastro WHERE cadastro_id = %s LIMIT %s",
                    (user["id"], self.MAX_IMAGES_PER_USER),
                )

                user_images[user["id"]] = {
                    "nome": user["nome"],
                    "sobrenome": user["sobrenome"],
                    "images": [img["caminho_imagem"] for img in cursor.fetchall()],
                }

            return user_images

        except Error as e:
            print(f"Erro ao buscar imagens: {e}")
            return None
        finally:
            if conn and conn.is_connected():
                conn.close()

    def generate_embeddings(self, user_images):
        """Gera embeddings faciais para todas as imagens"""
        print("\nGerando embeddings faciais...")
        start_time = time.time()

        embeddings_db = {}
        label_counter = 0

        for user_id, user_data in user_images.items():
            if not user_data["images"]:
                continue

            self.label_map[user_id] = label_counter
            self.reverse_label_map[label_counter] = user_id
            label_counter += 1

            embeddings = []
            for img_path in user_data["images"]:
                try:
                    full_path = os.path.join("uploads", img_path.replace("\\", "/"))
                    if not os.path.exists(full_path):
                        print(f"Arquivo não encontrado: {full_path}")
                        continue

                    # Extrair embedding usando represent
                    embedding_obj = DeepFace.represent(
                        img_path=full_path,
                        model_name="Facenet",
                        enforce_detection=True,
                        detector_backend="opencv",
                    )

                    if embedding_obj:
                        embedding = np.array(embedding_obj[0]["embedding"]).flatten()
                        embeddings.append(embedding)
                        print(
                            f"Embedding gerado para {img_path} - Shape: {embedding.shape}"
                        )

                except Exception as e:
                    print(f"Erro ao gerar embedding para {img_path}: {str(e)}")
                    continue

            if embeddings:
                avg_embedding = np.mean(embeddings, axis=0)
                embeddings_db[user_id] = {
                    "nome": user_data["nome"],
                    "sobrenome": user_data["sobrenome"],
                    "embedding": avg_embedding.tolist(),
                }
                print(
                    f"Embedding médio para {user_data['nome']} - Shape: {avg_embedding.shape}"
                )

        print(f"Embeddings gerados em {time.time()-start_time:.2f} segundos")
        return embeddings_db

    def save_model(self):
        """Salva o modelo e os mapeamentos"""
        model_data = {
            "embeddings_db": self.embeddings_db,
            "label_map": self.label_map,
            "reverse_label_map": self.reverse_label_map,
        }

        os.makedirs("model", exist_ok=True)
        with open(self.MODEL_PATH, "wb") as f:
            pickle.dump(model_data, f)

        print(f"\nModelo DeepFace salvo em: {self.MODEL_PATH}")

    def train(self):
        """Executa o treinamento completo"""
        print("\n=== Sistema de Treinamento com DeepFace ===")

        user_images = self.get_user_images()
        if not user_images:
            print("Nenhuma imagem encontrada para treinamento")
            return False

        self.embeddings_db = self.generate_embeddings(user_images)

        if not self.embeddings_db:
            print("Nenhum embedding facial gerado")
            return False

        self.save_model()
        return True


if __name__ == "__main__":
    trainer = DeepFaceTrainer()

    if not os.path.exists("uploads"):
        print("ERRO: Pasta 'uploads' não encontrada")
        exit(1)

    if trainer.train():
        print("\nTreinamento realizado com sucesso!")
    else:
        print("\nFalha no treinamento")
        exit(1)
