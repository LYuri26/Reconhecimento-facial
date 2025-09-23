import cv2
import numpy as np
import pickle
import logging
import os
import time
from deepface import DeepFace
from mysql.connector import Error
import mysql.connector


class FaceProcessor:
    def __init__(
        self, model_path="model/deepface_model.pkl", threshold=0.5
    ):  # Threshold reduzido
        self.MODEL_PATH = model_path
        self.THRESHOLD = threshold  # Threshold mais baixo para Facenet512
        self.EMBEDDING_MODEL = "Facenet512"
        self.DETECTOR = "opencv"
        self.model_loaded = False
        self.embeddings_db = {}
        self.db_connection = None

        # Otimizações de performance
        self.last_processed_time = 0
        self.processing_interval = 0.3  # Aumentei o intervalo para melhor qualidade
        self.min_face_size = 80

        # Carrega o modelo na inicialização
        self.load_model()
        self.initialize_database()

    def load_model(self):
        """Carrega o modelo de reconhecimento facial"""
        try:
            if not os.path.exists(self.MODEL_PATH):
                logging.warning(
                    "Arquivo do modelo não encontrado. Sistema sem treinamento."
                )
                self.model_loaded = False
                return False

            with open(self.MODEL_PATH, "rb") as f:
                data = pickle.load(f)
                self.embeddings_db = data["embeddings_db"]

            logging.info(f"✅ Modelo carregado com {len(self.embeddings_db)} pessoas")

            # Log detalhado das pessoas carregadas
            for user_id, data in self.embeddings_db.items():
                logging.info(
                    f"   👤 {data['nome']} - Embeddings: {len(data.get('embeddings', []))}"
                )

            self.model_loaded = True
            return True

        except Exception as e:
            logging.error(f"❌ Erro ao carregar o modelo: {str(e)}")
            self.model_loaded = False
            return False

    def initialize_database(self):
        """Conecta ao banco de dados MySQL"""
        try:
            self.db_connection = mysql.connector.connect(
                host="localhost",
                user="root",
                password="",
                database="reconhecimento_facial",
                autocommit=True,
                connect_timeout=5,
            )
            logging.info("✅ Conexão com o banco estabelecida")
            return True
        except Error as e:
            logging.warning(f"⚠️ Erro na conexão com o banco: {str(e)}")
            self.db_connection = None
            return False

    def calculate_similarity(self, emb1, emb2):
        """Calcula similaridade cosseno corretamente"""
        try:
            if emb1 is None or emb2 is None:
                return 0.0

            # Converte para arrays numpy
            emb1 = np.array(emb1, dtype=np.float32).flatten()
            emb2 = np.array(emb2, dtype=np.float32).flatten()

            # Verifica se os embeddings são válidos
            if (
                emb1.size == 0
                or emb2.size == 0
                or np.all(emb1 == 0)
                or np.all(emb2 == 0)
                or np.isnan(emb1).any()
                or np.isnan(emb2).any()
            ):
                return 0.0

            # Normalização L2 (igual ao treinamento)
            norm1 = np.linalg.norm(emb1)
            norm2 = np.linalg.norm(emb2)

            if norm1 == 0 or norm2 == 0:
                return 0.0

            emb1_normalized = emb1 / norm1
            emb2_normalized = emb2 / norm2

            # Similaridade cosseno (já entre 0-1 para Facenet512 normalizado)
            similarity = np.dot(emb1_normalized, emb2_normalized)

            logging.debug(f"Similaridade calculada: {similarity:.4f}")
            return float(np.clip(similarity, 0.0, 1.0))

        except Exception as e:
            logging.debug(f"Erro no cálculo de similaridade: {str(e)}")
            return 0.0

    def safe_get_embedding(self, face_img):
        """Gera embedding facial de forma consistente com o treinamento"""
        try:
            if not isinstance(face_img, np.ndarray) or face_img.size == 0:
                return None

            # Verifica tamanho mínimo
            if (
                face_img.shape[0] < self.min_face_size
                or face_img.shape[1] < self.min_face_size
            ):
                return None

            # Salva temporariamente a imagem para processamento
            temp_path = "/tmp/temp_face.jpg"
            cv2.imwrite(temp_path, face_img)

            # Usa o MESMO método do treinamento
            embedding_objs = DeepFace.represent(
                img_path=temp_path,
                model_name=self.EMBEDDING_MODEL,
                detector_backend="opencv",  # Usa opencv para consistência
                enforce_detection=False,
                align=True,
                normalization="base",
            )

            # Remove arquivo temporário
            if os.path.exists(temp_path):
                os.remove(temp_path)

            if not embedding_objs or not isinstance(embedding_objs, list):
                return None

            embedding = np.array(embedding_objs[0]["embedding"], dtype=np.float32)

            # Normalização igual ao treinamento
            norm = np.linalg.norm(embedding)
            if norm == 0:
                return None

            embedding = embedding / norm
            return embedding

        except Exception as e:
            logging.debug(f"Erro na geração de embedding: {str(e)}")
            return None

    def detect_faces_fast(self, frame):
        """Detecção de faces otimizada"""
        try:
            # Usa detector Haar Cascade para maior velocidade
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            )

            faces = face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(self.min_face_size, self.min_face_size),
            )

            detected_faces = []
            for x, y, w, h in faces:
                detected_faces.append({"x": x, "y": y, "w": w, "h": h})

            return detected_faces

        except Exception as e:
            logging.error(f"Erro na detecção de faces: {str(e)}")
            return []

    def recognize_face(self, live_embedding):
        """Reconhece uma face baseada no embedding"""
        try:
            if not self.model_loaded or not self.embeddings_db:
                return None, 0.0

            best_match = None
            best_similarity = 0.0

            for user_id, user_data in self.embeddings_db.items():
                # Usa o embedding médio (principal) para comparação
                known_embedding = user_data.get("embedding")

                if known_embedding is None:
                    continue

                similarity = self.calculate_similarity(live_embedding, known_embedding)

                logging.info(
                    f"Comparando: {user_data['nome']} - Similaridade: {similarity:.4f}"
                )

                if similarity > best_similarity:
                    best_similarity = similarity
                    best_match = user_id

            # Verifica se atingiu o threshold
            if best_match and best_similarity >= self.THRESHOLD:
                logging.info(
                    f"✅ MATCH: {best_match} - Similaridade: {best_similarity:.4f}"
                )
                return best_match, best_similarity
            else:
                if best_match:
                    logging.info(
                        f"❌ Threshold não atingido: {best_similarity:.4f} < {self.THRESHOLD}"
                    )
                return None, best_similarity

        except Exception as e:
            logging.error(f"Erro no reconhecimento: {str(e)}")
            return None, 0.0

    def process_frame(self, frame):
        """Processa um frame para reconhecimento facial"""
        try:
            if frame is None or frame.size == 0:
                return frame

            display_frame = frame.copy()
            current_time = time.time()

            # Limita a taxa de processamento
            if current_time - self.last_processed_time < self.processing_interval:
                return display_frame

            self.last_processed_time = current_time

            # Verifica se o modelo foi carregado
            if not self.model_loaded:
                cv2.putText(
                    display_frame,
                    "SISTEMA SEM TREINAMENTO",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2,
                )
                return display_frame

            # Detecta faces
            faces = self.detect_faces_fast(frame)
            if not faces:
                return display_frame

            for face in faces:
                x, y, w, h = face["x"], face["y"], face["w"], face["h"]

                # Extrai a região do rosto
                face_region = frame[y : y + h, x : x + w]
                if face_region.size == 0:
                    continue

                # Gera embedding
                live_emb = self.safe_get_embedding(face_region)
                if live_emb is None:
                    label = "Analisando..."
                    color = (0, 255, 255)
                else:
                    # Reconhece a face
                    user_id, similarity = self.recognize_face(live_emb)

                    if user_id:
                        user_info = self.get_user_info(user_id)
                        label = f"{user_info['nome']} ({similarity:.3f})"
                        color = (0, 255, 0)  # Verde para reconhecido
                    else:
                        label = (
                            f"Não identificado ({similarity:.3f})"
                            if similarity > 0
                            else "Não identificado"
                        )
                        color = (0, 0, 255)  # Vermelho para não reconhecido

                # Desenha retângulo e label
                cv2.rectangle(display_frame, (x, y), (x + w, y + h), color, 2)
                text_y = max(y - 10, 20)
                cv2.putText(
                    display_frame,
                    label,
                    (x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    color,
                    2,
                )

            return display_frame

        except Exception as e:
            logging.error(f"Erro no processamento do frame: {str(e)}")
            return frame

    def get_user_info(self, user_id):
        """Obtém informações do usuário"""
        if not self.db_connection:
            return {"nome": "Desconhecido", "sobrenome": ""}

        try:
            cursor = self.db_connection.cursor(dictionary=True)
            cursor.execute(
                "SELECT nome, sobrenome FROM cadastros WHERE id = %s", (user_id,)
            )
            result = cursor.fetchone()
            return result or {"nome": "Desconhecido", "sobrenome": ""}
        except Error as e:
            logging.error(f"Erro ao buscar usuário: {str(e)}")
            return {"nome": "Desconhecido", "sobrenome": ""}
        finally:
            if "cursor" in locals():
                cursor.close()

    def is_camera_covered(self, frame):
        """Detecta se a câmera está tampada/escura"""
        try:
            if frame is None or frame.size == 0:
                return True

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            mean_intensity = np.mean(gray)
            std_intensity = np.std(gray)

            camera_covered = mean_intensity < 30 or std_intensity < 15
            return camera_covered

        except Exception as e:
            logging.error(f"Erro ao verificar câmera tampada: {str(e)}")
            return False

    def cleanup(self):
        """Limpeza de recursos"""
        if self.db_connection:
            try:
                self.db_connection.close()
            except Exception as e:
                logging.error(f"Erro ao fechar conexão com o banco: {str(e)}")

        logging.info("Processador facial finalizado")
