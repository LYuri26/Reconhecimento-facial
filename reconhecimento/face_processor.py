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
    def __init__(self, model_path="model/deepface_model.pkl", threshold=0.45):
        self.MODEL_PATH = model_path
        self.THRESHOLD = threshold  # Threshold mais alto para evitar falsos positivos
        self.EMBEDDING_MODEL = "Facenet512"
        self.DETECTOR = "opencv"  # Mais rápido para tempo real
        self.model_loaded = False
        self.embeddings_db = {}
        self.db_connection = None

        # Otimizações de performance para tempo real
        self.last_processed_time = 0
        self.processing_interval = 0.15  # Processa a cada 150ms (~6-7 FPS)
        self.cache_size = 3  # Cache menor para economizar memória
        self.embedding_cache = {}
        self.last_faces = []
        self.skip_frames = 3  # Pula mais frames para melhor performance
        self.frame_count = 0
        self.min_face_size = (
            70  # Tamanho mínimo reduzido para detectar rostos mais distantes
        )

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

                # Normaliza os embeddings
                for user_id in self.embeddings_db:
                    if isinstance(self.embeddings_db[user_id]["embedding"], list):
                        self.embeddings_db[user_id]["embedding"] = np.array(
                            self.embeddings_db[user_id]["embedding"], dtype=np.float32
                        )
                    # Normalização L2
                    norm = np.linalg.norm(self.embeddings_db[user_id]["embedding"])
                    if norm > 0:
                        self.embeddings_db[user_id]["embedding"] /= norm

                logging.info(f"Modelo carregado com {len(self.embeddings_db)} pessoas")
                self.model_loaded = True
                return True

        except Exception as e:
            logging.error(f"Erro ao carregar o modelo: {str(e)}")
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
                connect_timeout=5,  # Timeout para não travar
            )
            logging.info("Conexão com o banco estabelecida")
            return True
        except Error as e:
            logging.warning(f"Erro na conexão com o banco: {str(e)}")
            self.db_connection = None
            return False

    def get_user_info(self, user_id):
        """Obtém informações do usuário do banco de dados"""
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

    def calculate_similarity(self, emb1, emb2):
        """Calcula similaridade entre embeddings de forma precisa e eficiente"""
        try:
            if emb1 is None or emb2 is None:
                return 0.0

            # Garante que os embeddings são arrays numpy
            emb1 = np.array(emb1, dtype=np.float32)
            emb2 = np.array(emb2, dtype=np.float32)

            # Verifica se os embeddings são válidos
            if np.all(emb1 == 0) or np.all(emb2 == 0):
                return 0.0

            # Normalização L2
            emb1_norm = emb1 / (np.linalg.norm(emb1) + 1e-10)
            emb2_norm = emb2 / (np.linalg.norm(emb2) + 1e-10)

            # Calcula a similaridade cosseno (já normalizada entre -1 e 1)
            similarity = np.dot(emb1_norm, emb2_norm)

            # Converte para escala 0-1
            similarity = (similarity + 1) / 2

            return max(0.0, min(1.0, similarity))

        except Exception as e:
            logging.debug(f"Erro no cálculo de similaridade: {str(e)}")
            return 0.0

    def safe_get_embedding(self, face_img):
        """Gera embedding facial otimizado para tempo real"""
        try:
            if not isinstance(face_img, np.ndarray) or face_img.size == 0:
                return None

            # Verifica tamanho mínimo
            if (
                face_img.shape[0] < self.min_face_size
                or face_img.shape[1] < self.min_face_size
            ):
                return None

            # Gera hash da imagem para cache
            img_hash = hash(face_img.tobytes())

            # Verifica se já está em cache
            if img_hash in self.embedding_cache:
                return self.embedding_cache[img_hash]

            # Converte para RGB
            rgb_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)

            # Gera o embedding com configurações otimizadas
            embedding_obj = DeepFace.represent(
                img_path=rgb_img,
                model_name=self.EMBEDDING_MODEL,
                enforce_detection=False,  # Não falha se não detectar rostos
                detector_backend="skip",  # Pula detecção (já detectamos o rosto)
                normalization="base",
            )

            if not embedding_obj or not isinstance(embedding_obj, list):
                return None

            embedding = np.array(embedding_obj[0]["embedding"], dtype=np.float32)

            # Verifica se o embedding é válido
            if np.all(embedding == 0):
                return None

            # Normalização
            norm = np.linalg.norm(embedding)
            if norm == 0:
                return None

            embedding = embedding / norm

            # Adiciona ao cache
            if len(self.embedding_cache) >= self.cache_size:
                self.embedding_cache.pop(next(iter(self.embedding_cache)))
            self.embedding_cache[img_hash] = embedding

            return embedding

        except Exception as e:
            logging.debug(f"Erro na geração de embedding: {str(e)}")
            return None

    def detect_faces_fast(self, frame):
        """Detecção de faces otimizada para tempo real"""
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
                detected_faces.append(
                    {
                        "x": x,
                        "y": y,
                        "w": w,
                        "h": h,
                        "confidence": 1.0,  # Haar não retorna confidence
                    }
                )

            return detected_faces

        except Exception as e:
            logging.error(f"Erro na detecção de faces: {str(e)}")
            return []

    def process_frame(self, frame):
        """Processa um frame para reconhecimento facial otimizado"""
        try:
            if frame is None or frame.size == 0:
                return frame

            display_frame = frame.copy()
            current_time = time.time()

            # Controle de frames para pular processamento
            self.frame_count += 1
            if self.frame_count % (self.skip_frames + 1) != 0:
                # Reutiliza detecção anterior se disponível
                if self.last_faces:
                    for face_data in self.last_faces:
                        x, y, w, h = (
                            face_data["x"],
                            face_data["y"],
                            face_data["w"],
                            face_data["h"],
                        )
                        label = face_data.get("label", "Processando...")
                        color = face_data.get("color", (0, 255, 255))

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

            # VERIFICA SE A CÂMERA ESTÁ TAMPADA
            if self.is_camera_covered(frame):
                cv2.putText(
                    display_frame,
                    "CAMERA TAMPADA/ESCURA",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2,
                )
                return display_frame

            # Limita a taxa de processamento
            if current_time - self.last_processed_time < self.processing_interval:
                return display_frame

            self.last_processed_time = current_time
            self.last_faces = []

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

            # Detecta faces (método rápido)
            faces = self.detect_faces_fast(frame)
            if not faces:
                return display_frame

            for face in faces:
                x, y, w, h = face["x"], face["y"], face["w"], face["h"]

                # Extrai a região do rosto
                face_region = frame[y : y + h, x : x + w]
                if face_region.size == 0:
                    continue

                # Gera embedding (processamento mais leve)
                live_emb = self.safe_get_embedding(face_region)
                if live_emb is None:
                    label = "Analisando..."
                    color = (0, 255, 255)
                    self.last_faces.append(
                        {"x": x, "y": y, "w": w, "h": h, "label": label, "color": color}
                    )
                    continue

                best_match = None
                best_score = 0

                # Comparação rápida com embeddings do banco
                for user_id, user_data in self.embeddings_db.items():
                    score = self.calculate_similarity(live_emb, user_data["embedding"])
                    if score > best_score:
                        best_score = score
                        best_match = user_id

                # Verificação conservadora para evitar falsos positivos
                if best_match and best_score >= self.THRESHOLD:
                    user_info = self.get_user_info(best_match)
                    # Exibe apenas o primeiro nome para economizar espaço
                    label = f"{user_info['nome']} ({best_score:.2f})"
                    color = (0, 255, 0)
                else:
                    label = "Não identificado"
                    color = (0, 0, 255)

                # Armazena para reutilização
                self.last_faces.append(
                    {"x": x, "y": y, "w": w, "h": h, "label": label, "color": color}
                )

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

    def is_camera_covered(self, frame):
        """Detecta se a câmera está tampada/escura"""
        try:
            if frame is None or frame.size == 0:
                return True

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Calcula estatísticas simples para performance
            mean_intensity = np.mean(gray)
            std_intensity = np.std(gray)

            # Critérios simplificados
            camera_covered = (
                mean_intensity < 30  # Muito escuro
                or std_intensity < 15  # Pouca variação (imagem uniforme)
            )

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
