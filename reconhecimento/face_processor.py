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
        self.THRESHOLD = threshold
        self.EMBEDDING_MODEL = "Facenet512"
        self.DETECTOR = "opencv"  # Usar opencv que é mais rápido
        self.model_loaded = False
        self.embeddings_db = {}
        self.db_connection = None

        # Otimizações de performance
        self.last_processed_time = 0
        self.processing_interval = 0.3  # Processa a cada 300ms (3 FPS)
        self.cache_size = 5
        self.embedding_cache = {}
        self.last_faces = []

        # Carrega o modelo na inicialização
        self.load_model()
        self.initialize_database()
        self.processing_interval = 0.2  # Reduzido para 200ms (5 FPS)
        self.skip_frames = 2  # Pula 2 frames entre processamentos
        self.frame_count = 0

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
        """Calcula similaridade entre embeddings corretamente"""
        try:
            if emb1 is None or emb2 is None:
                return 0.0

            # Garante que os embeddings são arrays numpy
            emb1 = np.array(emb1, dtype=np.float32)
            emb2 = np.array(emb2, dtype=np.float32)

            # Normalização L2 correta
            emb1_norm = emb1 / np.linalg.norm(emb1)
            emb2_norm = emb2 / np.linalg.norm(emb2)

            # Calcula a similaridade cosseno (já normalizada entre -1 e 1)
            similarity = np.dot(emb1_norm, emb2_norm)

            # Ajusta para ficar entre 0 e 1 (mais intuitivo)
            similarity = (similarity + 1) / 2

            return max(0.0, min(1.0, similarity))

        except Exception as e:
            logging.error(f"Erro no cálculo de similaridade: {str(e)}")
            return 0.0

    def safe_get_embedding(self, face_img):
        """Gera embedding facial com cache para melhor performance"""
        try:
            if not isinstance(face_img, np.ndarray) or face_img.size == 0:
                return None

            # Verifica tamanho mínimo
            if face_img.shape[0] < 50 or face_img.shape[1] < 50:
                return None

            # Gera hash da imagem para cache
            img_hash = hash(face_img.tobytes())

            # Verifica se já está em cache
            if img_hash in self.embedding_cache:
                return self.embedding_cache[img_hash]

            # Converte para RGB
            rgb_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)

            # Gera o embedding (mais rápido com enforce_detection=False)
            embedding_obj = DeepFace.represent(
                img_path=rgb_img,
                model_name=self.EMBEDDING_MODEL,
                enforce_detection=False,  # Não tenta detectar rosto novamente
                detector_backend="skip",  # Pula detecção, já temos o rosto
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

            # Adiciona ao cache (mantém tamanho limitado)
            if len(self.embedding_cache) >= self.cache_size:
                self.embedding_cache.pop(next(iter(self.embedding_cache)))
            self.embedding_cache[img_hash] = embedding

            return embedding

        except Exception as e:
            logging.error(f"Erro na geração de embedding: {str(e)}")
            return None

    def detect_faces_fast(self, frame):
        """Detecção de faces otimizada"""
        try:
            # Usa o detector do DeepFace mas com configurações mais rápidas
            faces = DeepFace.extract_faces(
                img_path=frame,
                detector_backend=self.DETECTOR,
                enforce_detection=False,  # Não falha se não detectar rostos
                align=False,  # Não alinha para economizar processamento
            )

            if not isinstance(faces, list):
                return []

            detected_faces = []
            for face in faces:
                if face and face.get("facial_area"):
                    area = face["facial_area"]
                    detected_faces.append(
                        {
                            "x": area["x"],
                            "y": area["y"],
                            "w": area["w"],
                            "h": area["h"],
                            "confidence": area.get("confidence", 0),
                        }
                    )

            return detected_faces

        except Exception as e:
            logging.error(f"Erro na detecção de faces: {str(e)}")
            return []

    def process_frame(self, frame):
        """Processa um frame para reconhecimento facial com otimizações"""
        try:
            if frame is None or frame.size == 0:
                return frame

            # Controle de frames para pular processamento
            self.frame_count += 1
            if self.frame_count % (self.skip_frames + 1) != 0:
                # Reutiliza detecção anterior se disponível
                if self.last_faces:
                    display_frame = frame.copy()
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
                return frame

            display_frame = frame.copy()
            current_time = time.time()

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

            # Limita a taxa de processamento para melhor performance
            if current_time - self.last_processed_time < self.processing_interval:
                # Reutiliza a detecção anterior se disponível
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

            self.last_processed_time = current_time
            self.last_faces = []  # Limpa faces anteriores

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
                    # Mostra que está detectando mas não reconheceu ainda
                    label = "Detectado..."
                    color = (0, 255, 255)  # Amarelo
                    self.last_faces.append(
                        {"x": x, "y": y, "w": w, "h": h, "label": label, "color": color}
                    )
                    continue

                best_match = None
                best_score = 0

                # Compara com embeddings do banco
                for user_id, user_data in self.embeddings_db.items():
                    score = self.calculate_similarity(live_emb, user_data["embedding"])

                    if score > best_score and score > self.THRESHOLD:
                        best_score = score
                        best_match = user_id

                # Define label e cor
                label = "Nao identificado"
                color = (0, 0, 255)  # Vermelho

                if best_match:
                    user_info = self.get_user_info(best_match)
                    label = f"{user_info['nome']} {user_info['sobrenome']} ({best_score:.2f})"
                    color = (0, 255, 0)  # Verde
                    logging.info(f"Reconhecido: {label}")

                # Armazena para reutilização no próximo frame
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
            # Em caso de erro, retorna o frame original sem processamento
            return frame

    def is_camera_covered(self, frame):
        """Detecta se a câmera está tampada/escura com múltiplas verificações"""
        try:
            if frame is None or frame.size == 0:
                return True

            # Converte para escala de cinza
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Calcula estatísticas da imagem
            mean_intensity = np.mean(gray)  # Brilho médio (0-255)
            variance = np.var(gray)  # Variância (contraste)
            min_val = np.min(gray)  # Valor mínimo
            max_val = np.max(gray)  # Valor máximo

            # Calcula histograma para análise de distribuição
            hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
            hist = hist.flatten()

            # Verifica se a imagem está predominantemente escura
            dark_pixels = np.sum(hist[:50])  # Pixels com valor 0-50
            total_pixels = frame.shape[0] * frame.shape[1]
            dark_ratio = dark_pixels / total_pixels

            # Critérios para considerar câmera tampada:
            # 1. Brilho médio muito baixo
            # 2. Pouca variação (imagem uniforme)
            # 3. Alta porcentagem de pixels escuros
            # 4. Baixa diferença entre min e max (pouco contraste)

            camera_covered = (
                mean_intensity < 25  # Muito escuro
                or variance < 50  # Pouca variação
                or dark_ratio > 0.8  # Mais de 80% pixels escuros
                or (max_val - min_val) < 30  # Pouco contraste
            )

            # Log para debugging (opcional)
            if camera_covered:
                logging.debug(
                    f"Câmera tampada detectada - "
                    f"Brilho: {mean_intensity:.1f}, "
                    f"Variância: {variance:.1f}, "
                    f"Escuros: {dark_ratio:.2%}"
                )

            return camera_covered

        except Exception as e:
            logging.error(f"Erro ao verificar câmera tampada: {str(e)}")
            return False

    def cleanup(self):
        """Limpeza de recursos"""
        # Fecha a conexão com o banco
        if self.db_connection:
            try:
                self.db_connection.close()
            except Exception as e:
                logging.error(f"Erro ao fechar conexão com o banco: {str(e)}")

        logging.info("Processador facial finalizado")
