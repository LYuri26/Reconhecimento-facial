# face_processor.py - OTIMIZADO PARA DESEMPENHO
import cv2
import numpy as np
import pickle
import logging
import os
import time
from deepface import DeepFace
from mysql.connector import Error
import mysql.connector
from sklearn.preprocessing import Normalizer

try:
    from .expression_analyzer import ExpressionAnalyzer
except ImportError:
    # Fallback para desenvolvimento
    from expression_analyzer import ExpressionAnalyzer


class FaceProcessor:
    def __init__(self, model_path="model/deepface_model.pkl", threshold=0.65):
        self.MODEL_PATH = model_path
        self.THRESHOLD = threshold  # Otimizado para Facenet512
        self.EMBEDDING_MODEL = "Facenet512"
        self.DETECTOR = "opencv"
        self.model_loaded = False
        self.embeddings_db = {}
        self.db_connection = None

        # Normalizador consistente com treinamento
        self.normalizer = Normalizer(norm="l2")

        # OTIMIZAÇÕES DE PERFORMANCE
        self.last_processed_time = 0
        self.processing_interval = 0.5  # Aumentado de 0.2 para 0.5 segundos
        self.min_face_size = 60  # Reduzido de 80 para 60 (mais faces)
        self.face_cache = {}  # Cache de embeddings para performance
        self.cache_timeout = 30  # segundos

        # Analise de expressoes - desativada por padrão
        self.expression_analyzer = ExpressionAnalyzer()
        self.enable_expression_analysis = False  # Padrão desligado
        self.expression_results = {}
        self.last_expression_analysis = 0
        self.expression_analysis_interval = (
            1.0  # Aumentado de 0.5 para 1.0 (menos frequente)
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

            logging.info(f"✅ Modelo carregado com {len(self.embeddings_db)} pessoas")

            # Pré-calcula embeddings normalizados
            for user_id, data in self.embeddings_db.items():
                if "embedding" in data:
                    data["normalized_embedding"] = self.normalizer.transform(
                        [np.array(data["embedding"])]
                    )[0]

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
        """Calcula similaridade cosseno otimizada"""
        try:
            if emb1 is None or emb2 is None:
                return 0.0

            # Garante que são arrays numpy
            emb1 = np.array(emb1, dtype=np.float32).flatten()
            emb2 = np.array(emb2, dtype=np.float32).flatten()

            # Verifica validade
            if (
                emb1.size == 0
                or emb2.size == 0
                or np.all(emb1 == 0)
                or np.all(emb2 == 0)
                or np.isnan(emb1).any()
                or np.isnan(emb2).any()
            ):
                return 0.0

            # Similaridade cosseno otimizada
            similarity = np.dot(emb1, emb2) / (
                np.linalg.norm(emb1) * np.linalg.norm(emb2)
            )

            return float(np.clip(similarity, 0.0, 1.0))

        except Exception as e:
            logging.debug(f"Erro no cálculo de similaridade: {str(e)}")
            return 0.0

    def safe_get_embedding(self, face_img):
        """Gera embedding facial otimizado - sem arquivos temporários e com detector='skip'"""
        try:
            if not isinstance(face_img, np.ndarray) or face_img.size == 0:
                return None

            # Verifica tamanho mínimo
            if (
                face_img.shape[0] < self.min_face_size
                or face_img.shape[1] < self.min_face_size
            ):
                return None

            # Cache key baseado na imagem
            cache_key = hash(face_img.tobytes())
            current_time = time.time()

            # Verifica cache
            if (
                cache_key in self.face_cache
                and current_time - self.face_cache[cache_key]["timestamp"]
                < self.cache_timeout
            ):
                return self.face_cache[cache_key]["embedding"]

            # OTIMIZAÇÃO: usar detector_backend='skip' para evitar nova detecção e não usar arquivo temporário
            embedding_objs = DeepFace.represent(
                img_path=face_img,  # passa array numpy diretamente
                model_name=self.EMBEDDING_MODEL,
                detector_backend="skip",  # pula detecção, usa a imagem como está
                enforce_detection=False,
                align=False,  # alinhamento requer detecção, desliga
                normalization="base",
            )

            if not embedding_objs or not isinstance(embedding_objs, list):
                return None

            embedding = np.array(embedding_objs[0]["embedding"], dtype=np.float32)

            # Normalização consistente
            embedding = self.normalizer.transform([embedding])[0]

            # Atualiza cache
            self.face_cache[cache_key] = {
                "embedding": embedding,
                "timestamp": current_time,
            }

            # Limpa cache antigo
            self.clean_old_cache()

            return embedding

        except Exception as e:
            logging.debug(f"Erro na geração de embedding: {str(e)}")
            return None

    def clean_old_cache(self):
        """Limpa cache antigo"""
        current_time = time.time()
        keys_to_remove = [
            key
            for key, value in self.face_cache.items()
            if current_time - value["timestamp"] > self.cache_timeout
        ]
        for key in keys_to_remove:
            del self.face_cache[key]

    def detect_faces_fast(self, frame):
        """Detecção de faces otimizada para catraca"""
        try:
            # Usa detector mais rápido para ambiente controlado
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Equaliza histograma para melhor detecção
            gray = cv2.equalizeHist(gray)

            face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            )

            faces = face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.05,  # Mais rápido
                minNeighbors=3,  # Menos rigoroso
                minSize=(self.min_face_size, self.min_face_size),
                flags=cv2.CASCADE_SCALE_IMAGE,
            )

            detected_faces = []
            for x, y, w, h in faces:
                # Expande um pouco a região da face
                expand = 10
                x = max(0, x - expand)
                y = max(0, y - expand)
                w = min(frame.shape[1] - x, w + 2 * expand)
                h = min(frame.shape[0] - y, h + 2 * expand)

                detected_faces.append({"x": x, "y": y, "w": w, "h": h})

            return detected_faces

        except Exception as e:
            logging.error(f"Erro na detecção de faces: {str(e)}")
            return []

    def recognize_face(self, live_embedding):
        """Reconhecimento otimizado para poucas imagens"""
        try:
            if not self.model_loaded or not self.embeddings_db:
                return None, 0.0

            best_match = None
            best_similarity = 0.0

            for user_id, user_data in self.embeddings_db.items():
                # Usa embedding normalizado pré-calculado
                known_embedding = user_data.get(
                    "normalized_embedding", user_data.get("embedding")
                )

                if known_embedding is None:
                    continue

                similarity = self.calculate_similarity(live_embedding, known_embedding)

                if similarity > best_similarity:
                    best_similarity = similarity
                    best_match = user_id

            # Threshold adaptativo baseado na qualidade
            adaptive_threshold = self.THRESHOLD
            if best_similarity > 0.8:  # Confiança muito alta
                adaptive_threshold = max(self.THRESHOLD - 0.05, 0.5)
            elif best_similarity < 0.4:  # Confiança muito baixa
                adaptive_threshold = min(self.THRESHOLD + 0.05, 0.8)

            if best_match and best_similarity >= adaptive_threshold:
                logging.info(
                    f"✅ MATCH: {best_match} - Similaridade: {best_similarity:.4f}"
                )
                return best_match, best_similarity
            else:
                if best_match:
                    logging.debug(
                        f"❌ Threshold não atingido: {best_similarity:.4f} < {adaptive_threshold}"
                    )
                return None, best_similarity

        except Exception as e:
            logging.error(f"Erro no reconhecimento: {str(e)}")
            return None, 0.0

    def process_expressions(self, frame):
        """Processa expressões faciais (apenas se ativado) - APENAS ATUALIZA RESULTADOS, NÃO DESENHA"""
        if not self.enable_expression_analysis:
            return frame, {}

        try:
            current_time = time.time()

            # Limita a frequência de Analise de expressoes
            if (
                current_time - self.last_expression_analysis
                < self.expression_analysis_interval
            ):
                return frame, self.expression_results

            self.last_expression_analysis = current_time

            results = self.expression_analyzer.analyze_expressions(frame)
            # NÃO desenha mais aqui; apenas armazena resultados
            self.expression_results = results
            return frame, results

        except Exception as e:
            logging.debug(f"Erro na Analise de expressoes: {str(e)}")
            return frame, {}

    def process_frame(self, frame):
        """Processamento otimizado para catraca com Analise de expressoes (opcional) - APENAS DESENHA ROSTOS E NOMES"""
        try:
            if frame is None or frame.size == 0:
                return frame

            display_frame = frame.copy()
            current_time = time.time()

            # Limita taxa de processamento
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
                # Ainda assim processa expressões se ativado (mesmo sem modelo)
                if self.enable_expression_analysis:
                    display_frame, expression_results = self.process_expressions(
                        display_frame
                    )
                return display_frame

            # Detecta faces
            faces = self.detect_faces_fast(frame)

            # Processa expressões faciais se ativado (apenas atualiza resultados)
            if self.enable_expression_analysis:
                display_frame, expression_results = self.process_expressions(
                    display_frame
                )
            else:
                expression_results = {}

            if not faces:
                return display_frame

            for face in faces:
                x, y, w, h = face["x"], face["y"], face["w"], face["h"]

                # Extrai região do rosto
                face_region = frame[y : y + h, x : x + w]
                if face_region.size == 0:
                    continue

                # Gera embedding
                live_emb = self.safe_get_embedding(face_region)
                if live_emb is None:
                    label = "Analisando..."
                    color = (0, 255, 255)  # Amarelo
                else:
                    # Reconhece a face
                    user_id, similarity = self.recognize_face(live_emb)

                    if user_id:
                        user_info = self.get_user_info(user_id)
                        label = f"{user_info['nome']} ({similarity:.3f})"
                        color = (0, 255, 0)  # Verde

                        # Adiciona confiança visual
                        confidence_width = int(w * similarity)
                        cv2.rectangle(
                            display_frame,
                            (x, y + h + 5),
                            (x + confidence_width, y + h + 10),
                            color,
                            -1,
                        )
                    else:
                        label = f"Não identificado ({similarity:.3f})"
                        color = (0, 0, 255)  # Vermelho

                # Desenha retângulo e label
                cv2.rectangle(display_frame, (x, y), (x + w, y + h), color, 2)

                # Fundo para texto
                text_bg_y = max(y - 25, 0)
                text_width = len(label) * 10
                cv2.rectangle(
                    display_frame, (x, text_bg_y), (x + text_width, y), color, -1
                )
                cv2.rectangle(
                    display_frame, (x, text_bg_y), (x + text_width, y), color, 2
                )

                cv2.putText(
                    display_frame,
                    label,
                    (x + 5, y - 8),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),  # Texto branco
                    1,
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

            # Thresholds mais tolerantes para ambiente de catraca
            camera_covered = mean_intensity < 25 or std_intensity < 10
            return camera_covered

        except Exception as e:
            logging.error(f"Erro ao verificar câmera tampada: {str(e)}")
            return False

    def get_expression_statistics(self):
        """Retorna estatísticas das expressões analisadas"""
        if not self.enable_expression_analysis or not hasattr(
            self, "expression_analyzer"
        ):
            return {}

        try:
            return {
                "trend_analysis": self.expression_analyzer.get_trend_analysis(),
                "history_size": len(self.expression_analyzer.expression_history),
                "current_results": self.expression_results,
            }
        except Exception as e:
            logging.debug(f"Erro ao obter estatísticas: {str(e)}")
            return {}

    def cleanup(self):
        """Limpeza de recursos"""
        if self.db_connection:
            try:
                self.db_connection.close()
            except Exception as e:
                logging.error(f"Erro ao fechar conexão com o banco: {str(e)}")

        if hasattr(self, "expression_analyzer"):
            try:
                self.expression_analyzer.cleanup()
            except Exception as e:
                logging.error(f"Erro ao finalizar analisador de expressões: {str(e)}")

        self.face_cache.clear()
        logging.info("Processador facial finalizado")
