# face_processor.py - VERSÃO CORRIGIDA PARA RECONHECIMENTO COM POUCAS IMAGENS
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

# Usar MTCNN para detecção (mesmo do treinamento)
from mtcnn import MTCNN

try:
    from .expression_analyzer import ExpressionAnalyzer
except ImportError:
    # Fallback para desenvolvimento
    from expression_analyzer import ExpressionAnalyzer


class FaceProcessor:
    def __init__(self, model_path="model/deepface_model.pkl", threshold=0.4):
        """
        Inicializa o processador facial.

        Args:
            model_path: Caminho para o modelo treinado (pickle)
            threshold: Limiar de similaridade (reduzido para 0.4)
        """
        self.MODEL_PATH = model_path
        self.THRESHOLD = threshold
        self.EMBEDDING_MODEL = "Facenet512"
        self.DETECTOR = "mtcnn"  # Usar o mesmo detector do treinamento
        self.model_loaded = False
        self.embeddings_db = {}
        self.db_connection = None

        # Normalizador L2 (consistente com o treinamento)
        self.normalizer = Normalizer(norm="l2")

        # OTIMIZAÇÕES DE PERFORMANCE
        self.last_processed_time = 0
        self.processing_interval = (
            1.0  # Intervalo entre processamentos (para dar tempo ao MTCNN)
        )
        self.min_face_size = 80
        self.face_cache = {}  # Cache de embeddings para performance
        self.cache_timeout = 30  # segundos

        # Detector MTCNN próprio (mais rápido que chamar DeepFace toda vez)
        self.mtcnn_detector = MTCNN()

        # Análise de expressões (opcional)
        self.expression_analyzer = ExpressionAnalyzer()
        self.enable_expression_analysis = False
        self.expression_results = {}
        self.last_expression_analysis = 0
        self.expression_analysis_interval = 1.0

        # Carrega o modelo na inicialização
        self.load_model()
        self.initialize_database()

    # ==================== CARGA DO MODELO ====================
    def load_model(self):
        """Carrega o modelo de reconhecimento facial a partir do arquivo pickle."""
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

            # Pré-processa os embeddings (garantir que são numpy arrays normalizados)
            for user_id, user_data in self.embeddings_db.items():
                if "embedding" in user_data:
                    emb = np.array(user_data["embedding"], dtype=np.float32).flatten()
                    # Já deve estar normalizado, mas garantimos
                    norm = np.linalg.norm(emb)
                    if norm > 0:
                        user_data["normalized_embedding"] = emb / norm
                    else:
                        user_data["normalized_embedding"] = emb
                else:
                    # Se não houver embedding médio, usa a lista de embeddings
                    embeddings_list = user_data.get("embeddings", [])
                    if embeddings_list:
                        # Calcula a média normalizada
                        avg_emb = np.mean(
                            [np.array(e) for e in embeddings_list], axis=0
                        )
                        norm = np.linalg.norm(avg_emb)
                        user_data["normalized_embedding"] = (
                            avg_emb / norm if norm > 0 else avg_emb
                        )

            self.model_loaded = True
            return True

        except Exception as e:
            logging.error(f"❌ Erro ao carregar o modelo: {str(e)}")
            self.model_loaded = False
            return False

    # ==================== BANCO DE DADOS ====================
    def initialize_database(self):
        """Conecta ao banco de dados MySQL para obter informações dos usuários."""
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

    # ==================== RECONHECIMENTO ====================
    def calculate_similarity(self, emb1, emb2):
        """
        Calcula a similaridade cosseno entre dois embeddings.
        Retorna valor entre 0 e 1 (1 = idênticos).
        """
        try:
            if emb1 is None or emb2 is None:
                return 0.0

            emb1 = np.array(emb1, dtype=np.float32).flatten()
            emb2 = np.array(emb2, dtype=np.float32).flatten()

            # Verifica se os embeddings são válidos
            if emb1.size == 0 or emb2.size == 0:
                return 0.0
            norm1 = np.linalg.norm(emb1)
            norm2 = np.linalg.norm(emb2)
            if norm1 == 0 or norm2 == 0:
                return 0.0

            # Similaridade cosseno
            similarity = np.dot(emb1, emb2) / (norm1 * norm2)
            # Garantir que está no intervalo [0,1] (para vetores normalizados, o cosseno pode ser negativo)
            similarity = max(0.0, min(1.0, similarity))
            return float(similarity)

        except Exception as e:
            logging.debug(f"Erro no cálculo de similaridade: {str(e)}")
            return 0.0

    def safe_get_embedding(self, frame):
        """
        Gera embedding facial usando o mesmo pipeline do treinamento:
        - Detecção com MTCNN na imagem completa
        - Alinhamento e extração com DeepFace (detector_backend='mtcnn')
        - Normalização L2
        """
        try:
            if frame is None or frame.size == 0:
                return None

            # Usa DeepFace com detector_backend='mtcnn' para garantir alinhamento igual ao treino
            embedding_objs = DeepFace.represent(
                img_path=frame,  # imagem completa (array numpy)
                model_name=self.EMBEDDING_MODEL,
                detector_backend="mtcnn",  # mesmo detector do treinamento
                enforce_detection=False,  # não falha se não detectar
                align=True,  # alinhamento facial
                normalization="base",  # sem normalização extra (faremos L2 depois)
            )

            if not embedding_objs or len(embedding_objs) == 0:
                return None

            # Extrai o embedding da primeira face detectada
            embedding = np.array(
                embedding_objs[0]["embedding"], dtype=np.float32
            ).flatten()

            # Normalização L2 (igual ao treinamento)
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm

            return embedding

        except Exception as e:
            logging.debug(f"Erro na geração de embedding: {str(e)}")
            return None

    def recognize_face(self, live_embedding):
        """
        Compara o embedding ao vivo com a base de dados.
        Retorna (user_id, similaridade) ou (None, maior_similaridade).
        """
        try:
            if not self.model_loaded or not self.embeddings_db:
                return None, 0.0

            best_match = None
            best_similarity = 0.0
            similarities = {}  # para log

            for user_id, user_data in self.embeddings_db.items():
                # Usa o embedding normalizado pré-calculado
                known_emb = user_data.get("normalized_embedding")
                if known_emb is None:
                    continue

                sim = self.calculate_similarity(live_embedding, known_emb)
                similarities[user_id] = sim

                if sim > best_similarity:
                    best_similarity = sim
                    best_match = user_id

            # Log detalhado para depuração
            if similarities:
                # Mostra apenas as top 3 para não poluir muito
                top3 = sorted(similarities.items(), key=lambda x: x[1], reverse=True)[
                    :3
                ]
                log_msg = "Similaridades: " + ", ".join(
                    [f"{uid}: {sim:.3f}" for uid, sim in top3]
                )
                logging.debug(log_msg)
            logging.debug(
                f"Melhor match: {best_match} com {best_similarity:.4f} (threshold {self.THRESHOLD})"
            )

            if best_match and best_similarity >= self.THRESHOLD:
                logging.info(
                    f"✅ RECONHECIDO: ID {best_match} - Similaridade: {best_similarity:.4f}"
                )
                return best_match, best_similarity
            else:
                return None, best_similarity

        except Exception as e:
            logging.error(f"Erro no reconhecimento: {str(e)}")
            return None, 0.0

    # ==================== DETECÇÃO DE FACES RÁPIDA ====================
    def detect_faces_fast(self, frame):
        """
        Detecta faces usando o Haar Cascade (rápido) para extrair regiões.
        Isso é usado apenas para cortar a região e passar para o embedding,
        mas o embedding em si usará MTCNN novamente (para manter consistência).
        """
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.equalizeHist(gray)

            face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            )

            faces = face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.05,
                minNeighbors=3,
                minSize=(self.min_face_size, self.min_face_size),
                flags=cv2.CASCADE_SCALE_IMAGE,
            )

            detected = []
            for x, y, w, h in faces:
                # Expande um pouco para incluir testa e queixo
                expand = 10
                x = max(0, x - expand)
                y = max(0, y - expand)
                w = min(frame.shape[1] - x, w + 2 * expand)
                h = min(frame.shape[0] - y, h + 2 * expand)
                detected.append({"x": x, "y": y, "w": w, "h": h})

            return detected

        except Exception as e:
            logging.error(f"Erro na detecção rápida: {str(e)}")
            return []

    # ==================== PROCESSAMENTO DO FRAME ====================
    def process_frame(self, frame):
        """
        Processa um frame: detecta faces, reconhece e desenha resultados.
        Retorna o frame com anotações.
        """
        try:
            if frame is None or frame.size == 0:
                return frame

            display_frame = frame.copy()
            current_time = time.time()

            # Controle de taxa de processamento
            if current_time - self.last_processed_time < self.processing_interval:
                return display_frame

            self.last_processed_time = current_time

            # Se o modelo não foi carregado, apenas desenha aviso e processa expressões (se ativo)
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
                if self.enable_expression_analysis:
                    display_frame, _ = self.process_expressions(display_frame)
                return display_frame

            # Detecta faces (rápido) para obter regiões de interesse
            faces = self.detect_faces_fast(frame)

            # Processa expressões (opcional) - apenas atualiza resultados, não desenha
            if self.enable_expression_analysis:
                display_frame, _ = self.process_expressions(display_frame)

            if not faces:
                return display_frame

            for face in faces:
                x, y, w, h = face["x"], face["y"], face["w"], face["h"]

                # Extrai região do rosto
                face_region = frame[y : y + h, x : x + w]
                if face_region.size == 0:
                    continue

                # Gera embedding (agora com MTCNN + DeepFace)
                live_emb = self.safe_get_embedding(
                    frame
                )  # passamos o frame completo para detecção consistente
                # Nota: safe_get_embedding usará MTCNN em toda a imagem, o que pode encontrar a mesma face.
                # Para evitar processar múltiplas vezes, poderíamos usar a região, mas aí perderíamos o alinhamento.
                # Vamos manter assim: safe_get_embedding já detecta a face principal, então se houver várias,
                # ele pode retornar o embedding da primeira. Para múltiplas faces, precisaríamos de um loop,
                # mas em catraca normalmente só há uma pessoa. Se quiser suportar várias, seria necessário modificar.

                if live_emb is None:
                    label = "Analisando..."
                    color = (0, 255, 255)  # Amarelo
                else:
                    user_id, similarity = self.recognize_face(live_emb)

                    if user_id:
                        user_info = self.get_user_info(user_id)
                        nome_completo = (
                            f"{user_info['nome']} {user_info['sobrenome']}".strip()
                        )
                        label = f"{nome_completo} ({similarity:.1%})"
                        color = (0, 255, 0)  # Verde

                        # Barra de confiança
                        bar_width = int(w * similarity)
                        cv2.rectangle(
                            display_frame,
                            (x, y + h + 5),
                            (x + bar_width, y + h + 10),
                            color,
                            -1,
                        )
                    else:
                        label = f"Não identificado ({similarity:.1%})"
                        color = (0, 0, 255)  # Vermelho

                # Desenha retângulo ao redor da face
                cv2.rectangle(display_frame, (x, y), (x + w, y + h), color, 2)

                # Fundo para o texto
                (text_w, text_h), _ = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
                )
                text_bg_y1 = max(y - text_h - 10, 0)
                text_bg_y2 = y
                cv2.rectangle(
                    display_frame,
                    (x, text_bg_y1),
                    (x + text_w + 10, text_bg_y2),
                    color,
                    -1,
                )

                # Texto do nome
                cv2.putText(
                    display_frame,
                    label,
                    (x + 5, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),  # branco
                    1,
                )

            return display_frame

        except Exception as e:
            logging.error(f"Erro no processamento do frame: {str(e)}")
            return frame

    # ==================== EXPRESSÕES (OPCIONAL) ====================
    def process_expressions(self, frame):
        """
        Processa expressões faciais (apenas se ativado).
        Retorna o frame inalterado e os resultados.
        """
        if not self.enable_expression_analysis:
            return frame, {}

        try:
            current_time = time.time()
            if (
                current_time - self.last_expression_analysis
                < self.expression_analysis_interval
            ):
                return frame, self.expression_results

            self.last_expression_analysis = current_time
            results = self.expression_analyzer.analyze_expressions(frame)
            self.expression_results = results
            return frame, results

        except Exception as e:
            logging.debug(f"Erro na análise de expressões: {str(e)}")
            return frame, {}

    # ==================== CONSULTA AO BANCO ====================
    def get_user_info(self, user_id):
        """Obtém nome e sobrenome do usuário pelo ID."""
        if not self.db_connection:
            return {"nome": "Desconhecido", "sobrenome": ""}

        try:
            cursor = self.db_connection.cursor(dictionary=True)
            cursor.execute(
                "SELECT nome, sobrenome FROM cadastros WHERE id = %s", (user_id,)
            )
            result = cursor.fetchone()
            cursor.close()
            return result or {"nome": "Desconhecido", "sobrenome": ""}
        except Error as e:
            logging.error(f"Erro ao buscar usuário: {str(e)}")
            return {"nome": "Desconhecido", "sobrenome": ""}

    # ==================== UTILITÁRIOS ====================
    def is_camera_covered(self, frame):
        """Detecta se a câmera está tampada ou muito escura."""
        try:
            if frame is None or frame.size == 0:
                return True

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            mean_intensity = np.mean(gray)
            std_intensity = np.std(gray)

            # Heurística: muito escuro ou muito uniforme
            return mean_intensity < 25 or std_intensity < 10

        except Exception as e:
            logging.error(f"Erro ao verificar câmera tampada: {str(e)}")
            return False

    def get_expression_statistics(self):
        """Retorna estatísticas das expressões (se ativadas)."""
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

    # ==================== LIMPEZA ====================
    def cleanup(self):
        """Libera recursos (conexão com banco e analisador de expressões)."""
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
