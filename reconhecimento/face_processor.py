# face_processor.py - VERSÃO COMPLETA COM RASTREAMENTO DE UM ÚNICO ROSTO E PRIORIDADE
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
        Inicializa o processador facial com rastreamento de um único rosto.

        Args:
            model_path: Caminho para o modelo treinado (pickle)
            threshold: Limiar de similaridade
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
        self.processing_interval = 0.5  # Intervalo entre processamentos (0.5s)
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
        self.expression_analysis_interval = 1.0  # Análise de expressão a cada 1s

        # ========== RASTREAMENTO DE UM ÚNICO ROSTO ==========
        self.tracking = {
            'active': False,           # Há um rosto sendo rastreado?
            'bbox': None,               # (x, y, w, h) da última detecção
            'user_id': None,            # ID do usuário se reconhecido
            'similarity': 0.0,           # Similaridade do reconhecimento
            'priority': None,            # 'registered' ou 'unknown'
            'lost_frames': 0,            # Número de frames sem detecção
            'first_seen': 0,              # Timestamp da primeira aparição
            'last_seen': 0,               # Timestamp da última atualização
        }
        self.max_lost_frames = 30        # Perde o rastreamento após 30 frames sem detecção
        self.iou_threshold = 0.3          # IOU mínimo para associar ao mesmo rosto

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

            if emb1.size == 0 or emb2.size == 0:
                return 0.0
            norm1 = np.linalg.norm(emb1)
            norm2 = np.linalg.norm(emb2)
            if norm1 == 0 or norm2 == 0:
                return 0.0

            similarity = np.dot(emb1, emb2) / (norm1 * norm2)
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

            embedding_objs = DeepFace.represent(
                img_path=frame,  # imagem completa (array numpy)
                model_name=self.EMBEDDING_MODEL,
                detector_backend="mtcnn",
                enforce_detection=False,
                align=True,
                normalization="base",
            )

            if not embedding_objs or len(embedding_objs) == 0:
                return None

            embedding = np.array(
                embedding_objs[0]["embedding"], dtype=np.float32
            ).flatten()

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
            similarities = {}

            for user_id, user_data in self.embeddings_db.items():
                known_emb = user_data.get("normalized_embedding")
                if known_emb is None:
                    continue

                sim = self.calculate_similarity(live_embedding, known_emb)
                similarities[user_id] = sim

                if sim > best_similarity:
                    best_similarity = sim
                    best_match = user_id

            if similarities:
                top3 = sorted(similarities.items(), key=lambda x: x[1], reverse=True)[:3]
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

    # ==================== RASTREAMENTO E PRIORIDADE ====================
    def compute_iou(self, box1, box2):
        """
        Calcula Intersection over Union entre duas bounding boxes.
        Cada box: (x, y, w, h)
        """
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2
        left1, right1 = x1, x1 + w1
        top1, bottom1 = y1, y1 + h1
        left2, right2 = x2, x2 + w2
        top2, bottom2 = y2, y2 + h2

        x_left = max(left1, left2)
        y_top = max(top1, top2)
        x_right = min(right1, right2)
        y_bottom = min(bottom1, bottom2)

        if x_right < x_left or y_bottom < y_top:
            return 0.0
        intersection = (x_right - x_left) * (y_bottom - y_top)
        area1 = w1 * h1
        area2 = w2 * h2
        union = area1 + area2 - intersection
        return intersection / union if union > 0 else 0

    def _update_tracking(self, faces_data):
        """
        Atualiza o rosto alvo com base nas faces detectadas e regras de prioridade.
        faces_data: lista de dicionários com 'bbox', 'user_id', 'similarity', 'priority'
        """
        if not faces_data:
            # Nenhuma face detectada
            if self.tracking['active']:
                self.tracking['lost_frames'] += 1
                if self.tracking['lost_frames'] > self.max_lost_frames:
                    self.tracking['active'] = False
            return

        if not self.tracking['active']:
            # Sem alvo: ativa com a primeira face (a que apareceu primeiro)
            first = faces_data[0]
            self.tracking.update({
                'active': True,
                'bbox': first['bbox'],
                'user_id': first['user_id'],
                'similarity': first['similarity'],
                'priority': first['priority'],
                'lost_frames': 0,
                'first_seen': time.time(),
                'last_seen': time.time(),
            })
            return

        # Tem alvo ativo: tenta associar por IOU
        best_iou = 0
        best_idx = None
        for i, face in enumerate(faces_data):
            iou = self.compute_iou(self.tracking['bbox'], face['bbox'])
            if iou > best_iou and iou >= self.iou_threshold:
                best_iou = iou
                best_idx = i

        if best_idx is not None:
            # Associou com a mesma pessoa
            face = faces_data[best_idx]
            self.tracking.update({
                'bbox': face['bbox'],
                'user_id': face['user_id'],
                'similarity': face['similarity'],
                'priority': face['priority'],
                'lost_frames': 0,
                'last_seen': time.time(),
            })
            # Remove a face associada da lista para não ser considerada como nova
            faces_data.pop(best_idx)
        else:
            # Não associou: incrementa perda
            self.tracking['lost_frames'] += 1
            if self.tracking['lost_frames'] > self.max_lost_frames:
                self.tracking['active'] = False
                return

        # Verifica se alguma face restante tem prioridade maior (cadastrado)
        if self.tracking['priority'] != 'registered':
            for face in faces_data:
                if face['priority'] == 'registered':
                    # Troca o alvo para o rosto cadastrado
                    self.tracking.update({
                        'active': True,
                        'bbox': face['bbox'],
                        'user_id': face['user_id'],
                        'similarity': face['similarity'],
                        'priority': 'registered',
                        'lost_frames': 0,
                        'first_seen': time.time(),
                        'last_seen': time.time(),
                    })
                    break

    def _draw_target(self, frame):
        """
        Desenha o retângulo e informações do rosto alvo.
        """
        if not self.tracking['active']:
            return
        x, y, w, h = self.tracking['bbox']
        user_id = self.tracking['user_id']
        sim = self.tracking['similarity']

        if user_id:
            info = self.get_user_info(user_id)
            nome = f"{info['nome']} {info['sobrenome']}".strip()
            label = f"{nome} ({sim:.1%})"
            color = (0, 255, 0)
        else:
            label = f"Não identificado ({sim:.1%})"
            color = (0, 0, 255)

        # Retângulo ao redor da face
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

        # Fundo para o texto
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        text_bg_y1 = max(y - th - 10, 0)
        cv2.rectangle(frame, (x, text_bg_y1), (x + tw + 10, y), color, -1)

        # Texto do nome
        cv2.putText(
            frame,
            label,
            (x + 5, y - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
        )

        # Barra de confiança
        bar_width = int(w * sim)
        cv2.rectangle(frame, (x, y + h + 5), (x + bar_width, y + h + 10), color, -1)

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

            # Controle de taxa de processamento (DeepFace é pesado)
            if current_time - self.last_processed_time < self.processing_interval:
                # Apenas desenha o último alvo conhecido (se houver)
                if self.tracking['active']:
                    self._draw_target(display_frame)
                return display_frame

            self.last_processed_time = current_time

            # 1. Detecta faces (rápido) para obter bounding boxes
            faces_bbox = self.detect_faces_fast(frame)
            faces_data = []

            # 2. Para cada face, extrai embedding e tenta reconhecer
            for bbox in faces_bbox:
                x, y, w, h = bbox['x'], bbox['y'], bbox['w'], bbox['h']
                face_region = frame[y:y + h, x:x + w]
                if face_region.size == 0:
                    continue

                # Gera embedding (usando MTCNN no frame completo)
                embedding = self.safe_get_embedding(frame)
                user_id = None
                similarity = 0.0
                if embedding is not None and self.model_loaded:
                    user_id, similarity = self.recognize_face(embedding)

                faces_data.append({
                    'bbox': (x, y, w, h),
                    'user_id': user_id,
                    'similarity': similarity,
                    'priority': 'registered' if user_id else 'unknown'
                })

            # 3. Atualiza rastreamento com base nas faces detectadas
            self._update_tracking(faces_data)

            # 4. Processa expressões APENAS para o rosto alvo
            if self.tracking['active'] and self.enable_expression_analysis:
                x, y, w, h = self.tracking['bbox']
                face_roi = frame[y:y + h, x:x + w]
                if face_roi.size > 0:
                    _, expr_results = self.process_expressions(face_roi)
                    self.expression_results = expr_results

            # 5. Desenha informações do alvo na tela
            if self.tracking['active']:
                self._draw_target(display_frame)

            return display_frame

        except Exception as e:
            logging.error(f"Erro no processamento do frame: {str(e)}")
            return frame

    # ==================== EXPRESSÕES (OPCIONAL) ====================
    def process_expressions(self, face_roi):
        """
        Processa expressões faciais na região do rosto (ROI).
        Retorna o ROI inalterado e os resultados.
        """
        if not self.enable_expression_analysis or face_roi is None or face_roi.size == 0:
            return face_roi, {}

        try:
            current_time = time.time()
            if (
                current_time - self.last_expression_analysis
                < self.expression_analysis_interval
            ):
                return face_roi, self.expression_results

            self.last_expression_analysis = current_time
            results = self.expression_analyzer.analyze_expressions(face_roi)
            self.expression_results = results
            return face_roi, results

        except Exception as e:
            logging.debug(f"Erro na análise de expressões: {str(e)}")
            return face_roi, {}

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