# expression_analyzer.py - VERSÃO OTIMIZADA COM SUPORTE A ROI E SUAVIZAÇÃO MELHORADA
import cv2
import numpy as np
import dlib
import logging
from scipy.spatial import distance as dist
import time
from deepface import DeepFace
import os
from collections import deque, Counter

class ExpressionAnalyzer:
    def __init__(
        self, landmarks_model_path="treinamento/shape_predictor_68_face_landmarks.dat"
    ):
        self.landmarks_model_path = landmarks_model_path
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(landmarks_model_path)

        # Índices dos landmarks faciais
        self.FACIAL_LANDMARKS = {
            "mouth": list(range(48, 68)),
            "right_eyebrow": list(range(17, 22)),
            "left_eyebrow": list(range(22, 27)),
            "right_eye": list(range(36, 42)),
            "left_eye": list(range(42, 48)),
            "nose": list(range(27, 36)),
            "jaw": list(range(0, 17)),
        }

        # Configurações para cansaço/tristeza
        self.EYE_AR_THRESH = 0.25
        self.MOUTH_AR_THRESH = 0.35
        self.EAR_CONSEC_FRAMES = 3
        self.counter = 0
        self.total_blinks = 0
        self.eye_closed_frames = 0

        # Histórico para análise temporal e suavização
        self.expression_history = []
        self.history_size = 30
        self.emotion_buffer = deque(maxlen=5)          # buffer para suavizar emoção dominante
        self.emotion_scores_buffer = {                 # buffer para suavizar scores individuais
            emotion: deque(maxlen=5) for emotion in ["angry","disgust","fear","happy","sad","surprise","neutral"]
        }

        # Emoções básicas
        self.emotions = [
            "angry",
            "disgust",
            "fear",
            "happy",
            "sad",
            "surprise",
            "neutral",
        ]

        # Cache simples para evitar reanálise da mesma imagem (último hash)
        self.last_frame_hash = None
        self.last_analysis_time = 0
        self.cache_duration = 0.5  # segundos

        logging.info("Analisador de expressões e emoções inicializado (modo ROI)")

    # ---------- EXTRAIR REGIÃO DA FACE A PARTIR DE BBOX ----------
    def _extract_face_region(self, frame, bbox, padding=0.2):
        """
        Extrai e redimensiona a região da face baseado na bounding box.
        bbox: (x, y, w, h) em coordenadas do frame original.
        """
        if bbox is None:
            return None
        x, y, w, h = bbox
        # Expande um pouco para incluir testa e queixo
        pad = int(padding * w)
        x = max(0, x - pad)
        y = max(0, y - pad)
        w = min(frame.shape[1] - x, w + 2 * pad)
        h = min(frame.shape[0] - y, h + 2 * pad)
        face_img = frame[y:y+h, x:x+w]
        if face_img.size == 0:
            return None
        # Redimensiona para tamanho mínimo (224x224) se necessário
        if face_img.shape[0] < 224 or face_img.shape[1] < 224:
            face_img = cv2.resize(face_img, (224, 224))
        return face_img

    # ---------- DETECTAR LANDMARKS USANDO BBOX (SE FORNECIDA) ----------
    def detect_landmarks_from_bbox(self, frame, bbox):
        """
        Detecta landmarks usando a bounding box fornecida (já detectada por Haar Cascade).
        Retorna (landmarks, face_rect) ou (None, None) em caso de falha.
        """
        try:
            x, y, w, h = bbox
            # Cria um retângulo dlib a partir da bbox
            face_rect = dlib.rectangle(left=x, top=y, right=x+w, bottom=y+h)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            landmarks = self.predictor(gray, face_rect)
            return landmarks, face_rect
        except Exception as e:
            logging.debug(f"Erro ao detectar landmarks a partir de bbox: {str(e)}")
            return None, None

    # ---------- DETECTAR LANDMARKS (MÉTODO ORIGINAL, USADO QUANDO NÃO HÁ BBOX) ----------
    def detect_landmarks(self, frame):
        """Detecta landmarks no frame completo."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.detector(gray)
        if len(faces) == 0:
            return None, None
        face_rect = faces[0]
        landmarks = self.predictor(gray, face_rect)
        return landmarks, face_rect

    # ---------- ANÁLISE DE EMOÇÕES BÁSICAS (COM SUAVIZAÇÃO) ----------
    def analyze_basic_emotions(self, face_img):
        """
        Analisa emoções básicas usando DeepFace em uma imagem de rosto recortada.
        Retorna dicionário com emoção dominante, scores suavizados e confiança.
        """
        try:
            if face_img is None or face_img.size == 0:
                return None

            # Cache simples: se a imagem não mudou significativamente, retorna último resultado
            current_hash = hash(face_img.tobytes())
            if (current_hash == self.last_frame_hash and 
                time.time() - self.last_analysis_time < self.cache_duration):
                # Retorna o último resultado suavizado (evita chamadas repetidas)
                return self._get_smoothed_emotion_result()

            self.last_frame_hash = current_hash
            self.last_analysis_time = time.time()

            analysis = DeepFace.analyze(
                img_path=face_img,
                actions=["emotion"],
                enforce_detection=False,
                detector_backend="mtcnn",  # mais preciso que opencv
                silent=True,
            )

            if analysis and isinstance(analysis, list):
                emotion_data = analysis[0].get("emotion", {})
                dominant_emotion = analysis[0].get("dominant_emotion", "neutral")
                emotion_scores = {
                    emotion: emotion_data.get(emotion, 0) for emotion in self.emotions
                }

                # Atualiza buffers de suavização
                self.emotion_buffer.append(dominant_emotion)
                for emotion, score in emotion_scores.items():
                    self.emotion_scores_buffer[emotion].append(score)

                return self._get_smoothed_emotion_result()
            return None
        except Exception as e:
            logging.debug(f"Erro na análise de emoções básicas: {str(e)}")
            return None

    def _get_smoothed_emotion_result(self):
        """Retorna resultado de emoção suavizado a partir dos buffers."""
        # Emoção dominante suavizada (moda)
        if len(self.emotion_buffer) > 0:
            dominant_emotion = Counter(self.emotion_buffer).most_common(1)[0][0]
        else:
            dominant_emotion = "neutral"

        # Scores suavizados (média móvel)
        emotion_scores = {}
        for emotion in self.emotions:
            buffer = self.emotion_scores_buffer[emotion]
            if len(buffer) > 0:
                emotion_scores[emotion] = sum(buffer) / len(buffer)
            else:
                emotion_scores[emotion] = 0.0

        confidence = max(emotion_scores.values()) if emotion_scores else 0

        return {
            "dominant_emotion": dominant_emotion,
            "emotion_scores": emotion_scores,
            "confidence": confidence,
        }

    # ---------- MÉTODO PRINCIPAL: ANALISAR EXPRESSÕES (ACEITA BBOX OPCIONAL) ----------
    def analyze_expressions(self, frame, bbox=None):
        """
        Analisa todas as expressões faciais e emoções.
        Se bbox for fornecido, usa-o para localizar a face e landmarks,
        evitando redetecção completa.
        """
        results = {
            "basic_emotions": {
                "dominant_emotion": "neutral",
                "emotion_scores": {},
                "confidence": 0,
            },
            "fatigue": {"score": 0, "indicators": [], "level": "Baixo"},
            "sadness": {"score": 0, "indicators": [], "level": "Baixo"},
            "landmarks": None,
        }

        landmarks = None
        face_rect = None

        # Se bbox foi fornecido, tenta obter landmarks diretamente
        if bbox is not None:
            landmarks, face_rect = self.detect_landmarks_from_bbox(frame, bbox)

        # Se não conseguiu com bbox ou não foi fornecido, tenta detecção completa
        if landmarks is None:
            landmarks, face_rect = self.detect_landmarks(frame)

        if landmarks is not None:
            results["landmarks"] = landmarks

            # Analisa expressões específicas (cansaço, tristeza)
            fatigue_score, fatigue_indicators = self.analyze_fatigue(landmarks)
            sadness_score, sadness_indicators = self.analyze_sadness(landmarks)

            results["fatigue"] = {
                "score": fatigue_score,
                "indicators": fatigue_indicators,
                "level": self.get_expression_level(fatigue_score),
            }
            results["sadness"] = {
                "score": sadness_score,
                "indicators": sadness_indicators,
                "level": self.get_expression_level(sadness_score),
            }

            # Extrai a região da face para análise de emoções
            if face_rect is not None:
                # Usa o retângulo do dlib (pode ser ligeiramente diferente da bbox original)
                dlib_bbox = (face_rect.left(), face_rect.top(), 
                             face_rect.width(), face_rect.height())
            else:
                dlib_bbox = bbox  # fallback

            face_img = self._extract_face_region(frame, dlib_bbox)
            if face_img is not None:
                emotion_results = self.analyze_basic_emotions(face_img)
                if emotion_results:
                    results["basic_emotions"] = emotion_results
        else:
            # Fallback: tenta DeepFace no frame inteiro (menos preciso)
            emotion_results = self.analyze_basic_emotions(frame)
            if emotion_results:
                results["basic_emotions"] = emotion_results

        # Atualiza histórico
        self.update_expression_history(results)

        return results

    # ---------- MÉTODOS DE ANÁLISE DE FATIGUE E SADNESS (INALTERADOS) ----------
    def eye_aspect_ratio(self, eye):
        A = dist.euclidean(eye[1], eye[5])
        B = dist.euclidean(eye[2], eye[4])
        C = dist.euclidean(eye[0], eye[3])
        ear = (A + B) / (2.0 * C)
        return ear

    def mouth_aspect_ratio(self, mouth):
        A = dist.euclidean(mouth[0], mouth[6])
        B = dist.euclidean(mouth[2], mouth[10])
        C = dist.euclidean(mouth[4], mouth[8])
        mar = (B + C) / (2.0 * A)
        return mar

    def get_eyebrow_tension(self, eyebrow, eye):
        distances = []
        for i in range(len(eyebrow)):
            dist_val = dist.euclidean(eyebrow[i], eye[i % len(eye)])
            distances.append(dist_val)
        return np.mean(distances)

    def get_head_pose(self, landmarks):
        nose_tip = landmarks[30]
        chin = landmarks[8]
        left_eye = landmarks[36]
        right_eye = landmarks[45]
        vertical_vector = np.array([chin.x - nose_tip.x, chin.y - nose_tip.y])
        vertical_angle = np.degrees(np.arctan2(vertical_vector[1], vertical_vector[0]))
        eye_center_x = (left_eye.x + right_eye.x) / 2
        head_tilt = (right_eye.y - left_eye.y) / (right_eye.x - left_eye.x)
        return vertical_angle, head_tilt

    def analyze_fatigue(self, landmarks):
        fatigue_score = 0
        indicators = []
        left_eye = [
            (landmarks[i].x, landmarks[i].y) for i in self.FACIAL_LANDMARKS["left_eye"]
        ]
        right_eye = [
            (landmarks[i].x, landmarks[i].y) for i in self.FACIAL_LANDMARKS["right_eye"]
        ]
        left_eyebrow = [
            (landmarks[i].x, landmarks[i].y)
            for i in self.FACIAL_LANDMARKS["left_eyebrow"]
        ]
        right_eyebrow = [
            (landmarks[i].x, landmarks[i].y)
            for i in self.FACIAL_LANDMARKS["right_eyebrow"]
        ]
        mouth = [
            (landmarks[i].x, landmarks[i].y) for i in self.FACIAL_LANDMARKS["mouth"]
        ]
        left_ear = self.eye_aspect_ratio(left_eye)
        right_ear = self.eye_aspect_ratio(right_eye)
        ear = (left_ear + right_ear) / 2.0
        if ear < self.EYE_AR_THRESH:
            self.eye_closed_frames += 1
            fatigue_score += 0.3
            indicators.append("Olhos semicerrados")
        else:
            self.eye_closed_frames = max(0, self.eye_closed_frames - 1)
        if self.eye_closed_frames > 15:
            fatigue_score += 0.4
            indicators.append("Olhos fechados por longo período")
        left_brow_tension = self.get_eyebrow_tension(left_eyebrow, left_eye)
        right_brow_tension = self.get_eyebrow_tension(right_eyebrow, right_eye)
        brow_tension = (left_brow_tension + right_brow_tension) / 2.0
        if brow_tension > 15:
            fatigue_score += 0.2
            indicators.append("Sobrancelhas tensionadas")
        vertical_angle, head_tilt = self.get_head_pose(landmarks)
        if abs(vertical_angle) > 20:
            fatigue_score += 0.2
            indicators.append("Cabeça inclinada")
        mar = self.mouth_aspect_ratio(mouth)
        if mar < 0.2:
            fatigue_score += 0.1
            indicators.append("Boca tensionada")
        return min(fatigue_score, 1.0), indicators

    def analyze_sadness(self, landmarks):
        sadness_score = 0
        indicators = []
        left_eyebrow = [
            (landmarks[i].x, landmarks[i].y)
            for i in self.FACIAL_LANDMARKS["left_eyebrow"]
        ]
        right_eyebrow = [
            (landmarks[i].x, landmarks[i].y)
            for i in self.FACIAL_LANDMARKS["right_eyebrow"]
        ]
        mouth = [
            (landmarks[i].x, landmarks[i].y) for i in self.FACIAL_LANDMARKS["mouth"]
        ]
        left_mouth_corner = landmarks[48]
        right_mouth_corner = landmarks[54]
        mouth_center = landmarks[57]
        if (
            left_mouth_corner.y > mouth_center.y
            and right_mouth_corner.y > mouth_center.y
        ):
            sadness_score += 0.4
            indicators.append("Cantos da boca para baixo")
        left_brow_outer = landmarks[17]
        left_brow_inner = landmarks[21]
        right_brow_outer = landmarks[26]
        right_brow_inner = landmarks[22]
        left_brow_slope = (left_brow_outer.y - left_brow_inner.y) / (
            left_brow_outer.x - left_brow_inner.x
        )
        right_brow_slope = (right_brow_inner.y - right_brow_outer.y) / (
            right_brow_inner.x - right_brow_outer.x
        )
        if left_brow_slope > 0.1 and right_brow_slope > 0.1:
            sadness_score += 0.3
            indicators.append("Sobrancelhas inclinadas")
        left_eye = [
            (landmarks[i].x, landmarks[i].y) for i in self.FACIAL_LANDMARKS["left_eye"]
        ]
        right_eye = [
            (landmarks[i].x, landmarks[i].y) for i in self.FACIAL_LANDMARKS["right_eye"]
        ]
        left_ear = self.eye_aspect_ratio(left_eye)
        right_ear = self.eye_aspect_ratio(right_eye)
        if left_ear < 0.22 or right_ear < 0.22:
            sadness_score += 0.2
            indicators.append("Pálpebras caídas")
        return min(sadness_score, 1.0), indicators

    def get_expression_level(self, score):
        if score >= 0.7:
            return "Alto"
        elif score >= 0.4:
            return "Médio"
        else:
            return "Baixo"

    def update_expression_history(self, results):
        self.expression_history.append(
            {
                "timestamp": time.time(),
                "dominant_emotion": results["basic_emotions"]["dominant_emotion"],
                "fatigue": results["fatigue"]["score"],
                "sadness": results["sadness"]["score"],
            }
        )
        if len(self.expression_history) > self.history_size:
            self.expression_history.pop(0)

    def get_emotion_emoji(self, emotion):
        emojis = {
            "happy": "😊",
            "sad": "😔",
            "angry": "😠",
            "surprise": "😲",
            "fear": "😨",
            "disgust": "🤢",
            "neutral": "😐",
        }
        return emojis.get(emotion, "😐")

    def draw_analysis(self, frame, results):
        """Desenha a análise completa no frame (mantido para compatibilidade)."""
        if results["landmarks"] is not None:
            landmarks = results["landmarks"]
            for i in range(68):
                x, y = landmarks.part(i).x, landmarks.part(i).y
                cv2.circle(frame, (x, y), 1, (0, 255, 255), -1)
        h, w = frame.shape[:2]
        dominant_emotion = results["basic_emotions"]["dominant_emotion"]
        confidence = results["basic_emotions"]["confidence"]
        emoji = self.get_emotion_emoji(dominant_emotion)
        emotion_text = f"{emoji} {dominant_emotion.upper()} ({confidence:.1%})"
        cv2.putText(
            frame,
            emotion_text,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
        )
        emotion_scores = results["basic_emotions"]["emotion_scores"]
        y_offset = 60
        for emotion, score in emotion_scores.items():
            if score > 5:
                emoji = self.get_emotion_emoji(emotion)
                text = f"{emoji} {emotion}: {score:.1f}%"
                color = (0, 255, 0) if emotion == dominant_emotion else (200, 200, 200)
                cv2.putText(
                    frame,
                    text,
                    (w - 200, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    color,
                    1,
                )
                y_offset += 20
        expr_y = h - 10
        fatigue_level = results["fatigue"]["level"]
        if fatigue_level != "Baixo":
            color = (0, 0, 255) if fatigue_level == "Alto" else (0, 165, 255)
            cv2.putText(
                frame,
                f"😴 Cansaço: {fatigue_level}",
                (10, expr_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
            )
            expr_y -= 25
        sadness_level = results["sadness"]["level"]
        if sadness_level != "Baixo":
            color = (0, 0, 255) if sadness_level == "Alto" else (0, 165, 255)
            cv2.putText(
                frame,
                f"😔 Tristeza: {sadness_level}",
                (10, expr_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
            )
            expr_y -= 25
        return frame

    def get_trend_analysis(self):
        if len(self.expression_history) < 10:
            return "Dados insuficientes para análise de tendência"
        recent_emotions = [
            entry["dominant_emotion"] for entry in self.expression_history[-10:]
        ]
        emotion_counts = {}
        for emotion in recent_emotions:
            emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
        dominant_recent = max(emotion_counts, key=emotion_counts.get)
        return f"Emocao predominante recente: {dominant_recent}"

    def cleanup(self):
        logging.info("Analisador de expressões finalizado")