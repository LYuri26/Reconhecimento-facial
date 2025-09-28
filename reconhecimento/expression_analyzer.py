# expression_analyzer.py
import cv2
import numpy as np
import dlib
import logging
from scipy.spatial import distance as dist
import time


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

        # Configurações
        self.EYE_AR_THRESH = 0.25  # Threshold para olhos fechados
        self.MOUTH_AR_THRESH = 0.35  # Threshold para boca aberta
        self.EAR_CONSEC_FRAMES = 3  # Frames consecutivos para detectar cansaço
        self.counter = 0
        self.total_blinks = 0
        self.eye_closed_frames = 0

        # Histórico para análise temporal
        self.expression_history = []
        self.history_size = 30

        logging.info("Analisador de expressões inicializado")

    def eye_aspect_ratio(self, eye):
        """Calcula a proporção de abertura dos olhos"""
        # Calcula as distâncias verticais
        A = dist.euclidean(eye[1], eye[5])
        B = dist.euclidean(eye[2], eye[4])
        # Calcula a distância horizontal
        C = dist.euclidean(eye[0], eye[3])
        # Calcula o EAR
        ear = (A + B) / (2.0 * C)
        return ear

    def mouth_aspect_ratio(self, mouth):
        """Calcula a proporção de abertura da boca"""
        # Distância entre os pontos horizontais externos
        A = dist.euclidean(mouth[0], mouth[6])
        # Distâncias verticais
        B = dist.euclidean(mouth[2], mouth[10])
        C = dist.euclidean(mouth[4], mouth[8])
        # Calcula o MAR
        mar = (B + C) / (2.0 * A)
        return mar

    def get_eyebrow_tension(self, eyebrow, eye):
        """Calcula a tensão nas sobrancelhas"""
        # Distância média entre sobrancelha e olho
        distances = []
        for i in range(len(eyebrow)):
            dist_val = dist.euclidean(eyebrow[i], eye[i % len(eye)])
            distances.append(dist_val)
        return np.mean(distances)

    def get_head_pose(self, landmarks):
        """Estima a inclinação da cabeça"""
        nose_tip = landmarks[30]
        chin = landmarks[8]
        left_eye = landmarks[36]
        right_eye = landmarks[45]

        # Calcula inclinação vertical
        vertical_vector = np.array([chin.x - nose_tip.x, chin.y - nose_tip.y])
        vertical_angle = np.degrees(np.arctan2(vertical_vector[1], vertical_vector[0]))

        # Calcula inclinação horizontal
        eye_center_x = (left_eye.x + right_eye.x) / 2
        head_tilt = (right_eye.y - left_eye.y) / (right_eye.x - left_eye.x)

        return vertical_angle, head_tilt

    def analyze_fatigue(self, landmarks):
        """Analisa sinais de cansaço/fadiga"""
        fatigue_score = 0
        indicators = []

        # Extrai regiões de interesse
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

        # 1. Análise de piscadas e olhos fechados
        left_ear = self.eye_aspect_ratio(left_eye)
        right_ear = self.eye_aspect_ratio(right_eye)
        ear = (left_ear + right_ear) / 2.0

        if ear < self.EYE_AR_THRESH:
            self.eye_closed_frames += 1
            fatigue_score += 0.3
            indicators.append("Olhos semicerrados")
        else:
            self.eye_closed_frames = max(0, self.eye_closed_frames - 1)

        # Olhos fechados por muito tempo
        if self.eye_closed_frames > 15:
            fatigue_score += 0.4
            indicators.append("Olhos fechados por longo período")

        # 2. Tensão nas sobrancelhas (sinal de esforço)
        left_brow_tension = self.get_eyebrow_tension(left_eyebrow, left_eye)
        right_brow_tension = self.get_eyebrow_tension(right_eyebrow, right_eye)
        brow_tension = (left_brow_tension + right_brow_tension) / 2.0

        if brow_tension > 15:  # Threshold empírico
            fatigue_score += 0.2
            indicators.append("Sobrancelhas tensionadas")

        # 3. Inclinação da cabeça
        vertical_angle, head_tilt = self.get_head_pose(landmarks)
        if abs(vertical_angle) > 20:  # Cabeça muito inclinada
            fatigue_score += 0.2
            indicators.append("Cabeça inclinada")

        # 4. Expressão facial geral (boca)
        mar = self.mouth_aspect_ratio(mouth)
        if mar < 0.2:  # Boca muito fechada/tensa
            fatigue_score += 0.1
            indicators.append("Boca tensionada")

        return min(fatigue_score, 1.0), indicators

    def analyze_sadness(self, landmarks):
        """Analisa sinais de tristeza"""
        sadness_score = 0
        indicators = []

        # Extrai regiões de interesse
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

        # 1. Cantos da boca para baixo
        left_mouth_corner = landmarks[48]
        right_mouth_corner = landmarks[54]
        mouth_center = landmarks[57]

        if (
            left_mouth_corner.y > mouth_center.y
            and right_mouth_corner.y > mouth_center.y
        ):
            sadness_score += 0.4
            indicators.append("Cantos da boca para baixo")

        # 2. Sobrancelhas inclinadas (formato de "V")
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

        # 3. Pálpebras caídas
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

    def analyze_demotivation(self, landmarks, previous_landmarks=None):
        """Analisa sinais de desânimo/desmotivação"""
        demotivation_score = 0
        indicators = []

        # Combina elementos de cansaço e tristeza
        fatigue_score, fatigue_indicators = self.analyze_fatigue(landmarks)
        sadness_score, sadness_indicators = self.analyze_sadness(landmarks)

        demotivation_score = fatigue_score * 0.6 + sadness_score * 0.4

        # Movimento reduzido (se tiver frame anterior)
        if previous_landmarks:
            movement = self.calculate_facial_movement(landmarks, previous_landmarks)
            if movement < 2.0:  # Pouco movimento facial
                demotivation_score += 0.2
                indicators.append("Expressão estática")

        indicators.extend(fatigue_indicators)
        indicators.extend(sadness_indicators)

        return min(demotivation_score, 1.0), list(set(indicators))

    def calculate_facial_movement(self, current_landmarks, previous_landmarks):
        """Calcula o movimento facial entre frames"""
        total_movement = 0
        for i in range(len(current_landmarks)):
            current_point = (current_landmarks[i].x, current_landmarks[i].y)
            previous_point = (previous_landmarks[i].x, previous_landmarks[i].y)
            total_movement += dist.euclidean(current_point, previous_point)
        return total_movement / len(current_landmarks)

    def detect_landmarks(self, frame):
        """Detecta landmarks faciais no frame"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.detector(gray)

        if len(faces) == 0:
            return None

        # Usa a primeira face detectada
        face = faces[0]
        landmarks = self.predictor(gray, face)

        return landmarks

    def analyze_expressions(self, frame, previous_landmarks=None):
        """Analisa todas as expressões faciais"""
        results = {
            "fatigue": {"score": 0, "indicators": [], "level": "Baixo"},
            "sadness": {"score": 0, "indicators": [], "level": "Baixo"},
            "demotivation": {"score": 0, "indicators": [], "level": "Baixo"},
            "landmarks": None,
        }

        landmarks = self.detect_landmarks(frame)
        if landmarks is None:
            return results

        results["landmarks"] = landmarks

        # Analisa cada expressão
        fatigue_score, fatigue_indicators = self.analyze_fatigue(landmarks)
        sadness_score, sadness_indicators = self.analyze_sadness(landmarks)
        demotivation_score, demotivation_indicators = self.analyze_demotivation(
            landmarks, previous_landmarks
        )

        # Atualiza resultados
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

        results["demotivation"] = {
            "score": demotivation_score,
            "indicators": demotivation_indicators,
            "level": self.get_expression_level(demotivation_score),
        }

        # Atualiza histórico
        self.update_expression_history(results)

        return results

    def get_expression_level(self, score):
        """Converte score em nível descritivo"""
        if score >= 0.7:
            return "Alto"
        elif score >= 0.4:
            return "Médio"
        else:
            return "Baixo"

    def update_expression_history(self, results):
        """Atualiza o histórico de expressões"""
        self.expression_history.append(
            {
                "timestamp": time.time(),
                "fatigue": results["fatigue"]["score"],
                "sadness": results["sadness"]["score"],
                "demotivation": results["demotivation"]["score"],
            }
        )

        # Mantém apenas o histórico mais recente
        if len(self.expression_history) > self.history_size:
            self.expression_history.pop(0)

    def get_trend_analysis(self):
        """Analisa tendências ao longo do tempo"""
        if len(self.expression_history) < 10:
            return "Dados insuficientes para análise de tendência"

        recent_fatigue = np.mean(
            [entry["fatigue"] for entry in self.expression_history[-10:]]
        )
        earlier_fatigue = np.mean(
            [entry["fatigue"] for entry in self.expression_history[:10]]
        )

        if recent_fatigue > earlier_fatigue + 0.2:
            return "Tendência de aumento de cansaço"
        elif recent_fatigue < earlier_fatigue - 0.2:
            return "Tendência de redução de cansaço"
        else:
            return "Estável"

    def draw_analysis(self, frame, results):
        """Desenha a análise no frame"""
        if results["landmarks"] is None:
            return frame

        # Desenha landmarks
        landmarks = results["landmarks"]
        for i in range(68):
            x, y = landmarks.part(i).x, landmarks.part(i).y
            cv2.circle(frame, (x, y), 1, (0, 255, 255), -1)

        # Adiciona informações textuais
        y_offset = 30
        expressions = [
            f"Cansaço: {results['fatigue']['level']} ({results['fatigue']['score']:.2f})",
            f"Tristeza: {results['sadness']['level']} ({results['sadness']['score']:.2f})",
            f"Desânimo: {results['demotivation']['level']} ({results['demotivation']['score']:.2f})",
        ]

        for expr in expressions:
            cv2.putText(
                frame,
                expr,
                (10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
            )
            y_offset += 20

        # Adiciona indicadores se houver
        if results["fatigue"]["indicators"]:
            indicators = ", ".join(results["fatigue"]["indicators"][:2])
            cv2.putText(
                frame,
                f"Indicadores: {indicators}",
                (10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (255, 255, 0),
                1,
            )

        return frame

    def cleanup(self):
        """Limpeza de recursos"""
        logging.info("Analisador de expressões finalizado")
