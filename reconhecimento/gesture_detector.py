# gesture_detector.py - Detecção de gesto de punho cerrado (alerta) com MediaPipe
import cv2
import mediapipe as mp
import numpy as np
import time
import logging


class GestureDetector:
    """
    Detecta o gesto de punho cerrado (mão aberta → polegar dobrado → dedos fechando → punho)
    usando MediaPipe Hands. Implementa uma máquina de estados com timeouts de 60 segundos
    por etapa. A perda da mão não reseta o estado; apenas pausa a progressão.
    """

    def __init__(self, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self.mp_draw = mp.solutions.drawing_utils

        # Estados da máquina
        self.STATE_IDLE = 0
        self.STATE_HAND_OPEN = 1
        self.STATE_THUMB_FOLDED = 2
        self.STATE_FINGERS_CLOSING = 3
        self.STATE_FIST = 4

        self.state = self.STATE_IDLE
        self.state_start_time = None
        self.last_gesture_time = 0
        self.cooldown = 2.0  # segundos para evitar repetição

        # Índices dos landmarks (MediaPipe)
        self.thumb_tip = 4
        self.thumb_ip = 3
        self.thumb_mcp = 2
        self.index_tip = 8
        self.index_mcp = 5
        self.middle_tip = 12
        self.middle_mcp = 9
        self.ring_tip = 16
        self.pinky_tip = 20
        self.pinky_mcp = 17
        self.wrist = 0

        # Limiares ajustados para maior tolerância
        self.OPEN_THRESHOLD = 0.2
        self.THUMB_FOLD_THRESHOLD = 0.2
        self.CLOSING_THRESHOLD = 0.2  # aumentado de 0.18 para 0.2
        self.FIST_THRESHOLD = 0.12  # reduzido de 0.15 para 0.12
        self.FINGER_SEPARATION = 0.02

        # Timeouts de 60 segundos para cada etapa (exceto o último)
        self.timeout_open = 60.0
        self.timeout_thumb = 60.0
        self.timeout_closing = 60.0
        self.timeout_fist = 0.5

        logging.info(
            "GestureDetector inicializado com MediaPipe Hands (timeouts de 60s por etapa)"
        )

    def _distance(self, p1, p2):
        """Distância euclidiana entre dois landmarks (coordenadas normalizadas)."""
        return np.linalg.norm(np.array([p1.x, p1.y]) - np.array([p2.x, p2.y]))

    def _is_hand_open(self, landmarks):
        """
        Verifica se a mão está aberta:
        - Todos os dedos (exceto polegar) estão estendidos (pontas distantes da palma)
        - Há separação entre os dedos indicador e médio
        """
        palm_base = landmarks.landmark[self.middle_mcp]

        tips = [self.index_tip, self.middle_tip, self.ring_tip, self.pinky_tip]
        for tip in tips:
            if self._distance(landmarks.landmark[tip], palm_base) < self.OPEN_THRESHOLD:
                return False

        # Verifica separação entre indicador e médio
        index_tip = landmarks.landmark[self.index_tip]
        middle_tip = landmarks.landmark[self.middle_tip]
        if self._distance(index_tip, middle_tip) < self.FINGER_SEPARATION:
            return False

        return True

    def _is_thumb_folded(self, landmarks):
        """
        Verifica se o polegar está dobrado sobre a palma:
        - Ponta do polegar próxima à base do mindinho
        - E a ponta do polegar está abaixo (em y) da base do indicador
        """
        thumb_tip = landmarks.landmark[self.thumb_tip]
        pinky_mcp = landmarks.landmark[self.pinky_mcp]
        d = self._distance(thumb_tip, pinky_mcp)

        index_mcp = landmarks.landmark[self.index_mcp]
        return d < self.THUMB_FOLD_THRESHOLD and thumb_tip.y > index_mcp.y

    def _are_fingers_closing(self, landmarks):
        """
        Verifica se os dedos (indicador, médio, anelar, mínimo) estão parcialmente dobrados.
        Pelo menos 3 dedos com ponta próxima à palma.
        """
        palm_base = landmarks.landmark[self.middle_mcp]
        tips = [self.index_tip, self.middle_tip, self.ring_tip, self.pinky_tip]
        closed_count = 0
        for tip in tips:
            if (
                self._distance(landmarks.landmark[tip], palm_base)
                < self.CLOSING_THRESHOLD
            ):
                closed_count += 1
        return closed_count >= 3

    def _is_fist(self, landmarks):
        """
        Verifica se a mão está fechada em punho de forma mais tolerante:
        - Todos os dedos (indicador, médio, anelar, mínimo) estão com pontas próximas à palma
        - Polegar está por dentro (ponta próxima à base do indicador)
        Usa um limiar mais baixo e verifica a média das distâncias.
        """
        palm_base = landmarks.landmark[self.middle_mcp]
        tips = [self.index_tip, self.middle_tip, self.ring_tip, self.pinky_tip]
        distances = [self._distance(landmarks.landmark[tip], palm_base) for tip in tips]
        avg_distance = np.mean(distances)
        max_distance = max(distances)

        # Condição: a média e o máximo das distâncias estão abaixo do limiar
        if avg_distance > self.FIST_THRESHOLD * 1.5:  # tolerância extra
            return False

        # Polegar por dentro: ponta do polegar próxima à base do indicador
        thumb_tip = landmarks.landmark[self.thumb_tip]
        index_mcp = landmarks.landmark[self.index_mcp]
        thumb_distance = self._distance(thumb_tip, index_mcp)

        if thumb_distance > self.FIST_THRESHOLD * 2:
            return False

        # Debug opcional (pode ser comentado)
        print(
            f"DEBUG FIST: avg_dist={avg_distance:.3f}, max_dist={max_distance:.3f}, thumb_dist={thumb_distance:.3f}"
        )
        return True

    def detect_gesture(self, frame):
        """
        Processa um frame e retorna:
            - gesture_complete: bool indicando se o gesto foi detectado neste frame
            - annotated_frame: frame com desenhos dos landmarks
        """
        gesture_complete = False
        annotated_frame = frame.copy()
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb)

        current_time = time.time()

        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            self.mp_draw.draw_landmarks(
                annotated_frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS
            )

            # Máquina de estados (progressão)
            if self.state == self.STATE_IDLE:
                if self._is_hand_open(hand_landmarks):
                    print("DEBUG: Mão aberta detectada - início do gesto")
                    self.state = self.STATE_HAND_OPEN
                    self.state_start_time = current_time
            elif self.state == self.STATE_HAND_OPEN:
                if self._is_thumb_folded(hand_landmarks):
                    print("DEBUG: Polegar dobrado")
                    self.state = self.STATE_THUMB_FOLDED
                    self.state_start_time = current_time
                elif current_time - self.state_start_time > self.timeout_open:
                    print(f"DEBUG: Timeout mão aberta ({self.timeout_open}s), reset")
                    self.reset()
            elif self.state == self.STATE_THUMB_FOLDED:
                if self._are_fingers_closing(hand_landmarks):
                    print("DEBUG: Dedos fechando")
                    self.state = self.STATE_FINGERS_CLOSING
                    self.state_start_time = current_time
                elif current_time - self.state_start_time > self.timeout_thumb:
                    print(f"DEBUG: Timeout polegar ({self.timeout_thumb}s), reset")
                    self.reset()
            elif self.state == self.STATE_FINGERS_CLOSING:
                if self._is_fist(hand_landmarks):
                    print("DEBUG: Punho fechado - GESTO COMPLETO!")
                    self.state = self.STATE_FIST
                    self.state_start_time = current_time
                    gesture_complete = True
                elif current_time - self.state_start_time > self.timeout_closing:
                    print(f"DEBUG: Timeout fechamento ({self.timeout_closing}s), reset")
                    self.reset()
            elif self.state == self.STATE_FIST:
                if current_time - self.state_start_time > self.timeout_fist:
                    self.reset()
        else:
            # Mão não detectada: NÃO reseta o estado, apenas informa (opcional)
            if self.state != self.STATE_IDLE:
                # Mantém o estado, o timeout continuará contando
                print(
                    "DEBUG: Mão perdida, mas estado mantido. Tempo restante ainda conta."
                )
                # Não faz nada, apenas aguarda a mão voltar

        if gesture_complete:
            now = time.time()
            if now - self.last_gesture_time > self.cooldown:
                self.last_gesture_time = now
                return True, annotated_frame
            else:
                return False, annotated_frame
        return False, annotated_frame

    def reset(self):
        self.state = self.STATE_IDLE
        self.state_start_time = None
        print("DEBUG: Reset")
