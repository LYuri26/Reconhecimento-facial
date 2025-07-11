import cv2
import numpy as np
from config import Config


class BasicRecognizer:
    def __init__(self):
        self.config = Config()
        self.known_faces = []
        self.known_names = []

    def load_known_faces(self, pessoas):
        # Implementação básica de carregamento
        pass

    def recognize_faces(self, frame, faces):
        # Implementação básica de reconhecimento
        results = []
        for x, y, w, h in faces:
            results.append(
                {
                    "location": (x, y, x + w, y + h),
                    "name": "Desconhecido",
                    "confidence": 0,
                    "user_id": None,
                    "danger_level": "BAIXO",
                }
            )
        return results
