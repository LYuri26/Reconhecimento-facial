import cv2
import numpy as np
from config import Config


class FaceDetector:
    def __init__(self):
        self.config = Config()
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        self.eye_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_eye.xml"
        )

    def detect_faces(self, frame):
        """Detecta faces com pré-processamento e validação de olhos"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)

        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=self.config.MIN_FACE_SIZE,
            flags=cv2.CASCADE_SCALE_IMAGE,
        )

        # Filtra faces que possuem olhos detectados
        valid_faces = []
        for x, y, w, h in faces:
            roi_gray = gray[y : y + h, x : x + w]
            eyes = self.eye_cascade.detectMultiScale(roi_gray)
            if len(eyes) >= 1:  # Pelo menos um olho detectado
                valid_faces.append((x, y, w, h))

        return valid_faces
