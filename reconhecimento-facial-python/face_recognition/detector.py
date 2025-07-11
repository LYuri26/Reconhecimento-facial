import cv2
import numpy as np
import face_recognition  # Adicionar este import


class FaceDetector:
    def __init__(self):
        self.config = Config()
        self.detection_method = self.config.RECOGNITION_MODEL  # "hog" ou "cnn"

    def detect_faces(self, frame):
        """Detecta faces usando face_recognition"""
        try:
            # Convertendo de BGR (OpenCV) para RGB (face_recognition)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Detectar faces
            face_locations = face_recognition.face_locations(
                rgb_frame, model=self.detection_method
            )

            # Converter coordenadas para o formato (x, y, w, h)
            faces = []
            for top, right, bottom, left in face_locations:
                faces.append((left, top, right - left, bottom - top))

            return faces
        except Exception as e:
            print(f"⚠️ Erro na detecção de faces: {str(e)}")
            return []

    def detect_faces(self, frame):
        """Detecta faces com pré-processamento e validação de olhos"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)

        if self.dnn_detector:
            faces = self._detect_faces_dnn(frame)
        else:
            faces = self._detect_faces_haar(gray)

        # Filtra faces que possuem olhos detectados
        valid_faces = []
        for x, y, w, h in faces:
            roi_gray = gray[y : y + h, x : x + w]
            eyes = self.eye_cascade.detectMultiScale(roi_gray)
            if len(eyes) >= 1:  # Pelo menos um olho detectado
                valid_faces.append((x, y, w, h))

        return valid_faces

    def _detect_faces_dnn(self, frame):
        """Detecta faces usando DNN"""
        h, w = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(
            cv2.resize(frame, (300, 300)),
            1.0,
            (300, 300),
            [104, 117, 123],
            False,
            False,
        )

        self.dnn_detector.setInput(blob)
        detections = self.dnn_detector.forward()

        faces = []
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > self.config.DETECTION_CONFIDENCE:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                x1, y1, x2, y2 = box.astype("int")
                # Garante que as coordenadas estão dentro dos limites do frame
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                faces.append((x1, y1, x2 - x1, y2 - y1))

        return faces

    def _detect_faces_haar(self, gray_frame):
        """Detecta faces usando Haar Cascade"""
        faces = self.face_cascade.detectMultiScale(
            gray_frame,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=self.config.MIN_FACE_SIZE,
            flags=cv2.CASCADE_SCALE_IMAGE,
        )
        return faces
