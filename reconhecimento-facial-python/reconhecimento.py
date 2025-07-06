import cv2
import numpy as np
import os


class FacialRecognizer:
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        self.known_faces = []
        self.known_names = []
        self.threshold = 0.6

    def load_known_faces(self, faces_dir="USUARIO"):
        try:
            for name in os.listdir(faces_dir):
                user_dir = os.path.join(faces_dir, name)
                if os.path.isdir(user_dir):
                    for file in os.listdir(user_dir):
                        img_path = os.path.join(user_dir, file)
                        img = cv2.imread(img_path)
                        if img is None:
                            continue
                        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
                        for x, y, w, h in faces:
                            face = gray[y : y + h, x : x + w]
                            face = cv2.resize(face, (100, 100))
                            self.known_faces.append(face)
                            self.known_names.append(name)
            return True
        except Exception as e:
            print(f"Erro ao carregar rostos: {e}")
            return False

    def recognize_face(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)

        results = []
        for x, y, w, h in faces:
            face = gray[y : y + h, x : x + w]
            face = cv2.resize(face, (100, 100))

            best_match = None
            best_score = 0

            for i, known_face in enumerate(self.known_faces):
                score = np.mean(np.abs(face - known_face))
                if score > best_score:
                    best_score = score
                    best_match = self.known_names[i]

            name = best_match if best_score > self.threshold else "Desconhecido"
            results.append(
                {
                    "location": (x, y, x + w, y + h),
                    "name": name,
                    "confidence": best_score,
                }
            )

        return results
