import cv2
import numpy as np
import os
from config import Config


class FaceRecognizer:
    def __init__(self):
        self.config = Config()
        self.known_faces = []
        self.known_names = []
        self.known_ids = []

        # Inicializa o classificador Haar Cascade para detecção de rostos
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )

    def load_known_faces(self, pessoas):
        """Carrega rostos conhecidos a partir dos dados do banco"""
        self.known_faces = []
        self.known_names = []
        self.known_ids = []

        total_faces = 0

        for pessoa in pessoas:
            full_path = self.config.get_full_path(pessoa["Pasta"])

            if not os.path.exists(full_path):
                print(f"⚠️ Pasta não encontrada para {pessoa['Nome']}: {full_path}")
                continue

            for file in os.listdir(full_path):
                if file.lower().endswith((".png", ".jpg", ".jpeg")):
                    img_path = os.path.join(full_path, file)
                    img = cv2.imread(img_path)
                    if img is None:
                        continue

                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)

                    for x, y, w, h in faces:
                        face = gray[y : y + h, x : x + w]
                        face = cv2.resize(face, self.config.FACE_IMAGE_SIZE)
                        self.known_faces.append(face)
                        self.known_names.append(pessoa["Nome"])
                        self.known_ids.append(pessoa["ID"])
                        total_faces += 1

        print(f"✅ {total_faces} rostos carregados de {len(pessoas)} pessoas")
        return total_faces > 0

    def recognize_faces(self, frame, faces):
        """Reconhece faces em um frame"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        results = []

        for x, y, w, h in faces:
            face = gray[y : y + h, x : x + w]
            face = cv2.resize(face, self.config.FACE_IMAGE_SIZE)

            best_match_idx = None
            best_score = 0

            for i, known_face in enumerate(self.known_faces):
                score = np.mean(np.abs(face - known_face))
                if score > best_score:
                    best_score = score
                    best_match_idx = i

            if (
                best_match_idx is not None
                and best_score > self.config.FACE_RECOGNITION_THRESHOLD
            ):
                results.append(
                    {
                        "location": (x, y, x + w, y + h),
                        "name": self.known_names[best_match_idx],
                        "confidence": best_score,
                        "user_id": self.known_ids[best_match_idx],
                    }
                )
            else:
                results.append(
                    {
                        "location": (x, y, x + w, y + h),
                        "name": "Desconhecido",
                        "confidence": best_score if best_match_idx else 0,
                        "user_id": None,
                    }
                )

        return results
