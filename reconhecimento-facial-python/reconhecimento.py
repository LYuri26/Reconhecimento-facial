import cv2
import face_recognition
import pickle
import os
from utils.mysql_utils import MySQLUtils


class FacialRecognition:
    def __init__(self):
        self.db = MySQLUtils()
        self.db.connect()
        self.known_encodings = []
        self.known_names = []
        self.load_encodings()

    def load_encodings(self):
        try:
            with open("modelos/encodings.pkl", "rb") as f:
                data = pickle.load(f)
                self.known_encodings = data["encodings"]
                self.known_names = data["names"]
        except FileNotFoundError:
            self.update_encodings()

    def update_encodings(self):
        cursor = self.db.connection.cursor(dictionary=True)
        cursor.execute("SELECT Nome, Foto FROM Pessoas WHERE Foto IS NOT NULL")

        encodings = []
        names = []

        for row in cursor:
            image_path = os.path.join(
                "cadastro-web-php/imagens/rostos_salvos", row["Foto"]
            )
            if os.path.exists(image_path):
                image = face_recognition.load_image_file(image_path)
                face_locations = face_recognition.face_locations(image)
                if face_locations:
                    encoding = face_recognition.face_encodings(image, face_locations)[0]
                    encodings.append(encoding)
                    names.append(row["Nome"])

        data = {"encodings": encodings, "names": names}
        with open("modelos/encodings.pkl", "wb") as f:
            pickle.dump(data, f)

        self.known_encodings = encodings
        self.known_names = names
        cursor.close()

    def recognize_face(self, frame):
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = small_frame[:, :, ::-1]

        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(
            rgb_small_frame, face_locations
        )

        names = []
        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(
                self.known_encodings, face_encoding
            )
            name = "Desconhecido"

            if True in matches:
                first_match_index = matches.index(True)
                name = self.known_names[first_match_index]

            names.append(name)

        return face_locations, names
