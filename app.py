#!/usr/bin/env python3
import cv2
import face_recognition
import pickle
import os
import mysql.connector
from config import DB_CONFIG
import argparse
import sys
from datetime import datetime


class FaceSystem:
    def __init__(self):
        self.model_path = "face_recognition/models/face_model.dat"
        self.known_face_encodings = []
        self.known_face_names = []

    def train_from_database(self):
        """Treina o modelo com imagens do banco de dados MySQL"""
        try:
            conn = mysql.connector.connect(**DB_CONFIG)
            cursor = conn.cursor(dictionary=True)

            query = """
            SELECT c.nome, c.sobrenome, i.caminho_imagem 
            FROM cadastros c
            JOIN imagens_cadastro i ON c.id = i.cadastro_id
            """
            cursor.execute(query)

            image_data = cursor.fetchall()

            if not image_data:
                print("Nenhuma imagem encontrada no banco de dados!")
                return False

            print(f"Processando {len(image_data)} imagens...")

            for row in image_data:
                image_path = os.path.join("uploads", row["caminho_imagem"])
                if not os.path.exists(image_path):
                    print(f"Arquivo não encontrado: {image_path}")
                    continue

                image = face_recognition.load_image_file(image_path)
                face_encodings = face_recognition.face_encodings(image)

                if face_encodings:
                    self.known_face_encodings.append(face_encodings[0])
                    self.known_face_names.append(f"{row['nome']} {row['sobrenome']}")
                    print(
                        f"Processado: {row['nome']} {row['sobrenome']} - {image_path}"
                    )

            self.save_model()
            print(f"Modelo treinado com {len(self.known_face_names)} rostos conhecidos")
            return True

        except mysql.connector.Error as err:
            print(f"Erro no banco de dados: {err}")
            return False
        finally:
            if "conn" in locals() and conn.is_connected():
                cursor.close()
                conn.close()

    def save_model(self):
        """Salva o modelo treinado em arquivo"""
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        with open(self.model_path, "wb") as f:
            pickle.dump(
                {
                    "encodings": self.known_face_encodings,
                    "names": self.known_face_names,
                    "last_train": datetime.now().isoformat(),
                },
                f,
            )

    def recognize_faces(self, timeout=30):
        """Reconhece faces em tempo real usando a webcam"""
        if not os.path.exists(self.model_path):
            print("Modelo não encontrado. Execute o treinamento primeiro.")
            return False

        with open(self.model_path, "rb") as f:
            data = pickle.load(f)
            self.known_face_encodings = data["encodings"]
            self.known_face_names = data["names"]

        video_capture = cv2.VideoCapture(0)
        start_time = datetime.now()
        recognized_persons = set()

        print("Iniciando reconhecimento facial... Pressione 'q' para sair")

        while (datetime.now() - start_time).seconds < timeout:
            ret, frame = video_capture.read()
            if not ret:
                break

            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
            rgb_small_frame = small_frame[:, :, ::-1]

            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(
                rgb_small_frame, face_locations
            )

            for face_encoding in face_encodings:
                matches = face_recognition.compare_faces(
                    self.known_face_encodings, face_encoding
                )
                name = "Desconhecido"

                if True in matches:
                    first_match_index = matches.index(True)
                    name = self.known_face_names[first_match_index]
                    recognized_persons.add(name)

                # Desenha retângulos e nomes no frame
                for (top, right, bottom, left), name in zip(
                    face_locations, [name] * len(face_locations)
                ):
                    top *= 4
                    right *= 4
                    bottom *= 4
                    left *= 4
                    color = (0, 255, 0) if name != "Desconhecido" else (0, 0, 255)

                    cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
                    cv2.rectangle(
                        frame, (left, bottom - 35), (right, bottom), color, cv2.FILLED
                    )
                    cv2.putText(
                        frame,
                        name,
                        (left + 6, bottom - 6),
                        cv2.FONT_HERSHEY_DUPLEX,
                        1.0,
                        (255, 255, 255),
                        1,
                    )

            cv2.imshow("Reconhecimento Facial", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        video_capture.release()
        cv2.destroyAllWindows()
        return list(recognized_persons)


def main():
    parser = argparse.ArgumentParser(description="Sistema de Reconhecimento Facial")
    parser.add_argument(
        "--train", action="store_true", help="Treinar modelo com imagens do banco"
    )
    parser.add_argument(
        "--recognize", action="store_true", help="Iniciar reconhecimento em tempo real"
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=30,
        help="Tempo máximo de reconhecimento (segundos)",
    )

    args = parser.parse_args()

    system = FaceSystem()

    if args.train:
        print("Iniciando treinamento...")
        success = system.train_from_database()
        sys.exit(0 if success else 1)
    elif args.recognize:
        print("Iniciando reconhecimento...")
        recognized = system.recognize_faces(args.timeout)
        print(
            "Pessoas reconhecidas:",
            ", ".join(recognized) if recognized else "Nenhuma pessoa reconhecida",
        )
    else:
        parser.print_help()


def recognize_faces(self, timeout=30):
    """Reconhece faces em tempo real usando a webcam"""
    if not os.path.exists(self.model_path):
        print("Modelo não encontrado. Execute o treinamento primeiro.")
        return False

    with open(self.model_path, "rb") as f:
        data = pickle.load(f)
        self.known_face_encodings = data["encodings"]
        self.known_face_names = data["names"]

    video_capture = cv2.VideoCapture(0)
    start_time = datetime.now()
    recognized_persons = set()

    print("Iniciando reconhecimento facial... Pressione 'q' para sair")

    while (datetime.now() - start_time).seconds < timeout:
        ret, frame = video_capture.read()
        if not ret:
            break

        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = small_frame[:, :, ::-1]

        face_locations = face_recognition.face_locations(rgb_small_frame)

        # Modificação aqui
        face_encodings = face_recognition.face_encodings(
            rgb_small_frame, known_face_locations=face_locations
        )

        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(
                self.known_face_encodings, face_encoding
            )
            name = "Desconhecido"

            if True in matches:
                first_match_index = matches.index(True)
                name = self.known_face_names[first_match_index]
                recognized_persons.add(name)

            # Restante do código de desenho permanece igual
            for (top, right, bottom, left), name in zip(
                face_locations, [name] * len(face_locations)
            ):
                top *= 4
                right *= 4
                bottom *= 4
                left *= 4
                color = (0, 255, 0) if name != "Desconhecido" else (0, 0, 255)

                cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
                cv2.rectangle(
                    frame, (left, bottom - 35), (right, bottom), color, cv2.FILLED
                )
                cv2.putText(
                    frame,
                    name,
                    (left + 6, bottom - 6),
                    cv2.FONT_HERSHEY_DUPLEX,
                    1.0,
                    (255, 255, 255),
                    1,
                )

        cv2.imshow("Reconhecimento Facial", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    video_capture.release()
    cv2.destroyAllWindows()
    return list(recognized_persons)


if __name__ == "__main__":
    main()
