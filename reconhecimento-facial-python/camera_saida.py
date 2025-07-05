import cv2
import time
from reconhecimento import FacialRecognition
from utils.mysql_utils import MySQLUtils


def main():
    fr = FacialRecognition()
    db = MySQLUtils()
    db.connect()

    cap = cv2.VideoCapture(1)  # Usar índice 1 para a segunda câmera

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        face_locations, names = fr.recognize_face(frame)

        for (top, right, bottom, left), name in zip(face_locations, names):
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4

            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
            cv2.putText(
                frame,
                name,
                (left + 6, bottom - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
            )

            if name != "Desconhecido":
                # Registrar saída no banco de dados
                db.register_access(name, "saida")
                print(f"Registrada saída: {name}")
                # Aqui você pode adicionar lógica para acionar a catraca

        cv2.imshow("Camera Saída", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    db.close()


if __name__ == "__main__":
    main()
