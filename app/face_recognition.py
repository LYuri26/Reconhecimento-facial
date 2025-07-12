import os
import cv2
import numpy as np
from datetime import datetime
from .database import Database
from .utils import load_classifier


class FaceRecognition:
    def __init__(self):
        self.db = Database()
        self.initialize_classifiers()

    def initialize_classifiers(self):
        """Inicializa os classificadores de reconhecimento facial"""
        base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        cascade_path = os.path.join(base_path, "app", "cascade")
        classifier_path = os.path.join(base_path, "app", "classifier")

        # Carrega Haar Cascades
        self.face_cascade = cv2.CascadeClassifier(
            os.path.join(cascade_path, "haarcascade_frontalface_default.xml")
        )

        # Carrega classificadores treinados
        self.classifier_lbph = cv2.face.LBPHFaceRecognizer_create()
        self.classifier_lbph.read(
            os.path.join(classifier_path, "classificadorLBPH.yml")
        )

        self.classifier_eigen = cv2.face.EigenFaceRecognizer_create()
        self.classifier_eigen.read(
            os.path.join(classifier_path, "classificadorEigen.yml")
        )

    def recognize_from_camera(self):
        """Reconhecimento em tempo real usando webcam com máscaras visuais"""
        cap = cv2.VideoCapture(0)
        font = cv2.FONT_HERSHEY_SIMPLEX

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100)
            )

            for x, y, w, h in faces:
                face = gray[y : y + h, x : x + w]
                face = cv2.resize(face, (220, 220))

                # Reconhecimento LBPH
                label_lbph, conf_lbph = self.classifier_lbph.predict(face)

                # Reconhecimento Eigen
                label_eigen, conf_eigen = self.classifier_eigen.predict(face)

                # Lógica de decisão
                if label_lbph == label_eigen and conf_lbph < 100 and conf_eigen < 100:
                    pessoa = self.db.obter_pessoa(label_lbph)
                    nome = f"{pessoa['nome']} {pessoa['sobrenome']}"
                    conf = (conf_lbph + conf_eigen) / 2
                    cor = (0, 255, 0)  # Verde para reconhecido
                else:
                    nome = "Desconhecido"
                    conf = max(conf_lbph, conf_eigen)
                    cor = (0, 0, 255)  # Vermelho para desconhecido

                # Desenha retângulo ao redor do rosto
                cv2.rectangle(frame, (x, y), (x + w, y + h), cor, 2)

                # Desenha fundo para o texto
                cv2.rectangle(frame, (x, y - 40), (x + w, y), cor, -1)

                # Escreve o nome da pessoa
                cv2.putText(frame, nome, (x, y - 10), font, 0.8, (255, 255, 255), 2)

                # Escreve a confiança abaixo do retângulo
                cv2.putText(
                    frame, f"Confianca: {conf:.2f}%", (x, y + h + 25), font, 0.6, cor, 1
                )

                # Registra no banco se reconheceu
                if "Desconhecido" not in nome:
                    self.db.registrar_reconhecimento(label_lbph, conf)

            # Mostra o frame com as máscaras
            cv2.imshow("Reconhecimento Facial - Pressione Q para sair", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()
