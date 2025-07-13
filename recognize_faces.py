import os
import cv2
import numpy as np
import mysql.connector
from mysql.connector import Error


class FaceRecognizer:
    def __init__(self):
        # Configurações do banco de dados
        self.DB_CONFIG = {
            "host": "localhost",
            "user": "root",
            "password": "",
            "database": "reconhecimento_facial",
        }

        # Carrega o modelo treinado
        self.recognizer = cv2.face.LBPHFaceRecognizer_create()
        self.recognizer.read("model/trained_model.yml")

        # Inicializa detector de faces
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )

        # Configurações da webcam
        self.cap = cv2.VideoCapture(0)
        self.cap.set(3, 640)  # Largura
        self.cap.set(4, 480)  # Altura

    def get_db_connection(self):
        """Estabelece conexão com o banco de dados"""
        try:
            return mysql.connector.connect(**self.DB_CONFIG)
        except Error as e:
            print(f"Erro ao conectar ao MySQL: {e}")
            return None

    def get_user_info(self, user_id):
        """Obtém informações do usuário"""
        conn = self.get_db_connection()
        if conn is None:
            return None

        try:
            cursor = conn.cursor(dictionary=True)
            cursor.execute(
                """
                SELECT id, nome, sobrenome 
                FROM cadastros 
                WHERE id = %s
            """,
                (user_id,),
            )
            return cursor.fetchone()
        except Error as e:
            print(f"Erro ao buscar usuário: {e}")
            return None
        finally:
            if conn and conn.is_connected():
                conn.close()

    def recognize_faces(self, frame):
        """Reconhece faces em um frame da webcam"""
        try:
            # Converte para escala de cinza
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Detecta faces
            faces = self.face_cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
            )

            if len(faces) == 0:  # Verificação correta para arrays NumPy
                return None

            results = []
            for x, y, w, h in faces:
                # Extrai e redimensiona a face
                face_img = gray[y : y + h, x : x + w]
                face_img = cv2.resize(face_img, (200, 200))

                # Faz a predição
                label, confidence = self.recognizer.predict(face_img)

                # Obtém informações do usuário
                user_info = self.get_user_info(label)
                if user_info is not None:  # Verificação explícita de None
                    results.append(
                        {
                            "user": user_info,
                            "confidence": confidence,
                            "position": (x, y, w, h),
                        }
                    )

            return results if len(results) > 0 else None  # Verificação correta

        except Exception as e:
            print(f"Erro durante o reconhecimento: {e}")
            return None

    def draw_boxes(self, frame, results):
        """Desenha retângulos e informações na imagem"""
        for result in results:
            x, y, w, h = result["position"]
            user = result["user"]
            confidence = result["confidence"]

            # Define a cor com base na confiança
            if confidence < 50:
                color = (0, 255, 0)  # Verde (confiável)
            elif confidence < 70:
                color = (0, 255, 255)  # Amarelo (moderado)
            else:
                color = (0, 0, 255)  # Vermelho (não confiável)

            # Desenha o retângulo
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

            # Exibe informações
            text = f"{user['nome']} {user['sobrenome']} ({confidence:.1f})"
            cv2.putText(
                frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2
            )

    def run_webcam_recognition(self):
        """Executa o reconhecimento em tempo real pela webcam"""
        print("\nIniciando reconhecimento facial pela webcam...")
        print("Pressione 'q' para sair")

        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("Erro ao capturar frame da webcam")
                break

            # Realiza o reconhecimento
            results = self.recognize_faces(frame)

            # Desenha os resultados
            if results is not None:  # Verificação explícita
                self.draw_boxes(frame, results)

            # Mostra o frame
            cv2.imshow("Reconhecimento Facial", frame)

            # Verifica se o usuário quer sair
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        # Libera os recursos
        self.cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    recognizer = FaceRecognizer()
    recognizer.run_webcam_recognition()
