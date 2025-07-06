import cv2
import numpy as np
import mysql.connector
from mysql.connector import Error
import os


class FacialRecognitionSystem:
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        self.known_faces = []
        self.known_names = []
        self.known_ids = []
        self.threshold = 0.6

        # Configuração do caminho base relativo ao projeto PHP
        self.base_dir = (
            "/home/lenon/Documentos/GitHub/Reconhecimento-facial/cadastro-web-php"
        )

        # Configuração do banco para XAMPP (root sem senha)
        self.db_config = {
            "host": "localhost",
            "database": "Catraca",
            "user": "root",
            "password": "",
            "port": 3306,
        }

    def connect_db(self):
        try:
            self.connection = mysql.connector.connect(**self.db_config)
            print("✅ Conexão com o banco de dados estabelecida")
            return True
        except Error as e:
            print(f"❌ Erro de conexão: {e}")
            return False

    def get_full_path(self, relative_path):
        """Converte caminho relativo para absoluto baseado no diretório do projeto PHP"""
        # Remove 'imagens/' do início se existir
        if relative_path.startswith("imagens/"):
            relative_path = relative_path[8:]
        return os.path.join(self.base_dir, "imagens", relative_path)

    def load_known_faces(self):
        try:
            cursor = self.connection.cursor(dictionary=True)
            cursor.execute("SELECT ID, Nome, Pasta FROM Pessoas")
            pessoas = cursor.fetchall()

            if not pessoas:
                print("⚠️ Nenhuma pessoa cadastrada no banco de dados")
                return False

            total_faces = 0
            for pessoa in pessoas:
                if not pessoa["Pasta"]:
                    print(f"⚠️ Pasta não definida para {pessoa['Nome']}")
                    continue

                full_path = self.get_full_path(pessoa["Pasta"])
                print(f"🔍 Procurando imagens em: {full_path}")

                if not os.path.exists(full_path):
                    print(f"⚠️ Pasta não encontrada para {pessoa['Nome']}: {full_path}")
                    continue

                # Carrega todas as imagens da pasta
                for file in os.listdir(full_path):
                    if file.lower().endswith((".png", ".jpg", ".jpeg")):
                        img_path = os.path.join(full_path, file)
                        img = cv2.imread(img_path)
                        if img is None:
                            print(f"⚠️ Não foi possível ler a imagem: {img_path}")
                            continue

                        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)

                        for x, y, w, h in faces:
                            face = gray[y : y + h, x : x + w]
                            face = cv2.resize(face, (100, 100))
                            self.known_faces.append(face)
                            self.known_names.append(pessoa["Nome"])
                            self.known_ids.append(pessoa["ID"])
                            total_faces += 1

            print(f"✅ {total_faces} rostos carregados de {len(pessoas)} pessoas")
            return total_faces > 0

        except Exception as e:
            print(f"❌ Erro ao carregar rostos: {e}")
            return False

    def recognize_face(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)

        results = []
        for x, y, w, h in faces:
            face = gray[y : y + h, x : x + w]
            face = cv2.resize(face, (100, 100))

            best_match_idx = None
            best_score = 0

            for i, known_face in enumerate(self.known_faces):
                score = np.mean(np.abs(face - known_face))
                if score > best_score:
                    best_score = score
                    best_match_idx = i

            if best_match_idx is not None and best_score > self.threshold:
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

    def registrar_acesso(self, user_id, direction):
        try:
            cursor = self.connection.cursor()
            cursor.execute(
                "INSERT INTO Acessos (user_id, direction) VALUES (%s, %s)",
                (user_id, direction),
            )
            self.connection.commit()
            print(f"✅ Acesso registrado: ID {user_id} - {direction}")
            return True
        except Error as e:
            print(f"❌ Erro ao registrar acesso: {e}")
            return False

    def run(self):
        if not self.connect_db():
            return

        if not self.load_known_faces():
            print("⚠️ Nenhum rosto conhecido carregado. Cadastre pessoas primeiro.")
            return

        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("❌ Não foi possível acessar a câmera")
            return

        print("\n✅ Sistema iniciado. Pressione:")
        print(" - 'e' para registrar entrada")
        print(" - 's' para registrar saída")
        print(" - 'r' para recarregar rostos")
        print(" - 'q' para sair")

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("⚠️ Erro ao capturar frame")
                    break

                results = self.recognize_face(frame)

                for result in results:
                    (x1, y1, x2, y2) = result["location"]
                    name = result["name"]
                    confidence = result["confidence"]

                    color = (0, 255, 0) if name != "Desconhecido" else (0, 0, 255)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    text = f"{name} ({confidence:.2f})"
                    cv2.putText(
                        frame,
                        text,
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        color,
                        2,
                    )

                cv2.imshow("Reconhecimento Facial - Catraca", frame)

                key = cv2.waitKey(1)
                if key & 0xFF == ord("q"):
                    break
                elif key & 0xFF == ord("e") and results and results[0]["user_id"]:
                    self.registrar_acesso(results[0]["user_id"], "entrada")
                elif key & 0xFF == ord("s") and results and results[0]["user_id"]:
                    self.registrar_acesso(results[0]["user_id"], "saida")
                elif key & 0xFF == ord("r"):
                    self.known_faces = []
                    self.known_names = []
                    self.known_ids = []
                    if self.load_known_faces():
                        print("✅ Rostos recarregados com sucesso")

        finally:
            cap.release()
            cv2.destroyAllWindows()
            if hasattr(self, "connection") and self.connection.is_connected():
                self.connection.close()
                print("✅ Conexão com o banco de dados encerrada")
            print("✅ Sistema encerrado")


if __name__ == "__main__":
    system = FacialRecognitionSystem()
    system.run()
