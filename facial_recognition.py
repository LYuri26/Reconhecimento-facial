import os
import cv2
import numpy as np
import time
from src.database.conexao import get_db_connection

# Configurações
BASE_DIR = os.path.expanduser("~/Documentos/GitHub/Reconhecimento-facial")
UPLOADS_DIR = os.path.join(BASE_DIR, "uploads")
HAARCASCADE_PATH = os.path.join(
    os.path.dirname(__file__), "haarcascade/haarcascade_frontalface_default.xml"
)


class FacialRecognition:
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(HAARCASCADE_PATH)
        # Parâmetros otimizados para poucas imagens
        self.recognizer = cv2.face.LBPHFaceRecognizer_create(
            radius=1, neighbors=8, grid_x=8, grid_y=8, threshold=80
        )
        self.cap = cv2.VideoCapture(0)
        self.trained = False
        self.pessoas = {}

        # Pré-aquecimento da câmera
        for _ in range(5):
            self.cap.read()

        self.train_model()

    def get_imagens_treinamento(self):
        conn = get_db_connection()
        if conn:
            try:
                cursor = conn.cursor(dictionary=True)
                query = """
                    SELECT c.id, c.nome, c.sobrenome, ic.caminho_imagem 
                    FROM cadastros c
                    JOIN imagens_cadastro ic ON c.id = ic.cadastro_id
                    ORDER BY c.id, ic.id
                """
                cursor.execute(query)
                resultados = cursor.fetchall()

                for pessoa in resultados:
                    if pessoa["id"] not in self.pessoas:
                        self.pessoas[pessoa["id"]] = {
                            "nome": f"{pessoa['nome']} {pessoa['sobrenome']}",
                            "qtd_imagens": 0,
                        }
                    self.pessoas[pessoa["id"]]["qtd_imagens"] += 1

                return resultados
            except Exception as e:
                print(f"Erro ao buscar imagens: {e}")
                return []
            finally:
                cursor.close()
                conn.close()
        return []

    def train_model(self):
        imagens = self.get_imagens_treinamento()

        if not imagens:
            print("❌ Nenhuma imagem encontrada no banco.")
            return

        faces = []
        ids = []

        for img in imagens:
            image_path = os.path.join(UPLOADS_DIR, img["caminho_imagem"])

            if not os.path.exists(image_path):
                print(f"[AVISO] Imagem não encontrada: {image_path}")
                continue

            img_array = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if img_array is None:
                print(f"[AVISO] Imagem inválida: {image_path}")
                continue

            # Redimensiona e equaliza
            face_img = cv2.resize(img_array, (200, 200))
            face_img = cv2.equalizeHist(face_img)

            faces.append(face_img)
            ids.append(img["id"])

        if len(faces) > 0:
            self.recognizer.train(faces, np.array(ids))
            self.trained = True
            print(
                f"✅ Modelo treinado com {len(faces)} imagens de {len(self.pessoas)} pessoas."
            )
        else:
            print("❌ Nenhuma face válida para treinamento.")

    def reconhecer(self):
        if not self.trained:
            print("❌ Modelo não treinado.")
            return

        print("🟢 Iniciando reconhecimento... Pressione 'q' para sair.")

        # Variáveis para suavização
        last_ids = []
        last_confidences = []
        smoothing_window = 5  # Número de frames para média móvel

        while True:
            ret, frame = self.cap.read()
            if not ret:
                continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.equalizeHist(gray)
            faces = self.face_cascade.detectMultiScale(gray, 1.2, 5)

            for x, y, w, h in faces:
                face_roi = gray[y : y + h, x : x + w]
                face_roi = cv2.resize(face_roi, (200, 200))
                face_roi = cv2.equalizeHist(face_roi)

                id_pred, conf = self.recognizer.predict(face_roi)

                # Suavização dos resultados
                last_ids.append(id_pred)
                last_confidences.append(conf)
                if len(last_ids) > smoothing_window:
                    last_ids.pop(0)
                    last_confidences.pop(0)

                # Usa moda para ID e média para confiança
                if len(last_ids) == smoothing_window:
                    id_pred = max(set(last_ids), key=last_ids.count)
                    conf = np.mean(last_confidences)

                # Ajuste dinâmico do limiar
                limiar = 80  # Base
                if w < 100:
                    limiar += 15  # Rostos pequenos
                if conf > 100:
                    limiar += 20  # Muito diferente

                if conf < limiar and id_pred in self.pessoas:
                    nome = self.pessoas[id_pred]["nome"]
                    status = f"{nome} ({conf:.1f})"
                    cor = (0, 255, 0)
                else:
                    status = "Desconhecido"
                    cor = (0, 0, 255)

                cv2.rectangle(frame, (x, y), (x + w, y + h), cor, 2)
                cv2.putText(
                    frame, status, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, cor, 2
                )

            cv2.imshow("Reconhecimento Facial", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        self.cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    fr = FacialRecognition()
    fr.reconhecer()
