import cv2
import numpy as np
import pickle
import subprocess
import mysql.connector
from deepface import DeepFace
from mysql.connector import Error
import threading
import queue
import time
import tempfile


class FaceRecognizer:
    def __init__(self):
        self.MODEL_PATH = "model/deepface_model.pkl"
        self.THRESHOLD = 0.65
        self.DETECTOR = "opencv"
        self.width = 1280  # Reduzindo a resolução para melhor performance
        self.height = 720
        self.frame_queue = queue.Queue(maxsize=2)  # Limitar o buffer de frames
        self.running = True

        # Carregar modelo em thread separada para não bloquear a inicialização
        self.model_loaded = False
        threading.Thread(target=self.load_model, daemon=True).start()

        # Conexão com banco de dados
        self.db_connection = self.create_db_connection()

        # Configuração de streaming otimizada
        rtsp_url = "rtsp://admin:Evento0128@192.168.1.101:559/Streaming/Channels/101"
        ffmpeg_cmd = [
            "ffmpeg",
            "-rtsp_transport",
            "tcp",
            "-i",
            rtsp_url,
            "-loglevel",
            "quiet",  # Reduzir logs do ffmpeg
            "-f",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "-s",
            f"{self.width}x{self.height}",  # Definir resolução fixa
            "-r",
            "15",  # Limitar framerate
            "-",
        ]
        self.proc = subprocess.Popen(
            ffmpeg_cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=10**6
        )

        # Thread para captura de frames
        threading.Thread(target=self.capture_frames, daemon=True).start()

    def create_db_connection(self):
        try:
            return mysql.connector.connect(
                host="localhost",
                user="root",
                password="",
                database="reconhecimento_facial",
            )
        except Error as e:
            print(f"Erro na conexão com o banco: {e}")
            return None

    def load_model(self):
        try:
            with open(self.MODEL_PATH, "rb") as f:
                data = pickle.load(f)
                self.embeddings_db = data["embeddings_db"]
                print(f"Modelo carregado com {len(self.embeddings_db)} pessoas")
                self.model_loaded = True
        except Exception as e:
            print(f"Erro ao carregar modelo: {e}")
            self.running = False

    def capture_frames(self):
        frame_size = self.width * self.height * 3
        while self.running:
            try:
                raw_frame = self.proc.stdout.read(frame_size)
                if len(raw_frame) != frame_size:
                    continue

                frame = np.frombuffer(raw_frame, np.uint8).reshape(
                    (self.height, self.width, 3)
                )
                if self.frame_queue.full():
                    self.frame_queue.get_nowait()  # Descartar frame antigo se a fila estiver cheia
                self.frame_queue.put(frame)
            except Exception as e:
                print(f"Erro na captura de frames: {e}")
                time.sleep(0.1)

    def get_user_info(self, user_id):
        if not self.db_connection:
            return {"nome": "Desconhecido", "sobrenome": ""}

        try:
            cursor = self.db_connection.cursor(dictionary=True)
            cursor.execute(
                "SELECT nome, sobrenome FROM cadastros WHERE id = %s", (user_id,)
            )
            result = cursor.fetchone()
            return result if result else {"nome": "Desconhecido", "sobrenome": ""}
        except Error as e:
            print(f"Erro ao buscar usuário: {e}")
            return {"nome": "Desconhecido", "sobrenome": ""}

    def safe_get_embedding(self, face_img):
        try:
            if isinstance(face_img, np.ndarray):
                if face_img.dtype != np.uint8:
                    face_img = face_img.astype(np.uint8)
                if len(face_img.shape) == 3 and face_img.shape[2] == 3:
                    face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)

            embedding_obj = DeepFace.represent(
                img_path=face_img,
                model_name="Facenet",
                enforce_detection=False,
                detector_backend="skip",
                normalization="base",
            )
            if embedding_obj and isinstance(embedding_obj, list):
                return np.array(embedding_obj[0]["embedding"]).flatten()
        except Exception as e:
            print(f"Erro na geração de embedding: {str(e)}")
        return None

    def calculate_similarity(self, emb1, emb2):
        try:
            emb1 = emb1 / (np.linalg.norm(emb1) + 1e-10)
            emb2 = emb2 / (np.linalg.norm(emb2) + 1e-10)
            return np.dot(emb1, emb2)
        except:
            return 0.0

    def process_frame(self, frame):
        try:
            if frame is None or frame.size == 0:
                return frame

            frame = frame.copy()

            faces = DeepFace.extract_faces(
                img_path=frame,  # CORREÇÃO AQUI
                detector_backend=self.DETECTOR,
                enforce_detection=False,
                align=True,
            )

            for face in faces:
                if not face or not face.get("facial_area"):
                    continue

                area = face["facial_area"]
                x, y, w, h = area["x"], area["y"], area["w"], area["h"]

                face_region = frame[y : y + h, x : x + w]

                live_emb = self.safe_get_embedding(face_region)
                if live_emb is None:
                    continue

                print("Embedding gerado (amostra):", live_emb[:5])

                best_match = None
                best_score = 0

                for user_id, user_data in self.embeddings_db.items():
                    stored_emb = np.array(user_data["embedding"]).flatten()
                    print(
                        f"Embedding armazenado para {user_id} (amostra):",
                        stored_emb[:5],
                    )

                    score = self.calculate_similarity(live_emb, stored_emb)
                    if score > best_score:
                        best_score = score
                        best_match = user_id

                print(f"Similaridade máxima encontrada: {best_score:.4f}")

                test_threshold = 0.3  # Ajuste para teste

                if best_score > test_threshold:
                    user_info = self.get_user_info(best_match)
                    label = f"{user_info['nome']} {user_info['sobrenome']}"
                    color = (0, 255, 0)
                else:
                    label = "Desconhecido"
                    color = (0, 0, 255)

                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(
                    frame,
                    f"{label} ({best_score:.2f})",
                    (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    color,
                    2,
                )

            return frame

        except Exception as e:
            print(f"Erro no processamento: {str(e)}")
            return frame

        except Exception as e:
            print(f"Erro no processamento: {str(e)}")
            return frame

        except Exception as e:
            print(f"Erro no processamento: {str(e)}")
            return frame

        except Exception as e:
            print(f"Erro no processamento: {str(e)}")
            return frame

    def run(self):
        print("\nSistema Ativo - Pressione 'q' para sair")

        # Configuração da janela para melhor performance
        cv2.namedWindow("Reconhecimento Facial", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Reconhecimento Facial", self.width, self.height)

        last_time = time.time()
        frame_count = 0

        while self.running:
            try:
                if not self.frame_queue.empty():
                    frame = self.frame_queue.get()

                    # Processar apenas 1 em cada 2 frames para melhorar performance
                    if frame_count % 2 == 0:
                        frame = self.process_frame(frame)

                    cv2.imshow("Reconhecimento Facial", frame)
                    frame_count += 1

                    # Calcular e mostrar FPS
                    current_time = time.time()
                    if current_time - last_time >= 1.0:
                        fps = frame_count / (current_time - last_time)
                        print(f"FPS: {fps:.2f}")
                        frame_count = 0
                        last_time = current_time

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            except Exception as e:
                print(f"Erro na exibição: {e}")
                time.sleep(0.1)

        self.cleanup()

    def cleanup(self):
        self.running = False
        if self.db_connection:
            self.db_connection.close()
        if self.proc:
            self.proc.terminate()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    fr = FaceRecognizer()
    fr.run()
