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
import logging
import os
from collections import deque

# Configurações para evitar travamentos
os.environ["QT_QPA_PLATFORM"] = "xcb"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["OPENCV_VIDEOIO_DEBUG"] = "0"

# Modifique a configuração de logging no início do arquivo
logging.basicConfig(
    level=logging.DEBUG,  # Alterado de INFO para DEBUG
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("face_recognition.log"), logging.StreamHandler()],
)


class FaceRecognizer:
    def __init__(self):
        # Configurações do sistema
        self.MODEL_PATH = "model/deepface_model.pkl"
        self.THRESHOLD = 0.40
        self.EMBEDDING_MODEL = "Facenet512"
        self.DETECTOR = "opencv"  # Mais rápido e estável

        # Configurações de câmera
        self.rtsp_url = (
            "rtsp://admin:Evento0128@192.168.1.101:559/Streaming/Channels/101"
        )
        self.width = 640
        self.height = 480
        self.target_fps = 10
        self.frame_skip = 2  # Pular frames para melhorar performance

        # Controles do sistema
        self.frame_queue = queue.Queue(maxsize=3)
        self.running = True
        self.model_loaded = False
        self.camera_initialized = False

        # Variáveis de desempenho
        self.frame_count = 0
        self.last_fps_time = time.time()

    def initialize_system(self):
        """Inicialização completa do sistema"""
        try:
            self.initialize_camera()
            self.load_model()
            self.initialize_database()
            logging.info("Sistema inicializado com sucesso")
            return True
        except Exception as e:
            logging.error(f"Falha na inicialização: {str(e)}")
            self.cleanup()
            return False

    def initialize_camera(self):
        """Inicialização da câmera RTSP"""
        try:
            # Comando FFmpeg para captura RTSP
            ffmpeg_cmd = [
                "ffmpeg",
                "-rtsp_transport",
                "tcp",
                "-i",
                self.rtsp_url,
                "-loglevel",
                "error",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "bgr24",
                "-s",
                f"{self.width}x{self.height}",
                "-r",
                str(self.target_fps),
                "-",
            ]

            # Inicia o processo FFmpeg
            self.proc = subprocess.Popen(
                ffmpeg_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=10**8,
            )

            # Testa a conexão
            if not self.test_camera_connection(timeout=10):
                raise RuntimeError("Não foi possível conectar à câmera")

            self.camera_initialized = True
            threading.Thread(target=self.capture_frames, daemon=True).start()
            logging.info("Câmera inicializada com sucesso")

        except Exception as e:
            logging.error(f"Erro na inicialização da câmera: {str(e)}")
            self.camera_initialized = False
            raise

    def test_camera_connection(self, timeout=5):
        """Testa a conexão com a câmera"""
        start_time = time.time()
        while time.time() - start_time < timeout:
            frame = self.get_test_frame()
            if frame is not None:
                return True
            time.sleep(0.1)
        return False

    def get_test_frame(self):
        """Obtém um frame de teste"""
        try:
            frame_size = self.width * self.height * 3
            raw_frame = self.proc.stdout.read(frame_size)
            if raw_frame:
                frame = np.frombuffer(raw_frame, np.uint8).reshape(
                    (self.height, self.width, 3)
                )
                if frame.size > 0 and frame.shape == (self.height, self.width, 3):
                    return frame
        except Exception as e:
            logging.debug(f"Erro ao obter frame de teste: {str(e)}")
        return None

    def load_model(self):
        """Carrega o modelo de reconhecimento facial"""
        try:
            if not os.path.exists(self.MODEL_PATH):
                raise FileNotFoundError(
                    f"Arquivo do modelo não encontrado: {self.MODEL_PATH}"
                )

            with open(self.MODEL_PATH, "rb") as f:
                data = pickle.load(f)
                self.embeddings_db = data["embeddings_db"]

                # Normaliza os embeddings
                for user_id in self.embeddings_db:
                    if isinstance(self.embeddings_db[user_id]["embedding"], list):
                        self.embeddings_db[user_id]["embedding"] = np.array(
                            self.embeddings_db[user_id]["embedding"], dtype=np.float32
                        )
                    # Normalização L2
                    norm = np.linalg.norm(self.embeddings_db[user_id]["embedding"])
                    if norm > 0:
                        self.embeddings_db[user_id]["embedding"] /= norm

                logging.info(f"Modelo carregado com {len(self.embeddings_db)} pessoas")
                self.model_loaded = True

        except Exception as e:
            logging.error(f"Erro ao carregar o modelo: {str(e)}")
            self.model_loaded = False
            raise

    def initialize_database(self):
        """Conecta ao banco de dados MySQL"""
        try:
            self.db_connection = mysql.connector.connect(
                host="localhost",
                user="root",
                password="",
                database="reconhecimento_facial",
            )
            logging.info("Conexão com o banco estabelecida")
        except Error as e:
            logging.warning(f"Erro na conexão com o banco: {str(e)}")
            self.db_connection = None

    def capture_frames(self):
        """Captura frames da câmera continuamente"""
        frame_size = self.width * self.height * 3
        frame_counter = 0

        while self.running and self.camera_initialized:
            try:
                raw_frame = self.proc.stdout.read(frame_size)
                if not raw_frame or len(raw_frame) != frame_size:
                    continue

                # Controle de frame skip
                frame_counter += 1
                if frame_counter % (self.frame_skip + 1) != 0:
                    continue

                frame = np.frombuffer(raw_frame, np.uint8).reshape(
                    (self.height, self.width, 3)
                )

                if not self.frame_queue.full():
                    self.frame_queue.put(frame)

            except Exception as e:
                logging.error(f"Erro na captura de frames: {str(e)}")
                time.sleep(0.1)

    def cleanup(self):
        """Limpeza de recursos"""
        self.running = False

        # Finaliza o processo FFmpeg
        if hasattr(self, "proc") and self.proc:
            try:
                self.proc.terminate()
                self.proc.wait(timeout=2)
            except Exception as e:
                logging.error(f"Erro ao finalizar FFmpeg: {str(e)}")
                try:
                    self.proc.kill()
                except:
                    pass

        # Fecha a conexão com o banco
        if hasattr(self, "db_connection") and self.db_connection:
            try:
                self.db_connection.close()
            except Exception as e:
                logging.error(f"Erro ao fechar conexão com o banco: {str(e)}")

        # Fecha janelas OpenCV
        try:
            cv2.destroyAllWindows()
        except Exception as e:
            logging.error(f"Erro ao fechar janelas: {str(e)}")

        logging.info("Sistema finalizado com segurança")

    def calculate_similarity(self, emb1, emb2):
        """Calcula similaridade entre embeddings corretamente"""
        try:
            if emb1 is None or emb2 is None:
                return 0.0

            # Garante que os embeddings são arrays numpy
            emb1 = np.array(emb1, dtype=np.float32)
            emb2 = np.array(emb2, dtype=np.float32)

            # Normaliza os embeddings (novamente, para garantir)
            emb1 = emb1 / np.linalg.norm(emb1)
            emb2 = emb2 / np.linalg.norm(emb2)

            # Calcula a similaridade cosseno (deve estar entre -1 e 1)
            similarity = np.dot(emb1, emb2)

            # Ajusta para ficar entre 0 e 1 (se necessário)
            similarity = (similarity + 1) / 2

            return similarity
        except Exception as e:
            logging.error(f"Erro no cálculo de similaridade: {str(e)}")
            return 0.0

    def get_user_info(self, user_id):
        """Obtém informações do usuário do banco de dados"""
        if not self.db_connection:
            return {"nome": "Desconhecido", "sobrenome": ""}

        try:
            cursor = self.db_connection.cursor(dictionary=True)
            cursor.execute(
                "SELECT nome, sobrenome FROM cadastros WHERE id = %s", (user_id,)
            )
            result = cursor.fetchone()
            return result or {"nome": "Desconhecido", "sobrenome": ""}
        except Error as e:
            logging.error(f"Erro ao buscar usuário: {str(e)}")
            return {"nome": "Desconhecido", "sobrenome": ""}
        finally:
            if "cursor" in locals():
                cursor.close()

    def safe_get_embedding(self, face_img):
        """Gera embedding facial com mais verificações"""
        try:
            if not isinstance(face_img, np.ndarray) or face_img.size == 0:
                return None
            if face_img.shape[0] < 50 or face_img.shape[1] < 50:
                return None

            # Converte para RGB
            rgb_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)

            # Gera o embedding
            embedding_obj = DeepFace.represent(
                img_path=rgb_img,
                model_name=self.EMBEDDING_MODEL,
                enforce_detection=False,
                detector_backend="skip",
                normalization="base",
            )

            if not embedding_obj or not isinstance(embedding_obj, list):
                return None

            embedding = np.array(embedding_obj[0]["embedding"], dtype=np.float32)

            # Verifica se o embedding é válido
            if np.all(embedding == 0):
                logging.warning("Embedding gerado é nulo/zero")
                return None

            # Normalização rigorosa
            norm = np.linalg.norm(embedding)
            if norm == 0:
                return None

            embedding = embedding / norm

            # Log para debug
            logging.debug(
                f"Embedding gerado - Norma: {np.linalg.norm(embedding):.4f}, "
                f"Valores: min={np.min(embedding):.4f}, max={np.max(embedding):.4f}"
            )

            return embedding

        except Exception as e:
            logging.error(f"Erro na geração de embedding: {str(e)}")
            return None

    def check_embeddings_db(self):
        """Verifica a qualidade dos embeddings armazenados"""
        if not self.embeddings_db:
            logging.error("Nenhum embedding encontrado no banco de dados")
            return False

        for user_id, user_data in self.embeddings_db.items():
            embedding = user_data["embedding"]
            norm = np.linalg.norm(embedding)
            logging.info(f"Usuário {user_id} - Norma do embedding: {norm:.4f}")

            if norm < 0.9 or norm > 1.1:  # Deve ser próximo de 1 após normalização
                logging.warning(
                    f"Embedding do usuário {user_id} não está normalizado corretamente!"
                )

        return True

    # Modifique o método process_frame na classe FaceRecognizer
    def process_frame(self, frame):
        """Processa um frame para reconhecimento facial"""
        try:
            if frame is None or frame.size == 0:
                return frame

            display_frame = frame.copy()

            if not self.model_loaded:
                return display_frame

            try:
                faces = DeepFace.extract_faces(
                    img_path=frame,
                    detector_backend=self.DETECTOR,
                    enforce_detection=False,
                    align=True,
                )
            except Exception as e:
                logging.error(f"Erro na detecção de faces: {str(e)}")
                return display_frame

            if not isinstance(faces, list):
                return display_frame

            for face in faces:
                if not face or not face.get("facial_area"):
                    continue

                area = face["facial_area"]
                x, y, w, h = area["x"], area["y"], area["w"], area["h"]
                face_region = frame[y : y + h, x : x + w]

                live_emb = self.safe_get_embedding(face_region)
                if live_emb is None:
                    continue

                # Adicionando log de debug para mostrar o processamento
                logging.debug(f"Processando face em ({x},{y}) - {w}x{h}")

                best_match = None
                best_score = 0
                all_scores = {}  # Para armazenar todos os scores

                for user_id, user_data in self.embeddings_db.items():
                    score = self.calculate_similarity(live_emb, user_data["embedding"])
                    all_scores[user_id] = score

                    if score > best_score and score > self.THRESHOLD:
                        best_score = score
                        best_match = user_id

                # Log detalhado de todos os scores calculados
                logging.info("Scores de similaridade calculados:")
                for user_id, score in all_scores.items():
                    user_info = self.get_user_info(user_id)
                    logging.info(f"  {user_info['nome']}: {score:.4f}")

                label = "Desconhecido"
                color = (0, 0, 255)  # Vermelho

                if best_match:
                    user_info = self.get_user_info(best_match)
                    label = f"{user_info['nome']} ({best_score:.2f})"
                    color = (0, 255, 0)  # Verde
                    logging.info(
                        f"Melhor correspondência: {user_info['nome']} com score {best_score:.4f}"
                    )

                cv2.rectangle(display_frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(
                    display_frame,
                    label,
                    (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    color,
                    1,
                )

            return display_frame

        except Exception as e:
            logging.error(f"Erro no processamento do frame: {str(e)}")
            return frame

    def run(self):
        """Loop principal de execução"""
        if not self.initialize_system():
            return

        logging.info("\nSistema Ativo - Pressione 'q' para sair")
        cv2.namedWindow("Reconhecimento Facial", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Reconhecimento Facial", self.width, self.height)

        last_time = time.time()
        frames_processed = 0

        while self.running:
            try:
                if not self.frame_queue.empty():
                    frame = self.frame_queue.get()
                    processed_frame = self.process_frame(frame)

                    if processed_frame is not None:
                        cv2.imshow("Reconhecimento Facial", processed_frame)
                        frames_processed += 1

                # Cálculo de FPS
                current_time = time.time()
                if current_time - last_time >= 1.0:
                    fps = frames_processed / (current_time - last_time)
                    logging.info(f"FPS: {fps:.2f}")
                    frames_processed = 0
                    last_time = current_time

                # Controles de teclado
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                elif key == ord("+"):
                    self.frame_skip = max(0, self.frame_skip - 1)
                    logging.info(f"Frame skip reduzido para {self.frame_skip}")
                elif key == ord("-"):
                    self.frame_skip += 1
                    logging.info(f"Frame skip aumentado para {self.frame_skip}")

            except Exception as e:
                logging.error(f"Erro no loop principal: {str(e)}")
                time.sleep(0.1)

        self.cleanup()


if __name__ == "__main__":
    try:
        recognizer = FaceRecognizer()
        recognizer.run()
    except Exception as e:
        logging.error(f"ERRO FATAL: {str(e)}")
    finally:
        cv2.destroyAllWindows()
