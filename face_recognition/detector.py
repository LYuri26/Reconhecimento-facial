import cv2
import face_recognition
import pickle
import mysql.connector
import numpy as np
from datetime import datetime
import os
import time
import logging
import pygame
import subprocess
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import sys
import dlib

# Adiciona o diretório raiz ao PATH
BASE_DIR = Path(__file__).parent.parent
sys.path.append(str(BASE_DIR))

try:
    from config import DB_CONFIG, RECOGNITION_SETTINGS, SECURITY_SETTINGS
except ImportError as e:
    logging.error(f"Erro ao importar configurações: {e}")
    sys.exit(1)

# Configuração de logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(BASE_DIR / "logs" / "face_detector.log"),
        logging.StreamHandler(),
    ],
)


class FaceDetector:
    def __init__(self):
        """Inicializa o detector facial"""
        self.model_path = Path(RECOGNITION_SETTINGS["model_path"])
        self.recognition_threshold = RECOGNITION_SETTINGS["recognition_threshold"]
        self.min_face_size = RECOGNITION_SETTINGS["min_face_size"]
        self.preview_scale = RECOGNITION_SETTINGS.get("preview_scale", 0.5)

        # Configuração do ambiente para evitar problemas com Qt
        os.environ["QT_QPA_PLATFORM"] = "xcb"
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Desabilita GPU se necessário

        # Inicializa mixer de áudio
        pygame.mixer.init()

        # Carrega modelos dlib
        self.detector = dlib.get_frontal_face_detector()
        self.shape_predictor = dlib.shape_predictor(
            str(BASE_DIR / "shape_predictor_68_face_landmarks.dat")
        )
        self.face_recognition_model = dlib.face_recognition_model_v1(
            str(BASE_DIR / "dlib_face_recognition_resnet_model_v1.dat")
        )

        # Carrega modelo
        self.load_model()

        # Inicializa câmera
        self.video_capture = self.initialize_camera()

        # Variáveis de estado
        self.unknown_counter = 0
        self.recognition_history = []

        logging.info("FaceDetector inicializado com sucesso")

    def initialize_camera(self) -> cv2.VideoCapture:
        """Configura a captura de vídeo"""
        try:
            cap = cv2.VideoCapture(0)

            if not cap.isOpened():
                raise RuntimeError("Não foi possível abrir a câmera")

            # Configurações otimizadas
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            cap.set(cv2.CAP_PROP_FPS, 30)

            return cap

        except Exception as e:
            logging.error(f"Erro ao inicializar câmera: {e}")
            raise

    def load_model(self) -> None:
        """Carrega o modelo de reconhecimento facial"""
        try:
            if not self.model_path.exists():
                raise FileNotFoundError(
                    f"Arquivo de modelo não encontrado: {self.model_path}"
                )

            with open(self.model_path, "rb") as f:
                model_data = pickle.load(f)
                self.known_face_encodings = model_data["encodings"]
                self.known_face_names = model_data["names"]
                self.known_face_ids = model_data.get("ids", [])

            logging.info(
                f"Modelo carregado com {len(self.known_face_encodings)} encodings"
            )

        except Exception as e:
            logging.error(f"Erro ao carregar modelo: {e}")
            raise

    def get_db_connection(self) -> mysql.connector.MySQLConnection:
        """Cria conexão com o banco de dados"""
        try:
            # Conexão direta sem pool
            conn = mysql.connector.connect(
                host=DB_CONFIG["host"],
                user=DB_CONFIG["user"],
                password=DB_CONFIG["password"],
                database=DB_CONFIG["database"],
                raise_on_warnings=DB_CONFIG["raise_on_warnings"],
            )
            return conn
        except mysql.connector.Error as err:
            logging.error(f"Erro de conexão com o banco: {err}")
            raise

    def log_recognition(self, person_id: int, name: str, confidence: float) -> None:
        """Registra um reconhecimento no banco de dados"""
        try:
            conn = self.get_db_connection()
            cursor = conn.cursor()

            query = """
                INSERT INTO reconhecimentos 
                (pessoa_id, nome_pessoa, confianca, data_reconhecimento)
                VALUES (%s, %s, %s, NOW())
            """
            cursor.execute(query, (person_id, name, float(confidence)))
            conn.commit()

            logging.debug(
                f"Reconhecimento registrado: {name} (ID: {person_id}, Confiança: {confidence:.2f})"
            )

        except mysql.connector.Error as err:
            logging.error(f"Erro ao registrar reconhecimento: {err}")
        finally:
            if "conn" in locals() and conn.is_connected():
                cursor.close()
                conn.close()

    def process_frame(
        self, frame: np.ndarray
    ) -> Tuple[List[Tuple], List[str], List[float]]:
        """Processa um frame e retorna faces, nomes e confianças"""
        # Redimensiona para processamento mais rápido
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = small_frame[:, :, ::-1]  # BGR para RGB

        # Detecta faces usando dlib diretamente
        faces = self.detector(rgb_small_frame, 1)

        face_locations = []
        for face in faces:
            # Converte retângulo dlib para formato (top, right, bottom, left)
            top = face.top()
            right = face.right()
            bottom = face.bottom()
            left = face.left()
            face_locations.append((top, right, bottom, left))

        # Filtra faces muito pequenas
        face_locations = [
            loc
            for loc in face_locations
            if (loc[2] - loc[0]) >= self.min_face_size
            and (loc[1] - loc[3]) >= self.min_face_size
        ]

        if not face_locations:
            return [], [], []

        # Processa cada face detectada
        face_encodings = []
        for top, right, bottom, left in face_locations:
            # Converte para retângulo dlib
            rect = dlib.rectangle(left, top, right, bottom)
            # Detecta landmarks
            shape = self.shape_predictor(rgb_small_frame, rect)
            # Calcula o encoding facial
            face_descriptor = self.face_recognition_model.compute_face_descriptor(
                rgb_small_frame, shape
            )
            face_encodings.append(np.array(face_descriptor))

        names = []
        confidences = []

        for face_encoding in face_encodings:
            # Compara com rostos conhecidos
            matches = face_recognition.compare_faces(
                self.known_face_encodings,
                face_encoding,
                tolerance=1 - self.recognition_threshold,
            )

            name = "Desconhecido"
            confidence = 0.0
            person_id = None

            if True in matches:
                face_distances = face_recognition.face_distance(
                    self.known_face_encodings, face_encoding
                )
                best_match_idx = np.argmin(face_distances)
                confidence = 1 - face_distances[best_match_idx]

                if confidence >= self.recognition_threshold:
                    name = self.known_face_names[best_match_idx]
                    person_id = (
                        self.known_face_ids[best_match_idx]
                        if self.known_face_ids
                        else None
                    )

                    # Registra no histórico
                    self.recognition_history.append(
                        {
                            "id": person_id,
                            "name": name,
                            "confidence": confidence,
                            "timestamp": datetime.now(),
                        }
                    )

                    # Limita histórico
                    if len(self.recognition_history) > 100:
                        self.recognition_history.pop(0)

                    # Registra no banco (evita muitos registros repetidos)
                    if (
                        person_id
                        and len(
                            [
                                r
                                for r in self.recognition_history[-5:]
                                if r.get("id") == person_id
                            ]
                        )
                        == 1
                    ):
                        self.log_recognition(person_id, name, confidence)

            names.append(name)
            confidences.append(confidence)

        return face_locations, names, confidences

    def handle_security_actions(self, has_unknown: bool) -> None:
        """Executa ações de segurança conforme configuração"""
        if has_unknown:
            self.unknown_counter += 1

            if self.unknown_counter >= SECURITY_SETTINGS["unknown_frames_threshold"]:
                logging.warning("Pessoa não autorizada detectada - Ações de segurança")

                if SECURITY_SETTINGS.get("play_alarm", False):
                    self.play_alarm()

                if SECURITY_SETTINGS.get("lock_screen", False):
                    self.lock_screen()

                self.unknown_counter = 0
        else:
            self.unknown_counter = max(0, self.unknown_counter - 1)

    def play_alarm(self) -> None:
        """Toca o alarme sonoro configurado"""
        try:
            sound_file = Path(SECURITY_SETTINGS.get("alarm_sound", ""))
            if sound_file.exists():
                pygame.mixer.music.load(str(sound_file))
                pygame.mixer.music.play()
                logging.info("Alarme sonoro acionado")
            else:
                logging.warning(f"Arquivo de alarme não encontrado: {sound_file}")
        except Exception as e:
            logging.error(f"Erro ao tocar alarme: {e}")

    def lock_screen(self) -> None:
        """Bloqueia a tela do computador"""
        try:
            if os.name == "nt":  # Windows
                subprocess.run(
                    ["rundll32.exe", "user32.dll,LockWorkStation"], check=True
                )
            else:  # Linux
                # Tenta vários métodos de bloqueio
                for cmd in [
                    ["gnome-screensaver-command", "--lock"],
                    ["xdg-screensaver", "lock"],
                    ["loginctl", "lock-session"],
                ]:
                    try:
                        subprocess.run(cmd, check=True)
                        break
                    except (subprocess.CalledProcessError, FileNotFoundError):
                        continue
                else:
                    raise RuntimeError("Nenhum método de bloqueio disponível")

            logging.info("Tela bloqueada com sucesso")
        except Exception as e:
            logging.error(f"Erro ao bloquear tela: {e}")

    def draw_annotations(
        self,
        frame: np.ndarray,
        face_locations: List[Tuple],
        names: List[str],
        confidences: List[float],
    ) -> None:
        """Desenha anotações no frame"""
        for (top, right, bottom, left), name, confidence in zip(
            face_locations, names, confidences
        ):
            # Escala de volta para o tamanho original
            top = int(top * 4 * self.preview_scale)
            right = int(right * 4 * self.preview_scale)
            bottom = int(bottom * 4 * self.preview_scale)
            left = int(left * 4 * self.preview_scale)

            # Define cores
            color = (0, 255, 0) if name != "Desconhecido" else (0, 0, 255)

            # Desenha retângulo
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)

            # Desenha label
            label = f"{name} ({confidence:.2f})" if name != "Desconhecido" else name
            cv2.rectangle(
                frame, (left, bottom - 35), (right, bottom), color, cv2.FILLED
            )
            cv2.putText(
                frame,
                label,
                (left + 6, bottom - 6),
                cv2.FONT_HERSHEY_DUPLEX,
                0.8,
                (255, 255, 255),
                1,
            )

    def run(self, timeout: int = 0) -> None:
        """Executa o loop principal de reconhecimento"""
        start_time = time.time()
        logging.info("Iniciando reconhecimento facial. Pressione 'q' para sair...")

        try:
            while True:
                # Verifica timeout
                if timeout > 0 and (time.time() - start_time) > timeout:
                    logging.info(f"Timeout de {timeout} segundos atingido")
                    break

                # Captura frame
                ret, frame = self.video_capture.read()
                if not ret:
                    logging.error("Erro ao capturar frame da câmera")
                    break

                # Processa frame
                face_locations, names, confidences = self.process_frame(frame)

                # Verifica desconhecidos
                has_unknown = "Desconhecido" in names
                self.handle_security_actions(has_unknown)

                # Prepara preview
                preview_frame = cv2.resize(
                    frame, (0, 0), fx=self.preview_scale, fy=self.preview_scale
                )
                self.draw_annotations(preview_frame, face_locations, names, confidences)

                # Exibe frame
                cv2.imshow("Reconhecimento Facial", preview_frame)

                # Verifica tecla de saída
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    logging.info("Reconhecimento interrompido pelo usuário")
                    break

        except KeyboardInterrupt:
            logging.info("Reconhecimento interrompido pelo usuário")
        except Exception as e:
            logging.error(f"Erro durante reconhecimento: {e}", exc_info=True)
        finally:
            self.cleanup()

    def cleanup(self) -> None:
        """Libera recursos"""
        if hasattr(self, "video_capture") and self.video_capture.isOpened():
            self.video_capture.release()
        cv2.destroyAllWindows()
        logging.info("Sistema de reconhecimento finalizado")


if __name__ == "__main__":
    try:
        detector = FaceDetector()
        detector.run(timeout=0)  # Execução contínua
    except Exception as e:
        logging.error(f"Erro fatal: {e}", exc_info=True)
        sys.exit(1)
