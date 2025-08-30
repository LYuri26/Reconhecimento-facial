import cv2
import time
import logging
import os
import sys
import numpy as np  # <-- adicionei aqui
from pathlib import Path

# Adiciona o diretório atual ao path para importações relativas
sys.path.insert(0, str(Path(__file__).parent))

from camera_manager import CameraManager
from face_processor import FaceProcessor

# Configurações para evitar travamentos
os.environ["QT_QPA_PLATFORM"] = "xcb"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["OPENCV_VIDEOIO_DEBUG"] = "0"
os.environ["OPENCV_LOG_LEVEL"] = "ERROR"

# Configuração de logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("face_recognition.log"), logging.StreamHandler()],
)

# Reduzir logs do OpenCV
try:
    cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_ERROR)
except AttributeError:
    if hasattr(cv2, "setLogLevel") and hasattr(cv2, "LOG_LEVEL_ERROR"):
        cv2.setLogLevel(cv2.LOG_LEVEL_ERROR)


class FaceRecognizer:
    def __init__(self):
        # Configurações do sistema - RTSP e fallback para webcam
        self.rtsp_url = (
            "rtsp://admin:Evento0128@192.168.1.101:559/Streaming/Channels/101"
        )
        self.width = 640
        self.height = 480
        self.target_fps = 10

        # Inicializa os módulos
        self.camera_manager = CameraManager(
            self.rtsp_url, self.width, self.height, self.target_fps
        )
        self.face_processor = FaceProcessor()

        # Controles do sistema
        self.running = False
        self.window_created = False

    def initialize_system(self):
        """Inicialização completa do sistema"""
        try:
            # Inicializa câmera (tenta RTSP, depois webcam)
            if not self.camera_manager.initialize_camera():
                logging.error("Não foi possível inicializar nenhuma câmera")
                return False

            # Log do tipo de câmera em uso
            camera_info = self.camera_manager.get_camera_info()
            logging.info(f"Câmera em uso: {camera_info}")

            # Carrega modelo
            if not self.face_processor.load_model():
                return False

            # Conecta ao banco
            if not self.face_processor.initialize_database():
                logging.warning("Banco de dados não disponível, usando modo offline")

            logging.info("Sistema inicializado com sucesso")
            return True

        except Exception as e:
            logging.error(f"Falha na inicialização: {str(e)}")
            self.cleanup()
            return False

    def run(self):
        """Loop principal de execução"""
        if not self.initialize_system():
            return

        camera_info = self.camera_manager.get_camera_info()
        logging.info(f"\nSistema Ativo - {camera_info}")
        logging.info("Pressione 'q' para sair")

        # Cria a janela apenas uma vez
        if not self.window_created:
            cv2.namedWindow("Reconhecimento Facial", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Reconhecimento Facial", self.width, self.height)
            self.window_created = True

        last_time = time.time()
        frames_processed = 0
        self.running = True

        # Frame padrão para exibir quando não há frames da câmera
        default_frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        cv2.putText(
            default_frame,
            "Aguardando frames da camera...",
            (50, self.height // 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )

        while self.running:
            try:
                # Obtém frame da câmera
                frame = self.camera_manager.get_frame()
                display_frame = default_frame.copy()

                if frame is not None:
                    # Processa o frame
                    processed_frame = self.face_processor.process_frame(frame)

                    if processed_frame is not None:
                        display_frame = processed_frame
                        # Adiciona informação da câmera no frame
                        camera_type = "RTSP" if "RTSP" in camera_info else "WEBCAM"
                        cv2.putText(
                            display_frame,
                            f"Camera: {camera_type}",
                            (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (255, 255, 255),
                            2,
                        )

                        frames_processed += 1

                # Exibe o frame
                cv2.imshow("Reconhecimento Facial", display_frame)

                # Cálculo de FPS
                current_time = time.time()
                if current_time - last_time >= 1.0:
                    fps = frames_processed / (current_time - last_time)
                    logging.info(f"FPS: {fps:.2f} - {camera_info}")
                    frames_processed = 0
                    last_time = current_time

                # Controles de teclado
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                elif key == ord("r"):  # Tecla 'r' para tentar reconectar RTSP
                    logging.info("Tentando reconectar câmera RTSP...")
                    self.reconnect_camera()
                elif key == ord("+"):
                    self.camera_manager.frame_skip = max(
                        0, self.camera_manager.frame_skip - 1
                    )
                    logging.info(
                        f"Frame skip reduzido para {self.camera_manager.frame_skip}"
                    )
                elif key == ord("-"):
                    self.camera_manager.frame_skip += 1
                    logging.info(
                        f"Frame skip aumentado para {self.camera_manager.frame_skip}"
                    )

            except Exception as e:
                logging.error(f"Erro no loop principal: {str(e)}")
                time.sleep(0.1)

        self.cleanup()

    def reconnect_camera(self):
        """Tenta reconectar a câmera RTSP"""
        try:
            self.cleanup()
            time.sleep(1)

            # Recria os objetos
            self.camera_manager = CameraManager(
                self.rtsp_url, self.width, self.height, self.target_fps
            )
            self.face_processor = FaceProcessor()

            if self.initialize_system():
                logging.info("Reconexão bem-sucedida!")
            else:
                logging.warning("Falha na reconexão, usando webcam")

        except Exception as e:
            logging.error(f"Erro na reconexão: {str(e)}")

    def cleanup(self):
        """Limpeza de recursos"""
        self.running = False

        # Limpa câmera
        self.camera_manager.cleanup()

        # Limpa processador facial
        self.face_processor.cleanup()

        # Fecha janelas OpenCV
        try:
            if self.window_created:
                cv2.destroyAllWindows()
                self.window_created = False
        except Exception as e:
            logging.error(f"Erro ao fechar janelas: {str(e)}")

        logging.info("Sistema finalizado com segurança")


if __name__ == "__main__":
    try:
        recognizer = FaceRecognizer()
        recognizer.run()
    except Exception as e:
        logging.error(f"ERRO FATAL: {str(e)}")
    finally:
        cv2.destroyAllWindows()
