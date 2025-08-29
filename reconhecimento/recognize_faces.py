import cv2
import time
import logging
import os
from camera_manager import CameraManager
from face_processor import FaceProcessor

# Configurações para evitar travamentos
os.environ["QT_QPA_PLATFORM"] = "xcb"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["OPENCV_VIDEOIO_DEBUG"] = "0"

# Configuração de logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("face_recognition.log"), logging.StreamHandler()],
)


class FaceRecognizer:
    def __init__(self):
        # Configurações do sistema
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

    def initialize_system(self):
        """Inicialização completa do sistema"""
        try:
            # Inicializa câmera
            if not self.camera_manager.initialize_camera():
                return False

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

        logging.info("\nSistema Ativo - Pressione 'q' para sair")
        cv2.namedWindow("Reconhecimento Facial", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Reconhecimento Facial", self.width, self.height)

        last_time = time.time()
        frames_processed = 0
        self.running = True

        while self.running:
            try:
                # Obtém frame da câmera
                frame = self.camera_manager.get_frame()
                if frame is not None:
                    # Processa o frame
                    processed_frame = self.face_processor.process_frame(frame)

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

    def cleanup(self):
        """Limpeza de recursos"""
        self.running = False

        # Limpa câmera
        self.camera_manager.cleanup()

        # Limpa processador facial
        self.face_processor.cleanup()

        # Fecha janelas OpenCV
        try:
            cv2.destroyAllWindows()
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
