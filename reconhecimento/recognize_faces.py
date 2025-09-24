# recognize_faces.py - ATUALIZADO
import cv2
import time
import logging
import os
import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from camera_manager import CameraManager
from face_processor import FaceProcessor

# Configurações otimizadas para catraca
os.environ["QT_QPA_PLATFORM"] = "xcb"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # Menos logs
os.environ["OPENCV_VIDEOIO_DEBUG"] = "0"
os.environ["OPENCV_LOG_LEVEL"] = "ERROR"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # CPU only para estabilidade

# Configuração de logging otimizada
logging.basicConfig(
    level=logging.WARNING,  # Menos logs
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
        # Configurações otimizadas para catraca
        self.rtsp_url = (
            "rtsp://admin:Evento0128@192.168.1.101:559/Streaming/Channels/101"
        )
        self.width = 640
        self.height = 480
        self.target_fps = 15  # Aumentado para resposta mais rápida

        # Inicializa os módulos com configurações otimizadas
        self.camera_manager = CameraManager(
            self.rtsp_url, self.width, self.height, self.target_fps
        )
        self.face_processor = FaceProcessor(threshold=0.65)  # Otimizado

        self.running = False
        self.window_created = False
        self.last_recognition_time = 0
        self.recognition_cooldown = 2  # segundos entre reconhecimentos

    def initialize_system(self):
        """Inicialização otimizada"""
        try:
            logging.info("Inicializando sistema de catraca...")

            # Tenta webcam primeiro (fallback rápido)
            if self.initialize_webcam():
                logging.info("Webcam inicializada como fallback")
                return True

            # Tenta RTSP se disponível
            if self.rtsp_url:
                logging.info("Tentando conectar com câmera RTSP...")
                if self.initialize_rtsp():
                    return True

            logging.error("Nenhuma câmera disponível")
            return False

        except Exception as e:
            logging.error(f"Falha na inicialização: {str(e)}")
            return False

    def initialize_webcam(self):
        """Inicialização otimizada da webcam"""
        try:
            self.camera_manager = CameraManager(
                None, self.width, self.height, self.target_fps
            )
            return self.camera_manager.initialize_camera()
        except Exception as e:
            logging.debug(f"Webcam não disponível: {str(e)}")
            return False

    def initialize_rtsp(self):
        """Inicialização otimizada do RTSP"""
        try:
            self.camera_manager = CameraManager(
                self.rtsp_url, self.width, self.height, self.target_fps
            )
            success = self.camera_manager.initialize_camera()
            if success:
                logging.info("✅ Câmera RTSP conectada")
            return success
        except Exception as e:
            logging.error(f"Erro ao inicializar RTSP: {str(e)}")
            return False

    def create_display_window(self):
        """Cria janela de exibição otimizada"""
        try:
            cv2.namedWindow("Sistema de Catraca", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Sistema de Catraca", 800, 600)
            self.window_created = True
            return True
        except Exception as e:
            logging.warning(f"Janela não criada: {str(e)}")
            return False

    def handle_keypress(self, key):
        """Manipulação otimizada de teclas"""
        if key == ord("q") or key == ord("Q"):
            logging.info("Solicitação de saída pelo usuário")
            return False
        elif key == ord("r") or key == ord("R"):
            logging.info("Recarregando modelo...")
            self.face_processor.load_model()
        return True

    def draw_status_overlay(self, frame, fps, status):
        """Overlay otimizado de status"""
        try:
            h, w = frame.shape[:2]

            # Fundo semi-transparente
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (w, 60), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

            # Status do sistema
            color = (0, 255, 0) if status == "OPERACIONAL" else (0, 0, 255)
            cv2.putText(
                frame,
                f"SISTEMA: {status}",
                (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2,
            )

            # FPS
            cv2.putText(
                frame,
                f"FPS: {fps:.1f}",
                (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                1,
            )

            # Instruções
            cv2.putText(
                frame,
                "Q - Sair | R - Recarregar Modelo",
                (w - 300, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (200, 200, 200),
                1,
            )

            return frame
        except Exception as e:
            logging.debug(f"Erro no overlay: {str(e)}")
            return frame

    def run(self):
        """Loop principal otimizado para catraca"""
        try:
            if not self.initialize_system():
                logging.error("Falha na inicialização do sistema")
                return

            self.create_display_window()
            self.running = True

            logging.info("Sistema de catraca iniciado")
            fps_counter = 0
            fps_time = time.time()
            last_frame_time = time.time()

            while self.running:
                try:
                    # Controle de FPS
                    current_time = time.time()
                    elapsed = current_time - last_frame_time
                    if elapsed < 1.0 / self.target_fps:
                        time.sleep(0.001)  # Pequena pausa para controle de FPS
                        continue

                    last_frame_time = current_time

                    # Captura frame
                    frame = self.camera_manager.get_frame()
                    if frame is None:
                        logging.warning("Frame vazio recebido")
                        time.sleep(0.1)
                        continue

                    # Verifica se a câmera está tampada
                    if self.face_processor.is_camera_covered(frame):
                        cv2.putText(
                            frame,
                            "CAMERA TAMPADA/ESCURA",
                            (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            (0, 0, 255),
                            2,
                        )
                    else:
                        # Processamento otimizado do frame
                        frame = self.face_processor.process_frame(frame)

                    # Cálculo de FPS
                    fps_counter += 1
                    if current_time - fps_time >= 1.0:
                        fps = fps_counter / (current_time - fps_time)
                        fps_counter = 0
                        fps_time = current_time
                    else:
                        fps = 1.0 / elapsed if elapsed > 0 else 0

                    # Status do sistema
                    status = (
                        "OPERACIONAL"
                        if self.face_processor.model_loaded
                        else "SEM TREINAMENTO"
                    )

                    # Overlay de status
                    frame = self.draw_status_overlay(frame, fps, status)

                    # Exibe frame
                    cv2.imshow("Sistema de Catraca", frame)

                    # Controle de teclas (otimizado)
                    key = cv2.waitKey(1) & 0xFF
                    if not self.handle_keypress(key):
                        break

                except Exception as e:
                    logging.error(f"Erro no loop principal: {str(e)}")
                    time.sleep(0.1)  # Previne loop infinito de erro

        except KeyboardInterrupt:
            logging.info("Interrupção pelo usuário")
        except Exception as e:
            logging.error(f"Erro crítico: {str(e)}")
        finally:
            self.cleanup()

    def cleanup(self):
        """Limpeza otimizada de recursos"""
        try:
            self.running = False
            if hasattr(self, "camera_manager"):
                self.camera_manager.cleanup()
            if hasattr(self, "face_processor"):
                self.face_processor.cleanup()
            if self.window_created:
                cv2.destroyAllWindows()
            logging.info("Sistema finalizado corretamente")
        except Exception as e:
            logging.error(f"Erro na limpeza: {str(e)}")


if __name__ == "__main__":
    try:
        print("=" * 60)
        print("🚀 SISTEMA DE CATRACA FACIAL - INICIANDO")
        print("=" * 60)
        print("Configurações otimizadas para:")
        print("  ✓ Baixo número de imagens (1-5 por pessoa)")
        print("  ✓ Velocidade de resposta")
        print("  ✓ Confiabilidade em ambiente de catraca")
        print("=" * 60)

        recognizer = FaceRecognizer()
        recognizer.run()

    except Exception as e:
        print(f"❌ ERRO INICIAL: {str(e)}")
        import traceback

        traceback.print_exc()
    finally:
        print("=" * 60)
        print("👋 SISTEMA FINALIZADO")
        print("=" * 60)
