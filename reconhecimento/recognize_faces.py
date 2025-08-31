import cv2
import time
import logging
import os
import sys
import numpy as np
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
        """Inicialização completa do sistema com fallback para webcam"""
        try:
            # Primeiro tenta webcam (mais confiável)
            logging.info("Tentando conectar com webcam primeiro...")
            if self.initialize_webcam():
                logging.info("Webcam inicializada como fallback")
                return True

            # Se webcam falhar, tenta RTSP
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
        """Tenta inicializar webcam"""
        try:
            self.camera_manager = CameraManager(
                None, self.width, self.height, self.target_fps  # Sem URL RTSP
            )
            return self.camera_manager.initialize_camera()
        except Exception as e:
            logging.error(f"Erro ao inicializar webcam: {str(e)}")
            return False

    def initialize_rtsp(self):
        """Tenta inicializar RTSP"""
        try:
            self.camera_manager = CameraManager(
                self.rtsp_url, self.width, self.height, self.target_fps
            )
            return self.camera_manager.initialize_camera()
        except Exception as e:
            logging.error(f"Erro ao inicializar RTSP: {str(e)}")
            return False

    def run(self):
        """Loop principal de execução otimizado"""
        print("=" * 60)
        print("👁️  INICIANDO SISTEMA DE RECONHECIMENTO FACIAL")
        print("=" * 60)

        if not self.initialize_system():
            print("❌ Falha ao inicializar o sistema de câmeras")
            return

        camera_info = self.camera_manager.get_camera_info()
        print(f"📷 Câmera: {camera_info}")
        print("🎮 Controles:")
        print("   - Pressione 'q' para sair")
        print("   - Pressione 'r' para reconectar RTSP")
        print("   - Pressione 'f' para modo tela cheia/Janela")
        print("=" * 60)

        # Cria a janela apenas uma vez
        if not self.window_created:
            cv2.namedWindow(
                "Reconhecimento Facial", cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO
            )
            cv2.resizeWindow("Reconhecimento Facial", self.width, self.height)
            self.window_created = True

        last_time = time.time()
        frames_processed = 0
        self.running = True
        fullscreen = False  # Controle de tela cheia

        while self.running:
            try:
                # Obtém frame da câmera de forma não bloqueante
                frame = self.camera_manager.get_frame()

                if frame is not None:
                    # Processa o frame
                    processed_frame = self.face_processor.process_frame(frame)

                    if processed_frame is not None:
                        # Exibe o frame processado
                        cv2.imshow("Reconhecimento Facial", processed_frame)
                        frames_processed += 1
                else:
                    # Se não há frame, mostra mensagem de espera
                    waiting_frame = np.zeros(
                        (self.height, self.width, 3), dtype=np.uint8
                    )
                    cv2.putText(
                        waiting_frame,
                        "Aguardando frames da camera...",
                        (50, self.height // 2),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (255, 255, 255),
                        2,
                    )
                    cv2.imshow("Reconhecimento Facial", waiting_frame)

                # VERIFICAÇÃO MELHORADA do fechamento da janela
                try:
                    # Método mais confiável para verificar se a janela foi fechada
                    window_visible = cv2.getWindowProperty(
                        "Reconhecimento Facial", cv2.WND_PROP_VISIBLE
                    )
                    if window_visible < 1:
                        print("\n🖱️  Janela fechada pelo usuário (botão X)")
                        break
                except:
                    # Se ocorrer erro na verificação, assume que a janela foi fechada
                    print("\n🖱️  Janela fechada pelo usuário")
                    break

                # Cálculo de FPS a cada segundo
                current_time = time.time()
                if current_time - last_time >= 1.0:
                    fps = frames_processed / (current_time - last_time)
                    logging.info(f"FPS: {fps:.2f} - {camera_info}")
                    frames_processed = 0
                    last_time = current_time

                # Controles de teclado com waitKey mais curto
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    print("\n⌨️  Tecla 'q' pressionada")
                    break
                elif key == ord("r"):  # Tecla 'r' para tentar reconectar RTSP
                    print("🔄 Tentando reconectar câmera RTSP...")
                    logging.info("Tentando reconectar câmera RTSP...")
                    self.reconnect_camera()
                    # Atualiza a informação da câmera após reconexão
                    camera_info = self.camera_manager.get_camera_info()
                elif key == ord("f"):  # Tecla 'f' para toggle tela cheia
                    fullscreen = not fullscreen
                    if fullscreen:
                        cv2.setWindowProperty(
                            "Reconhecimento Facial",
                            cv2.WND_PROP_FULLSCREEN,
                            cv2.WINDOW_FULLSCREEN,
                        )
                        print("📺 Modo tela cheia ativado")
                    else:
                        cv2.setWindowProperty(
                            "Reconhecimento Facial",
                            cv2.WND_PROP_FULLSCREEN,
                            cv2.WINDOW_NORMAL,
                        )
                        cv2.resizeWindow(
                            "Reconhecimento Facial", self.width, self.height
                        )
                        print("📺 Modo janela ativado")

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
                print("✅ Reconexão bem-sucedida!")
                logging.info("Reconexão bem-sucedida!")
            else:
                print("⚠️  Falha na reconexão, usando webcam")
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

        print("=" * 60)
        print("🛑 SISTEMA DE RECONHECIMENTO ENCERRADO")
        print("✅ Recursos liberados com segurança")
        print("=" * 60)
        logging.info("Sistema finalizado com segurança")


if __name__ == "__main__":
    try:
        recognizer = FaceRecognizer()
        recognizer.run()
    except KeyboardInterrupt:
        print("\n\n🛑 Interrompido pelo usuário (Ctrl+C)")
        print("✅ Sistema encerrado com segurança")
    except Exception as e:
        print(f"\n❌ ERRO FATAL: {str(e)}")
        logging.error(f"ERRO FATAL: {str(e)}")
    finally:
        try:
            cv2.destroyAllWindows()
        except:
            pass
