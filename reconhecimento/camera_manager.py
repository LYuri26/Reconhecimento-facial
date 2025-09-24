import cv2
import numpy as np
import subprocess
import threading
import queue
import time
import logging
import os


class CameraManager:
    def __init__(
        self, rtsp_url=None, width=320, height=240, target_fps=20
    ):  # Reduzido e FPS aumentado
        self.rtsp_url = rtsp_url
        self.width = width  # Reduzido para maior velocidade
        self.height = height  # Reduzido para maior velocidade
        self.target_fps = target_fps  # Aumentado
        self.frame_skip = 3  # Aumentado para pular mais frames
        self.frame_queue = queue.Queue(maxsize=2)  # Fila menor
        self.running = False
        self.camera_initialized = False
        self.proc = None
        self.cap = None
        self.camera_type = None
        self.last_frame_time = 0
        self.frame_counter = 0

    def initialize_camera(self):
        """Inicialização otimizada para velocidade"""
        try:
            # Tenta webcam primeiro (mais rápido)
            logging.info("Tentando webcam primeiro (mais rápido)...")
            if self.initialize_webcam():
                self.camera_type = "webcam"
                logging.info("Webcam inicializada com sucesso")
                return True

            # Se webcam falhar, tenta RTSP
            if self.rtsp_url:
                logging.info("Tentando RTSP...")
                if self.initialize_rtsp():
                    self.camera_type = "rtsp"
                    logging.info("RTSP inicializado com sucesso")
                    return True

            logging.error("Nenhuma câmera disponível")
            return False

        except Exception as e:
            logging.error(f"Erro na inicialização: {str(e)}")
            return False

    def initialize_rtsp(self):
        """RTSP otimizado para velocidade"""
        try:
            # Comando FFmpeg otimizado para velocidade
            ffmpeg_cmd = [
                "ffmpeg",
                "-rtsp_transport",
                "tcp",
                "-timeout",
                "3000000",  # Timeout reduzido
                "-i",
                self.rtsp_url,
                "-loglevel",
                "quiet",  # Sem logs
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

            self.proc = subprocess.Popen(
                ffmpeg_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=10**6,  # Buffer reduzido
            )

            # Teste rápido
            time.sleep(1)
            if self.proc.poll() is not None:
                return False

            self.camera_initialized = True
            self.running = True
            threading.Thread(target=self.capture_rtsp_frames, daemon=True).start()

            return True

        except Exception as e:
            logging.error(f"Erro no RTSP: {str(e)}")
            return False

    def initialize_webcam(self):
        """Webcam otimizada para velocidade máxima"""
        try:
            # Tenta apenas o índice 0 (mais comum)
            self.cap = cv2.VideoCapture(0, cv2.CAP_ANY)
            if not self.cap.isOpened():
                return False

            # Configurações para velocidade máxima
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            self.cap.set(cv2.CAP_PROP_FPS, self.target_fps)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Buffer mínimo
            self.cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)  # Desliga autofoco
            self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)  # Exposição automática off

            # Testa um frame
            ret, frame = self.cap.read()
            if not ret:
                self.cap.release()
                return False

            self.camera_initialized = True
            self.running = True

            # Thread de captura mais leve
            self.capture_thread = threading.Thread(
                target=self.capture_webcam_frames, daemon=True
            )
            self.capture_thread.start()

            return True

        except Exception as e:
            logging.error(f"Erro na webcam: {str(e)}")
            if self.cap:
                self.cap.release()
                self.cap = None
            return False

    def test_rtsp_connection(self, timeout=5):
        """Testa a conexão com a câmera RTSP"""
        start_time = time.time()
        while time.time() - start_time < timeout:
            frame = self.get_test_rtsp_frame()
            if frame is not None:
                return True
            time.sleep(0.1)
        return False

    def get_test_rtsp_frame(self):
        """Obtém um frame de teste da RTSP"""
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
            logging.debug(f"Erro ao obter frame de teste RTSP: {str(e)}")
        return None

    def capture_rtsp_frames(self):
        """Captura RTSP otimizada"""
        frame_size = self.width * self.height * 3

        while self.running and self.camera_initialized:
            try:
                raw_frame = self.proc.stdout.read(frame_size)
                if not raw_frame:
                    time.sleep(0.01)
                    continue

                if len(raw_frame) == frame_size:
                    frame = np.frombuffer(raw_frame, np.uint8).reshape(
                        (self.height, self.width, 3)
                    )

                    self.frame_counter += 1
                    if self.frame_counter % (self.frame_skip + 1) == 0:
                        try:
                            if self.frame_queue.full():
                                self.frame_queue.get_nowait()
                            self.frame_queue.put(frame, timeout=0.01)
                        except:
                            pass

            except Exception as e:
                logging.debug(f"Erro RTSP: {str(e)}")
                time.sleep(0.01)

    def capture_webcam_frames(self):
        """Captura ultra-rápida da webcam"""
        consecutive_errors = 0

        while self.running and self.camera_initialized:
            try:
                ret, frame = self.cap.read()
                if not ret:
                    consecutive_errors += 1
                    if consecutive_errors > 5:
                        break
                    time.sleep(0.01)
                    continue

                consecutive_errors = 0
                self.frame_counter += 1

                # Pula frames para aumentar velocidade
                if self.frame_counter % (self.frame_skip + 1) != 0:
                    continue

                # Redimensiona se necessário (mais rápido)
                if frame.shape[1] != self.width or frame.shape[0] != self.height:
                    frame = cv2.resize(frame, (self.width, self.height))

                # Adiciona à fila (não-bloqueante)
                try:
                    if self.frame_queue.full():
                        self.frame_queue.get_nowait()
                    self.frame_queue.put(frame, timeout=0.01)
                except:
                    pass  # Ignora erros de fila para velocidade

            except Exception as e:
                logging.debug(f"Erro na captura: {str(e)}")
                time.sleep(0.01)

    def get_frame(self):
        """Obtém frame de forma não-bloqueante"""
        try:
            return self.frame_queue.get_nowait()
        except queue.Empty:
            return None
        except Exception as e:
            logging.debug(f"Erro ao obter frame: {str(e)}")
            return None

    def get_camera_info(self):
        """Retorna informações da câmera em uso"""
        if self.camera_type == "rtsp":
            return f"RTSP: {self.rtsp_url}"
        elif self.camera_type == "webcam":
            return "Webcam local"
        else:
            return "Nenhuma câmera"

    def cleanup(self):
        """Limpeza rápida"""
        self.running = False
        self.camera_initialized = False

        if self.proc:
            try:
                self.proc.terminate()
                self.proc.wait(timeout=1)
            except:
                try:
                    self.proc.kill()
                except:
                    pass
            self.proc = None

        if self.cap:
            try:
                self.cap.release()
            except:
                pass
            self.cap = None

        try:
            while not self.frame_queue.empty():
                self.frame_queue.get_nowait()
        except:
            pass

        logging.info("Câmera finalizada")

    def cleanup_rtsp(self):
        """Finaliza o processo FFmpeg de forma mais robusta"""
        if self.proc:
            try:
                # Tenta terminar graciosamente
                self.proc.terminate()
                try:
                    self.proc.wait(timeout=1)
                except subprocess.TimeoutExpired:
                    # Se não responder, força o kill
                    self.proc.kill()
                    self.proc.wait(timeout=1)

                # Lê qualquer saída residual para evitar deadlocks
                try:
                    self.proc.stdout.read()
                    self.proc.stderr.read()
                except:
                    pass

            except Exception as e:
                logging.error(f"Erro ao finalizar FFmpeg: {str(e)}")
            finally:
                self.proc = None

    def cleanup_webcam(self):
        """Libera a webcam completamente"""
        if self.cap:
            try:
                # Para a captura primeiro
                self.running = False

                # Espera a thread terminar
                if hasattr(self, "capture_thread") and self.capture_thread.is_alive():
                    self.capture_thread.join(timeout=2.0)

                # Libera a câmera
                self.cap.release()

            except Exception as e:
                logging.error(f"Erro ao liberar webcam: {str(e)}")
            finally:
                self.cap = None

    def clear_frame_queue(self):
        """Limpa a fila de frames"""
        try:
            while not self.frame_queue.empty():
                try:
                    self.frame_queue.get_nowait()
                except queue.Empty:
                    break
        except Exception as e:
            logging.error(f"Erro ao limpar fila de frames: {str(e)}")
