import cv2
import numpy as np
import subprocess
import threading
import queue
import time
import logging
import os


class CameraManager:
    def __init__(self, rtsp_url=None, width=640, height=480, target_fps=10):
        self.rtsp_url = rtsp_url
        self.width = width
        self.height = height
        self.target_fps = target_fps
        self.frame_skip = 2
        self.frame_queue = queue.Queue(maxsize=3)
        self.running = False
        self.camera_initialized = False
        self.proc = None
        self.cap = None
        self.camera_type = None  # 'rtsp' ou 'webcam'
        self.last_frame_time = 0

    def initialize_camera(self):
        """Inicialização da câmera - tenta RTSP primeiro, depois webcam"""
        try:
            # Primeiro tenta a câmera RTSP
            if self.rtsp_url:
                logging.info(f"Tentando conectar com câmera RTSP: {self.rtsp_url}")
                if self.initialize_rtsp():
                    self.camera_type = "rtsp"
                    logging.info("Câmera RTSP inicializada com sucesso")
                    return True

            # Se RTSP falhar, tenta webcam
            logging.info("Tentando conectar com webcam...")
            if self.initialize_webcam():
                self.camera_type = "webcam"
                logging.info("Webcam inicializada com sucesso")
                return True

            logging.error("Nenhuma câmera disponível")
            return False

        except Exception as e:
            logging.error(f"Erro na inicialização da câmera: {str(e)}")
            self.camera_initialized = False
            return False

    def initialize_rtsp(self):
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
            if not self.test_rtsp_connection(timeout=5):
                raise RuntimeError("Não foi possível conectar à câmera RTSP")

            self.camera_initialized = True
            self.running = True
            threading.Thread(target=self.capture_rtsp_frames, daemon=True).start()
            return True

        except Exception as e:
            logging.warning(f"Falha na câmera RTSP: {str(e)}")
            self.cleanup_rtsp()
            return False

    def initialize_webcam(self):
        """Inicialização da webcam"""
        try:
            # Tenta diferentes índices de câmera (0, 1, 2)
            for camera_index in [0, 1, 2]:
                try:
                    self.cap = cv2.VideoCapture(camera_index, cv2.CAP_ANY)
                    if self.cap.isOpened():
                        # Configura a webcam
                        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
                        self.cap.set(cv2.CAP_PROP_FPS, self.target_fps)
                        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduzir buffer

                        # Testa se consegue ler um frame
                        ret, frame = self.cap.read()
                        if ret and frame is not None:
                            self.camera_initialized = True
                            self.running = True
                            # Inicia a captura em thread separada
                            self.capture_thread = threading.Thread(
                                target=self.capture_webcam_frames, daemon=True
                            )
                            self.capture_thread.start()
                            logging.info(f"Webcam encontrada no índice {camera_index}")
                            return True
                        else:
                            self.cap.release()
                            self.cap = None
                except Exception as e:
                    logging.warning(
                        f"Erro ao acessar webcam índice {camera_index}: {str(e)}"
                    )
                    if self.cap:
                        self.cap.release()
                        self.cap = None

            logging.error("Nenhuma webcam encontrada")
            return False

        except Exception as e:
            logging.error(f"Erro na inicialização da webcam: {str(e)}")
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
        """Captura frames da câmera RTSP continuamente"""
        frame_size = self.width * self.height * 3
        frame_counter = 0

        while self.running and self.camera_initialized and self.camera_type == "rtsp":
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
                logging.error(f"Erro na captura de frames RTSP: {str(e)}")
                time.sleep(0.1)

    def capture_webcam_frames(self):
        """Captura frames da webcam continuamente"""
        frame_counter = 0

        while self.running and self.camera_initialized and self.camera_type == "webcam":
            try:
                ret, frame = self.cap.read()
                if not ret or frame is None:
                    logging.warning("Erro ao capturar frame da webcam")
                    time.sleep(0.1)
                    continue

                # Redimensiona se necessário
                if frame.shape[1] != self.width or frame.shape[0] != self.height:
                    frame = cv2.resize(frame, (self.width, self.height))

                # Controle de frame skip e FPS
                current_time = time.time()
                if current_time - self.last_frame_time < 1.0 / self.target_fps:
                    continue

                self.last_frame_time = current_time
                frame_counter += 1

                if frame_counter % (self.frame_skip + 1) != 0:
                    continue

                if not self.frame_queue.full():
                    self.frame_queue.put(frame)

            except Exception as e:
                logging.error(f"Erro na captura de frames webcam: {str(e)}")
                time.sleep(0.1)

    def get_frame(self):
        """Obtém um frame da fila com timeout"""
        try:
            # Timeout curto para não travar a interface
            frame = self.frame_queue.get(timeout=0.1)
            return frame
        except queue.Empty:
            return None
        except Exception as e:
            logging.error(f"Erro ao obter frame: {str(e)}")
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
        """Limpeza de recursos da câmera"""
        self.running = False
        self.camera_initialized = False

        # Limpa RTSP
        self.cleanup_rtsp()

        # Limpa webcam
        self.cleanup_webcam()

        # Limpa a fila de frames
        self.clear_frame_queue()

        logging.info("Câmera finalizada com segurança")

    def cleanup_rtsp(self):
        """Finaliza o processo FFmpeg"""
        if self.proc:
            try:
                self.proc.terminate()
                self.proc.wait(timeout=2)
            except Exception as e:
                logging.error(f"Erro ao finalizar FFmpeg: {str(e)}")
                try:
                    self.proc.kill()
                except:
                    pass
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
