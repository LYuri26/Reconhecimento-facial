import cv2
import numpy as np
import subprocess
import threading
import queue
import time
import logging
import os


class CameraManager:
    def __init__(self, rtsp_url, width=640, height=480, target_fps=10):
        self.rtsp_url = rtsp_url
        self.width = width
        self.height = height
        self.target_fps = target_fps
        self.frame_skip = 2
        self.frame_queue = queue.Queue(maxsize=3)
        self.running = False
        self.camera_initialized = False
        self.proc = None

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
            self.running = True
            threading.Thread(target=self.capture_frames, daemon=True).start()
            logging.info("Câmera inicializada com sucesso")
            return True

        except Exception as e:
            logging.error(f"Erro na inicialização da câmera: {str(e)}")
            self.camera_initialized = False
            return False

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

    def get_frame(self):
        """Obtém um frame da fila"""
        try:
            if not self.frame_queue.empty():
                return self.frame_queue.get()
            return None
        except Exception as e:
            logging.error(f"Erro ao obter frame: {str(e)}")
            return None

    def cleanup(self):
        """Limpeza de recursos da câmera"""
        self.running = False

        # Finaliza o processo FFmpeg
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

        logging.info("Câmera finalizada com segurança")
