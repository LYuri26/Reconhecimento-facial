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
        """Inicialização da câmera RTSP com melhor logging"""
        try:
            # Primeiro testa a conexão com timeout
            test_cmd = [
                "ffmpeg",
                "-rtsp_transport",
                "tcp",
                "-timeout",
                "5000000",
                "-i",
                self.rtsp_url,
                "-t",
                "3",  # Apenas 3 segundos de teste
                "-f",
                "null",
                "-",
            ]

            logging.info("Testando conexão RTSP...")
            result = subprocess.run(test_cmd, capture_output=True, timeout=10)

            if result.returncode != 0:
                error_msg = result.stderr.decode()
                logging.error(f"Teste de conexão RTSP falhou: {error_msg}")

                # Verifica erros comuns
                if "Connection refused" in error_msg:
                    logging.error("Erro: Conexão recusada - verifique IP/porta")
                elif "Unauthorized" in error_msg:
                    logging.error("Erro: Não autorizado - verifique usuário/senha")
                elif "Not found" in error_msg:
                    logging.error("Erro: URL não encontrada - verifique o caminho RTSP")
                elif "Connection timed out" in error_msg:
                    logging.error("Erro: Timeout - verifique conectividade de rede")

                return False

            # Se o teste passou, inicia a captura contínua
            ffmpeg_cmd = [
                "ffmpeg",
                "-rtsp_transport",
                "tcp",
                "-timeout",
                "5000000",
                "-i",
                self.rtsp_url,
                "-loglevel",
                "info",  # Mais informações para debug
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

            logging.info(f"Iniciando FFmpeg com comando: {' '.join(ffmpeg_cmd)}")
            self.proc = subprocess.Popen(
                ffmpeg_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=10**8,
            )

            # Aguarda inicialização e lê stderr para debug
            time.sleep(2)
            stderr_lines = []
            for _ in range(10):  # Lê as primeiras linhas de erro
                if self.proc.stderr.readable():
                    line = self.proc.stderr.readline().decode().strip()
                    if line:
                        stderr_lines.append(line)
                        logging.info(f"FFmpeg: {line}")

            # Verifica se o processo está rodando
            if self.proc.poll() is not None:
                error_output = "\n".join(stderr_lines)
                logging.error(f"FFmpeg terminou. Erro: {error_output}")
                return False

            self.camera_initialized = True
            self.running = True
            threading.Thread(target=self.capture_rtsp_frames, daemon=True).start()

            logging.info("Captura RTSP iniciada com sucesso")
            return True

        except Exception as e:
            logging.error(f"Falha crítica na câmera RTSP: {str(e)}")
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
        error_count = 0

        while self.running and self.camera_initialized and self.camera_type == "rtsp":
            try:
                # Lê stderr para debug
                if self.proc.stderr and error_count < 5:
                    err_line = self.proc.stderr.readline().decode().strip()
                    if err_line:
                        logging.debug(f"FFmpeg: {err_line}")
                        error_count += 1

                raw_frame = self.proc.stdout.read(frame_size)
                if not raw_frame:
                    logging.warning("Nenhum dado recebido da RTSP")
                    time.sleep(1)
                    continue

                if len(raw_frame) != frame_size:
                    logging.warning(
                        f"Frame incompleto: {len(raw_frame)}/{frame_size} bytes"
                    )
                    continue

            except Exception as e:
                logging.error(f"Erro na captura de frames RTSP: {str(e)}")
                time.sleep(0.1)

    def capture_webcam_frames(self):
        """Captura frames da webcam continuamente - versão corrigida"""
        frame_counter = 0
        consecutive_errors = 0

        while self.running and self.camera_initialized and self.camera_type == "webcam":
            try:
                ret, frame = self.cap.read()
                if not ret or frame is None:
                    consecutive_errors += 1
                    logging.warning(
                        f"Erro ao capturar frame da webcam ({consecutive_errors}/10)"
                    )

                    if consecutive_errors >= 10:
                        logging.error(
                            "Muitos erros consecutivos, reiniciando webcam..."
                        )
                        self.cleanup_webcam()
                        time.sleep(1)
                        self.initialize_webcam()
                        consecutive_errors = 0

                    time.sleep(0.1)
                    continue

                consecutive_errors = 0  # Reset error counter

                # Redimensiona se necessário
                if frame.shape[1] != self.width or frame.shape[0] != self.height:
                    frame = cv2.resize(frame, (self.width, self.height))

                # Controle de FPS
                current_time = time.time()
                time_elapsed = current_time - self.last_frame_time
                if time_elapsed < 1.0 / self.target_fps:
                    continue

                self.last_frame_time = current_time
                frame_counter += 1

                # Aplica frame skip
                if frame_counter % (self.frame_skip + 1) != 0:
                    continue

                # Adiciona à fila (com timeout para evitar bloqueio)
                try:
                    if self.frame_queue.full():
                        self.frame_queue.get_nowait()  # Remove frame mais antigo

                    self.frame_queue.put(frame, timeout=0.1)
                    logging.debug(
                        f"Frame adicionado à fila. Tamanho: {self.frame_queue.qsize()}"
                    )

                except queue.Full:
                    logging.warning("Fila de frames cheia")
                except Exception as e:
                    logging.error(f"Erro ao adicionar frame na fila: {str(e)}")

            except Exception as e:
                logging.error(f"Erro crítico na captura de frames webcam: {str(e)}")
                time.sleep(0.1)

    def get_frame(self):
        """Obtém um frame da fila com fallbacks"""
        try:
            # Tenta obter da fila
            frame = self.frame_queue.get(timeout=0.5)
            return frame
        except queue.Empty:
            logging.debug("Fila vazia, tentando captura direta...")

            # Se é webcam, tenta capturar diretamente
            if self.camera_type == "webcam" and self.cap and self.cap.isOpened():
                try:
                    ret, frame = self.cap.read()
                    if ret and frame is not None:
                        # Redimensiona se necessário
                        if (
                            frame.shape[1] != self.width
                            or frame.shape[0] != self.height
                        ):
                            frame = cv2.resize(frame, (self.width, self.height))
                        return frame
                except Exception as e:
                    logging.error(f"Erro na captura direta: {str(e)}")

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
