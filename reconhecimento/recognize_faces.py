# recognize_faces.py - INTERFACE OTIMIZADA COM RASTREAMENTO DE UM ÚNICO ROSTO
import cv2
import time
import logging
import os
import sys
import numpy as np
from pathlib import Path

# Importações para desenho de texto Unicode
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).parent))

from camera_manager import CameraManager
from face_processor import FaceProcessor

# Configurações otimizadas para catraca
os.environ["QT_QPA_PLATFORM"] = "xcb"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
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


class UnicodeTextDrawer:
    """Desenha texto Unicode em imagens OpenCV usando PIL"""

    def __init__(self, font_size=20):
        self.font_size = font_size
        self.font = None
        # Lista de possíveis caminhos para fontes com suporte a acentos
        font_paths = [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
            "/System/Library/Fonts/Arial.ttf",  # macOS
            "C:\\Windows\\Fonts\\Arial.ttf",  # Windows
            "arial.ttf",
        ]
        for path in font_paths:
            try:
                self.font = ImageFont.truetype(path, self.font_size)
                logging.info(f"Fonte carregada: {path}")
                break
            except:
                continue
        if self.font is None:
            self.font = ImageFont.load_default()
            logging.warning(
                "Fonte TrueType não encontrada. Usando fonte padrão (acentos podem falhar)."
            )

    def draw_text(self, img_cv, text, pos, color=(255, 255, 255), font_size=None):
        """Desenha texto Unicode na imagem OpenCV e retorna a imagem modificada."""
        if font_size is not None and font_size != self.font_size:
            # Tenta criar fonte com tamanho diferente
            try:
                font = ImageFont.truetype(self.font.path, font_size)
            except:
                font = self.font
        else:
            font = self.font

        # Converte OpenCV BGR para PIL RGB
        img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        draw = ImageDraw.Draw(pil_img)
        # PIL espera cor em RGB, então invertemos a tupla BGR
        draw.text(pos, text, font=font, fill=(color[2], color[1], color[0]))
        # Converte de volta para OpenCV BGR
        img_cv = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        return img_cv


class FaceRecognizer:
    def __init__(self):
        # Configurações de qualidade/desempenho equilibradas
        self.rtsp_url = (
            "rtsp://admin:Evento0128@192.168.1.101:559/Streaming/Channels/101"
        )
        self.width = 640          # Aumentado para melhor qualidade
        self.height = 480
        self.target_fps = 15      # FPS mais suave

        # Inicializa os módulos
        self.camera_manager = CameraManager(
            self.rtsp_url, self.width, self.height, self.target_fps
        )
        self.face_processor = FaceProcessor(threshold=0.65)
        # Análise de expressões sempre ativa
        self.face_processor.enable_expression_analysis = True

        # Inicializa desenhador Unicode
        self.text_drawer = UnicodeTextDrawer(font_size=18)

        self.running = False
        self.window_created = False
        self.last_recognition_time = 0
        self.recognition_cooldown = 1  # Reduzido devido ao tracking contínuo

        # Estatísticas e monitoramento
        self.expression_alerts = []
        self.alert_cooldown = 10  # segundos entre alertas repetidos
        self.last_alert_time = {}
        self.performance_stats = {
            "frames_processed": 0,
            "expression_analyses": 0,
            "start_time": time.time(),
        }

    def initialize_system(self):
        """Inicialização otimizada"""
        try:
            logging.info("Inicializando sistema de reconhecimento...")

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
            cv2.namedWindow("Sistema de Reconhecimento Facial", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Sistema de Reconhecimento Facial", 960, 720)  # Maior para melhor visualização
            self.window_created = True
            return True
        except Exception as e:
            logging.warning(f"Janela não criada: {str(e)}")
            return False

    def handle_keypress(self, key):
        """Manipulação otimizada de teclas - apenas Q para sair e S para estatísticas"""
        if key == ord("q") or key == ord("Q"):
            logging.info("Solicitação de saída pelo usuário")
            return False
        elif key == ord("s") or key == ord("S"):
            self.show_statistics()
        return True

    def show_statistics(self):
        """Mostra estatísticas do sistema"""
        try:
            current_time = time.time()
            runtime = current_time - self.performance_stats["start_time"]
            fps = (
                self.performance_stats["frames_processed"] / runtime
                if runtime > 0
                else 0
            )

            stats_text = [
                f"Tempo de execucao: {runtime:.1f}s",
                f"Frames processados: {self.performance_stats['frames_processed']}",
                f"FPS medio: {fps:.1f}",
                f"Analises de expressao: {self.performance_stats['expression_analyses']}",
            ]

            # Adiciona estatísticas de expressões se disponíveis
            expr_stats = self.face_processor.get_expression_statistics()
            if expr_stats:
                stats_text.append(
                    f"Tendencia: {expr_stats.get('trend_analysis', 'N/A')}"
                )
                stats_text.append(
                    f"Historico: {expr_stats.get('history_size', 0)} amostras"
                )

            print("\n" + "=" * 50)
            print("📊 ESTATISTICAS DO SISTEMA")
            print("=" * 50)
            for stat in stats_text:
                print(f"  {stat}")
            print("=" * 50)

        except Exception as e:
            logging.debug(f"Erro ao mostrar estatisticas: {str(e)}")

    def draw_status_overlay(self, frame, fps, status):
        """
        Desenha informações na tela de forma discreta:
        - Barra superior com status e FPS
        - Emoção dominante (canto superior direito) se houver rosto alvo
        - Alertas de cansaço/tristeza (canto inferior esquerdo) se houver rosto alvo
        """
        try:
            h, w = frame.shape[:2]

            # --- Barra superior semi-transparente fina ---
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (w, 30), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

            # Status do sistema (esquerda)
            color_status = (0, 255, 0) if status == "OPERACIONAL" else (0, 0, 255)
            cv2.putText(
                frame,
                f"{status}",
                (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                color_status,
                1,
            )

            # FPS (direita na barra)
            fps_text = f"FPS: {fps:.1f}"
            (tw, th), _ = cv2.getTextSize(fps_text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
            cv2.putText(
                frame,
                fps_text,
                (w - tw - 10, 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (255, 255, 255),
                1,
            )

            # --- Emoção dominante (canto superior direito) apenas se houver rosto alvo ---
            if (
                self.face_processor.tracking["active"]
                and self.face_processor.expression_results
            ):
                expr = self.face_processor.expression_results
                basic = expr.get("basic_emotions", {})
                dominant = basic.get("dominant_emotion", "neutral")
                conf = basic.get("confidence", 0)

                emotion_translation = {
                    "happy": "ALEGRE",
                    "sad": "TRISTE",
                    "angry": "RAIVA",
                    "surprise": "SURPRESA",
                    "fear": "MEDO",
                    "disgust": "DESGOSTO",
                    "neutral": "NEUTRO",
                }
                emoji_map = {
                    "happy": "😊",
                    "sad": "😔",
                    "angry": "😠",
                    "surprise": "😲",
                    "fear": "😨",
                    "disgust": "🤢",
                    "neutral": "😐",
                }
                emoji = emoji_map.get(dominant, "😐")
                emotion_display = emotion_translation.get(dominant, "NEUTRO")
                emotion_text = f"{emoji} {emotion_display} {conf/100:.1%}"

                # Fundo semi-transparente
                bbox = self.text_drawer.font.getbbox(emotion_text)
                text_w = bbox[2] - bbox[0] if bbox else len(emotion_text) * 10
                text_h = bbox[3] - bbox[1] if bbox else 15
                overlay2 = frame.copy()
                cv2.rectangle(
                    overlay2,
                    (w - text_w - 15, 35),
                    (w - 5, 35 + text_h + 10),
                    (0, 0, 0),
                    -1,
                )
                cv2.addWeighted(overlay2, 0.5, frame, 0.5, 0, frame)

                frame = self.text_drawer.draw_text(
                    frame,
                    emotion_text,
                    (w - text_w - 10, 40),
                    color=(255, 255, 255),
                    font_size=16,
                )

            # --- Alertas de cansaço/tristeza (canto inferior esquerdo) apenas se houver rosto alvo ---
            if (
                self.face_processor.tracking["active"]
                and self.face_processor.expression_results
            ):
                expr = self.face_processor.expression_results
                fatigue = expr.get("fatigue", {})
                sadness = expr.get("sadness", {})
                alert_x, alert_y = 10, h - 10
                line_height = 20
                alerts = []
                if fatigue.get("score", 0) > 0.3:
                    alerts.append(("cansaco", fatigue.get("level", "Baixo")))
                if sadness.get("score", 0) > 0.3:
                    alerts.append(("tristeza", sadness.get("level", "Baixo")))

                if alerts:
                    # Desenha fundo agrupado
                    max_w = 0
                    for label, level in alerts:
                        text = f"{label}: {level}"
                        bbox = self.text_drawer.font.getbbox(text)
                        tw = bbox[2] - bbox[0] if bbox else len(text) * 8
                        if tw > max_w:
                            max_w = tw
                    overlay3 = frame.copy()
                    cv2.rectangle(
                        overlay3,
                        (alert_x - 5, alert_y - len(alerts) * line_height - 5),
                        (alert_x + max_w + 10, alert_y + 5),
                        (0, 0, 0),
                        -1,
                    )
                    cv2.addWeighted(overlay3, 0.5, frame, 0.5, 0, frame)

                    for label, level in reversed(alerts):
                        color = (0, 0, 255) if level == "Alto" else (0, 165, 255)
                        frame = self.text_drawer.draw_text(
                            frame,
                            f"{label}: {level}",
                            (alert_x, alert_y - line_height),
                            color=color,
                            font_size=14,
                        )
                        alert_y -= line_height

            return frame
        except Exception as e:
            logging.debug(f"Erro no overlay: {str(e)}")
            return frame

    def process_expression_alerts(self, expression_results):
        """Processa alertas de expressões e emoções críticas (apenas do rosto alvo)"""
        try:
            if not expression_results:
                return

            current_time = time.time()
            alerts = []

            # Emoções básicas
            basic_emotions = expression_results.get("basic_emotions", {})
            dominant_emotion = basic_emotions.get("dominant_emotion", "neutral")
            confidence = basic_emotions.get("confidence", 0)

            # Alertas para emoções fortes
            if confidence > 0.8:
                if dominant_emotion == "angry":
                    alert_key = "emotion_angry"
                    if (
                        current_time - self.last_alert_time.get(alert_key, 0)
                        > self.alert_cooldown
                    ):
                        alerts.append("🚨 EMOCAO FORTE: RAIVA")
                        self.last_alert_time[alert_key] = current_time

                elif dominant_emotion == "fear":
                    alert_key = "emotion_fear"
                    if (
                        current_time - self.last_alert_time.get(alert_key, 0)
                        > self.alert_cooldown
                    ):
                        alerts.append("🚨 EMOCAO FORTE: MEDO")
                        self.last_alert_time[alert_key] = current_time

            # Alertas de cansaço
            fatigue_level = expression_results.get("fatigue", {}).get("level", "Baixo")
            fatigue_score = expression_results.get("fatigue", {}).get("score", 0)

            if fatigue_level == "Alto" and fatigue_score > 0.7:
                alert_key = "fatigue_high"
                if (
                    current_time - self.last_alert_time.get(alert_key, 0)
                    > self.alert_cooldown
                ):
                    alerts.append("🚨 CANSACO ELEVADO")
                    self.last_alert_time[alert_key] = current_time

            # Adiciona alertas à lista
            for alert in alerts:
                if alert not in self.expression_alerts[-5:]:
                    self.expression_alerts.append(alert)

            # Mantém apenas os últimos 5 alertas
            if len(self.expression_alerts) > 5:
                self.expression_alerts = self.expression_alerts[-5:]

        except Exception as e:
            logging.debug(f"Erro no processamento de alertas: {str(e)}")

    def check_window_closed(self):
        """Verifica se a janela foi fechada pelo X"""
        try:
            window_prop = cv2.getWindowProperty(
                "Sistema de Reconhecimento Facial", cv2.WND_PROP_VISIBLE
            )
            if window_prop <= 0:
                logging.info("Janela fechada pelo usuário")
                return False
            return True
        except:
            logging.info("Janela fechada pelo usuário")
            return False

    def run(self):
        """Loop principal otimizado"""
        try:
            if not self.initialize_system():
                logging.error("Falha na inicialização do sistema")
                return

            self.create_display_window()
            self.running = True

            logging.info("Sistema de reconhecimento iniciado")
            fps_counter = 0
            fps_time = time.time()
            last_frame_time = time.time()
            last_statistics_time = time.time()

            while self.running:
                try:
                    if not self.check_window_closed():
                        break

                    current_time = time.time()
                    elapsed = current_time - last_frame_time
                    if elapsed < 1.0 / self.target_fps:
                        time.sleep(0.001)
                        continue

                    last_frame_time = current_time

                    frame = self.camera_manager.get_frame()
                    if frame is None:
                        logging.warning("Frame vazio recebido")
                        time.sleep(0.1)
                        continue

                    self.performance_stats["frames_processed"] += 1

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
                        frame = self.face_processor.process_frame(frame)
                        # Processa alertas (apenas se houver resultados do rosto alvo)
                        expr_results = self.face_processor.expression_results
                        if expr_results:
                            self.performance_stats["expression_analyses"] += 1
                            self.process_expression_alerts(expr_results)

                    fps_counter += 1
                    if current_time - fps_time >= 1.0:
                        fps = fps_counter / (current_time - fps_time)
                        fps_counter = 0
                        fps_time = current_time
                    else:
                        fps = 1.0 / elapsed if elapsed > 0 else 0

                    status = (
                        "OPERACIONAL"
                        if self.face_processor.model_loaded
                        else "SEM TREINAMENTO"
                    )

                    frame = self.draw_status_overlay(frame, fps, status)

                    cv2.imshow("Sistema de Reconhecimento Facial", frame)

                    if current_time - last_statistics_time >= 30:
                        runtime = current_time - self.performance_stats["start_time"]
                        avg_fps = self.performance_stats["frames_processed"] / runtime
                        logging.info(
                            f"Estatisticas: {avg_fps:.1f} FPS, {self.performance_stats['frames_processed']} frames, {self.performance_stats['expression_analyses']} analises"
                        )
                        last_statistics_time = current_time

                    key = cv2.waitKey(1) & 0xFF
                    if not self.handle_keypress(key):
                        break

                except Exception as e:
                    logging.error(f"Erro no loop principal: {str(e)}")
                    time.sleep(0.1)

        except KeyboardInterrupt:
            logging.info("Interrupção pelo usuário")
        except Exception as e:
            logging.error(f"Erro crítico: {str(e)}")
        finally:
            self.cleanup()

    def print_final_statistics(self):
        """Imprime estatísticas finais do sistema"""
        try:
            current_time = time.time()
            runtime = current_time - self.performance_stats["start_time"]

            print("\n" + "=" * 60)
            print("📊 ESTATISTICAS FINAIS")
            print("=" * 60)
            print(f"  ⏰ Tempo total de execucao: {runtime:.1f} segundos")
            print(
                f"  📷 Frames processados: {self.performance_stats['frames_processed']}"
            )
            if runtime > 0:
                print(
                    f"  🎯 FPS medio: {self.performance_stats['frames_processed'] / runtime:.1f}"
                )
            else:
                print("  🎯 FPS medio: N/A")
            print(
                f"  😊 Analises de expressao: {self.performance_stats['expression_analyses']}"
            )
            print(f"  ⚠️  Alertas gerados: {len(self.expression_alerts)}")

            expr_stats = self.face_processor.get_expression_statistics()
            if expr_stats:
                print(
                    f"  📈 Tendencia final: {expr_stats.get('trend_analysis', 'N/A')}"
                )
                print(
                    f"  🗂️  Amostras no historico: {expr_stats.get('history_size', 0)}"
                )

            print("=" * 60)

        except Exception as e:
            logging.debug(f"Erro ao imprimir estatisticas finais: {str(e)}")

    def cleanup(self):
        """Limpeza otimizada de recursos"""
        try:
            self.running = False
            self.print_final_statistics()

            logging.info("Iniciando limpeza de recursos...")

            if self.window_created:
                try:
                    cv2.destroyAllWindows()
                    cv2.waitKey(1)
                    time.sleep(0.1)
                except Exception as e:
                    logging.error(f"Erro ao fechar janelas: {str(e)}")

            if hasattr(self, "face_processor"):
                try:
                    self.face_processor.cleanup()
                    time.sleep(0.1)
                except Exception as e:
                    logging.error(f"Erro ao limpar processador facial: {str(e)}")

            if hasattr(self, "camera_manager"):
                try:
                    self.camera_manager.cleanup()
                    time.sleep(0.2)
                except Exception as e:
                    logging.error(f"Erro ao limpar gerenciador de câmera: {str(e)}")

            logging.info("✅ Sistema finalizado corretamente")

        except Exception as e:
            logging.error(f"Erro na limpeza: {str(e)}")


if __name__ == "__main__":
    try:
        print("=" * 60)
        print("🚀 SISTEMA DE RECONHECIMENTO FACIAL - INICIANDO")
        print("=" * 60)
        print("Configurações otimizadas para:")
        print("  ✓ Baixo número de imagens (1-5 por pessoa)")
        print("  ✓ Velocidade de resposta")
        print("  ✓ Rastreamento de um único rosto com prioridade")
        print("  ✓ Análise de expressões ativa apenas quando há rosto")
        print("  ✓ Resolução: 640x480, FPS: 15")
        print("=" * 60)
        print("Controles:")
        print("  Q - Sair do sistema")
        print("  S - Mostrar estatisticas")
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