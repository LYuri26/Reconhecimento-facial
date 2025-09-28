# recognize_faces.py - COMPLETO COM ANÁLISE DE EXPRESSÕES
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
            cv2.namedWindow("sistema de reconhecimento", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("sistema de reconhecimento", 800, 600)
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
        elif key == ord("e") or key == ord("E"):
            # Toggle análise de expressões
            self.face_processor.enable_expression_analysis = (
                not self.face_processor.enable_expression_analysis
            )
            status = (
                "ATIVADA"
                if self.face_processor.enable_expression_analysis
                else "DESATIVADA"
            )
            logging.info(f"Análise de expressões {status}")
            self.expression_alerts.append(f"Expressões {status}")
        elif key == ord("s") or key == ord("S"):
            # Mostrar estatísticas
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
                f"Tempo de execução: {runtime:.1f}s",
                f"Frames processados: {self.performance_stats['frames_processed']}",
                f"FPS médio: {fps:.1f}",
                f"Análises de expressão: {self.performance_stats['expression_analyses']}",
                f"Expressões ativas: {'SIM' if self.face_processor.enable_expression_analysis else 'NÃO'}",
            ]

            # Adiciona estatísticas de expressões se disponíveis
            expr_stats = self.face_processor.get_expression_statistics()
            if expr_stats:
                stats_text.append(
                    f"Tendência: {expr_stats.get('trend_analysis', 'N/A')}"
                )
                stats_text.append(
                    f"Histórico: {expr_stats.get('history_size', 0)} amostras"
                )

            print("\n" + "=" * 50)
            print("📊 ESTATÍSTICAS DO SISTEMA")
            print("=" * 50)
            for stat in stats_text:
                print(f"  {stat}")
            print("=" * 50)

        except Exception as e:
            logging.debug(f"Erro ao mostrar estatísticas: {str(e)}")

    def draw_status_overlay(self, frame, fps, status):
        """Overlay otimizado de status"""
        try:
            h, w = frame.shape[:2]

            # Fundo semi-transparente para status
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (w, 90), (0, 0, 0), -1)
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

            # Status da análise de expressões
            expr_status = (
                "ATIVADA"
                if self.face_processor.enable_expression_analysis
                else "DESATIVADA"
            )
            expr_color = (
                (0, 255, 0)
                if self.face_processor.enable_expression_analysis
                else (0, 0, 255)
            )
            cv2.putText(
                frame,
                f"EXPRESSOES: {expr_status}",
                (10, 75),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                expr_color,
                1,
            )

            # Instruções
            instructions = [
                "Q - Sair",
                "R - Recarregar Modelo",
                "E - Toggle Expressoes",
                "S - Estatisticas",
            ]

            y_offset = 25
            for instruction in instructions:
                cv2.putText(
                    frame,
                    instruction,
                    (w - 200, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (200, 200, 200),
                    1,
                )
                y_offset += 20

            # Alertas recentes de expressões
            if self.expression_alerts:
                recent_alerts = self.expression_alerts[-3:]  # Últimos 3 alertas
                alert_y = h - 10
                for alert in reversed(recent_alerts):
                    cv2.putText(
                        frame,
                        alert,
                        (10, alert_y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.4,
                        (0, 165, 255),  # Laranja
                        1,
                    )
                    alert_y -= 15

            return frame
        except Exception as e:
            logging.debug(f"Erro no overlay: {str(e)}")
            return frame

    def process_expression_alerts(self, expression_results):
        """Processa alertas de expressões críticas"""
        try:
            if not expression_results:
                return

            current_time = time.time()
            alerts = []

            # Verifica níveis críticos
            fatigue_level = expression_results.get("fatigue", {}).get("level", "Baixo")
            fatigue_score = expression_results.get("fatigue", {}).get("score", 0)

            sadness_level = expression_results.get("sadness", {}).get("level", "Baixo")
            sadness_score = expression_results.get("sadness", {}).get("score", 0)

            demotivation_level = expression_results.get("demotivation", {}).get(
                "level", "Baixo"
            )
            demotivation_score = expression_results.get("demotivation", {}).get(
                "score", 0
            )

            # Alertas de cansaço
            if fatigue_level == "Alto" and fatigue_score > 0.7:
                alert_key = "fatigue_high"
                if (
                    current_time - self.last_alert_time.get(alert_key, 0)
                    > self.alert_cooldown
                ):
                    alerts.append("🚨 ALERTA: CANSACO ELEVADO")
                    self.last_alert_time[alert_key] = current_time
                    logging.warning("ALERTA: Nível elevado de cansaço detectado")

            elif fatigue_level == "Médio" and fatigue_score > 0.4:
                alert_key = "fatigue_medium"
                if (
                    current_time - self.last_alert_time.get(alert_key, 0)
                    > self.alert_cooldown * 2
                ):
                    alerts.append("⚠️ ATENCAO: SINAIS DE CANSACO")
                    self.last_alert_time[alert_key] = current_time

            # Alertas de tristeza
            if sadness_level == "Alto" and sadness_score > 0.7:
                alert_key = "sadness_high"
                if (
                    current_time - self.last_alert_time.get(alert_key, 0)
                    > self.alert_cooldown
                ):
                    alerts.append("🚨 ALERTA: TRISTEZA ELEVADA")
                    self.last_alert_time[alert_key] = current_time
                    logging.warning("ALERTA: Nível elevado de tristeza detectado")

            elif sadness_level == "Médio" and sadness_score > 0.4:
                alert_key = "sadness_medium"
                if (
                    current_time - self.last_alert_time.get(alert_key, 0)
                    > self.alert_cooldown * 2
                ):
                    alerts.append("⚠️ ATENCAO: SINAIS DE TRISTEZA")
                    self.last_alert_time[alert_key] = current_time

            # Alertas de desânimo
            if demotivation_level == "Alto" and demotivation_score > 0.7:
                alert_key = "demotivation_high"
                if (
                    current_time - self.last_alert_time.get(alert_key, 0)
                    > self.alert_cooldown
                ):
                    alerts.append("🚨 ALERTA: DESANIMO ELEVADO")
                    self.last_alert_time[alert_key] = current_time
                    logging.warning("ALERTA: Nível elevado de desânimo detectado")

            elif demotivation_level == "Médio" and demotivation_score > 0.4:
                alert_key = "demotivation_medium"
                if (
                    current_time - self.last_alert_time.get(alert_key, 0)
                    > self.alert_cooldown * 2
                ):
                    alerts.append("⚠️ ATENCAO: SINAIS DE DESANIMO")
                    self.last_alert_time[alert_key] = current_time

            # Adiciona alertas à lista
            for alert in alerts:
                if (
                    alert not in self.expression_alerts[-5:]
                ):  # Evita repetições recentes
                    self.expression_alerts.append(alert)

            # Mantém apenas os últimos 10 alertas
            if len(self.expression_alerts) > 10:
                self.expression_alerts = self.expression_alerts[-10:]

        except Exception as e:
            logging.debug(f"Erro no processamento de alertas: {str(e)}")

    def run(self):
        """Loop principal otimizado para catraca com análise de expressões"""
        try:
            if not self.initialize_system():
                logging.error("Falha na inicialização do sistema")
                return

            self.create_display_window()
            self.running = True

            logging.info("sistema de reconhecimento iniciado")
            fps_counter = 0
            fps_time = time.time()
            last_frame_time = time.time()
            last_statistics_time = time.time()

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

                    # Atualiza estatísticas
                    self.performance_stats["frames_processed"] += 1

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
                        # Processamento otimizado do frame (inclui reconhecimento e expressões)
                        frame = self.face_processor.process_frame(frame)

                        # Processamento adicional de expressões (fallback)
                        if (
                            hasattr(self.face_processor, "process_expressions")
                            and self.face_processor.enable_expression_analysis
                        ):
                            try:
                                frame, expression_results = (
                                    self.face_processor.process_expressions(frame)
                                )
                                if expression_results:
                                    self.performance_stats["expression_analyses"] += 1
                                    self.process_expression_alerts(expression_results)
                            except Exception as e:
                                logging.debug(
                                    f"Erro na análise secundária de expressões: {str(e)}"
                                )

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
                    cv2.imshow("sistema de reconhecimento", frame)

                    # Log de estatísticas a cada 30 segundos
                    if current_time - last_statistics_time >= 30:
                        runtime = current_time - self.performance_stats["start_time"]
                        avg_fps = self.performance_stats["frames_processed"] / runtime
                        logging.info(
                            f"Estatísticas: {avg_fps:.1f} FPS, {self.performance_stats['frames_processed']} frames, {self.performance_stats['expression_analyses']} análises de expressão"
                        )
                        last_statistics_time = current_time

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

    def print_final_statistics(self):
        """Imprime estatísticas finais do sistema"""
        try:
            current_time = time.time()
            runtime = current_time - self.performance_stats["start_time"]

            print("\n" + "=" * 60)
            print("📊 ESTATÍSTICAS FINAIS")
            print("=" * 60)
            print(f"  ⏰ Tempo total de execução: {runtime:.1f} segundos")
            print(
                f"  📷 Frames processados: {self.performance_stats['frames_processed']}"
            )
            print(
                f"  🎯 FPS médio: {self.performance_stats['frames_processed'] / runtime:.1f}"
                if runtime > 0
                else "  🎯 FPS médio: N/A"
            )
            print(
                f"  😊 Análises de expressão: {self.performance_stats['expression_analyses']}"
            )
            print(f"  ⚠️  Alertas gerados: {len(self.expression_alerts)}")

            # Estatísticas de expressões
            expr_stats = self.face_processor.get_expression_statistics()
            if expr_stats:
                print(
                    f"  📈 Tendência final: {expr_stats.get('trend_analysis', 'N/A')}"
                )
                print(
                    f"  🗂️  Amostras no histórico: {expr_stats.get('history_size', 0)}"
                )

            print("=" * 60)

        except Exception as e:
            logging.debug(f"Erro ao imprimir estatísticas finais: {str(e)}")

    def cleanup(self):
        """Limpeza otimizada de recursos"""
        try:
            self.running = False

            # Imprime estatísticas finais
            self.print_final_statistics()

            # Limpeza dos módulos
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
        print("🚀 SISTEMA DE RECONHECIMENTO FACIAL - INICIANDO")
        print("=" * 60)
        print("Configurações otimizadas para:")
        print("  ✓ Baixo número de imagens (1-5 por pessoa)")
        print("  ✓ Velocidade de resposta")
        print("  ✓ Confiabilidade em ambiente de catraca")
        print("  ✓ Análise de expressões (cansaço, tristeza, desânimo)")
        print("=" * 60)
        print("Controles:")
        print("  Q - Sair do sistema")
        print("  R - Recarregar modelo de reconhecimento")
        print("  E - Ativar/desativar análise de expressões")
        print("  S - Mostrar estatísticas")
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
