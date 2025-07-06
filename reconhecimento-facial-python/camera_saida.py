import cv2
import time
import numpy as np
from database.db_operations import DBOperations
from face_recognition.detector import FaceDetector
from face_recognition.recognizer import FaceRecognizer
from config import Config
import sys


class CameraSaida:
    def __init__(self, camera_id=0):
        print("⏳ Inicializando modulo de saida...")
        self.config = Config()
        self.db_ops = DBOperations(self.config.DB_CONFIG)
        self.face_detector = FaceDetector()
        self.face_recognizer = FaceRecognizer()
        self.camera_id = camera_id
        self.window_name = f"{self.config.WINDOW_TITLE} - Saida"
        self.last_access = {}
        self.show_info = True

        if not self.db_ops.connect():
            print("❌ Falha critica: Nao foi possível conectar ao banco de dados")
            sys.exit(1)

        if not self.load_known_faces():
            print("⚠️ Atencao: Nenhum rosto conhecido carregado")

        print("✅ Sistema inicializado com sucesso")

    def load_known_faces(self):
        """Carrega rostos conhecidos do banco de dados"""
        try:
            known_faces_data = self.db_ops.get_all_known_faces()
            if known_faces_data:
                self.face_recognizer.load_known_faces(known_faces_data)
                print(f"✅ {len(known_faces_data)} rostos carregados para saida")
                return True
            return False
        except Exception as e:
            print(f"❌ Erro ao carregar rostos: {e}")
            return False

    def run(self):
        """Executa o loop principal da camera de saida"""
        print("🔍 Tentando abrir a camera...")
        cap = cv2.VideoCapture(self.camera_id)

        if not cap.isOpened():
            print(
                f"❌ Falha crítica: Nao foi possível acessar a camera (ID: {self.camera_id})"
            )
            print("👉 Solucao: Verifique se a camera esta conectada e disponivel")
            sys.exit(1)

        print("\n✅ Câmera de saída iniciada. Monitorando saídas...")
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, *self.config.DEFAULT_WINDOW_SIZE)

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("⚠️ Erro ao capturar frame")
                    break

                # Redimensiona o frame para se adaptar à tela
                frame = self._resize_frame(frame)

                # Processa o frame
                faces = self.face_detector.detect_faces(frame)
                results = self.face_recognizer.recognize_faces(frame, faces)

                # Registra saídas
                self._register_accesses(results)

                # Adiciona UI
                frame = self._add_ui_elements(frame, results)

                # Exibe o frame
                cv2.imshow(self.window_name, frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                elif key == ord("i"):
                    self.show_info = not self.show_info

        finally:
            cap.release()
            cv2.destroyAllWindows()
            self.db_ops.close()
            print("✅ Camera de saida encerrada")

    def _resize_frame(self, frame):
        """Redimensiona o frame para se adaptar a tela"""
        screen_width = 1920  # Ajuste conforme necessário
        screen_height = 1080
        h, w = frame.shape[:2]

        # Calcula a proporção de redimensionamento
        scale = min(screen_width / w, screen_height / h)

        # Redimensiona mantendo a proporção
        if scale < 1:
            frame = cv2.resize(
                frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA
            )

        return frame

    def _register_accesses(self, results):
        """Registra saidas validas"""
        for result in results:
            if (
                result["user_id"]
                and result["confidence"] > self.config.FACE_RECOGNITION_THRESHOLD
            ):

                user_id = result["user_id"]
                current_time = time.time()

                if (
                    user_id not in self.last_access
                    or (current_time - self.last_access[user_id])
                    > self.config.MIN_ACCESS_INTERVAL
                ):

                    if self._register_access(result, "saida"):
                        self.last_access[user_id] = current_time
                        print(f"👤 Saida registrada: {result['name']} (ID: {user_id})")

    def _register_access(self, result, direction):
        try:
            confidence_percent = min(round(result["confidence"] * 100, 2), 100)
            return self.db_ops.register_access(
                user_id=result["user_id"],
                name=result["name"],
                direction=direction,
                confidence=confidence_percent,
            )
        except Exception as e:
            print(f"❌ Erro ao registrar saida: {e}")
            return False

    def _add_ui_elements(self, frame, results):
        """Adiciona elementos de interface ao frame"""
        h, w = frame.shape[:2]

        # Borda decorativa (cor diferente para saída)
        frame = cv2.copyMakeBorder(
            frame,
            self.config.BORDER_SIZE,
            self.config.BORDER_SIZE,
            self.config.BORDER_SIZE,
            self.config.BORDER_SIZE,
            cv2.BORDER_CONSTANT,
            value=self.config.COLORS["secondary"],
        )

        # Informações na tela
        if self.show_info:
            info_text = [
                f"Pessoas detectadas: {len(results)}",
                f"Modo: Saida",
                "Pressione 'i' para ocultar informacoes",
                "Pressione 'q' para voltar ao menu",
            ]

            for i, text in enumerate(info_text):
                y_pos = 30 + i * 30
                cv2.putText(
                    frame,
                    text,
                    (10, y_pos),
                    self.config.FONT,
                    0.6,
                    self.config.COLORS["light"],
                    1,
                    cv2.LINE_AA,
                )

        # Desenha detecções
        for result in results:
            (x1, y1, x2, y2) = result["location"]
            name = result["name"]
            confidence = result["confidence"]

            # Ajusta coordenadas para a frame com borda
            x1 += self.config.BORDER_SIZE
            y1 += self.config.BORDER_SIZE
            x2 += self.config.BORDER_SIZE
            y2 += self.config.BORDER_SIZE

            color = (
                self.config.COLORS["success"]
                if name != "Desconhecido"
                else self.config.COLORS["danger"]
            )

            # Desenha retângulo e fundo do texto
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.rectangle(frame, (x1, y1 - 30), (x2, y1), color, -1)

            # Texto da etiqueta
            label = f"{name} ({confidence:.2f})"
            cv2.putText(
                frame,
                label,
                (x1, y1 - 10),
                self.config.FONT,
                self.config.FONT_SCALE,
                self.config.COLORS["light"],
                self.config.FONT_THICKNESS,
                cv2.LINE_AA,
            )

        return frame


if __name__ == "__main__":
    print("=" * 50)
    print("SISTEMA DE RECONHECIMENTO FACIAL - SAIDA")
    print("=" * 50)

    try:
        camera = CameraSaida(camera_id=0)
        camera.run()
    except Exception as e:
        print(f"❌ Erro fatal: {str(e)}")
        sys.exit(1)
