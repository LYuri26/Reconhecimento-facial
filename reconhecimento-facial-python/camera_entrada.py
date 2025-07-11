import cv2
import time
import numpy as np
from database.db_operations import DBOperations
from face_recognition.detector import FaceDetector
from face_recognition.recognizer import FaceRecognizer
from config import Config


class CameraEntrada:
    def __init__(self, camera_id=0):
        self.config = Config()
        self.db_ops = DBOperations(self.config.DB_CONFIG)
        self.face_detector = FaceDetector()
        self.face_recognizer = FaceRecognizer()
        self.camera_id = camera_id
        self.window_name = f"{self.config.WINDOW_TITLE} - Reconhecimento"
        self.last_access = {}
        self.show_info = True
        self.unknown_faces = []  # Armazena faces desconhecidas para análise

        if not self.db_ops.connect():
            raise Exception("❌ Falha ao conectar no banco de dados")

        self.load_known_faces()

    def load_known_faces(self):
        """Carrega os rostos conhecidos com feedback visual"""
        print("⏳ Carregando rostos conhecidos...")
        start_time = time.time()

        try:
            known_faces_data = self.db_ops.get_all_known_faces()
            if known_faces_data:
                success = self.face_recognizer.load_known_faces(known_faces_data)
                if success:
                    elapsed = time.time() - start_time
                    print(
                        f"✅ {len(known_faces_data)} rostos carregados em {elapsed:.2f}s"
                    )
                else:
                    print("⚠️ Falha ao carregar rostos conhecidos")
            else:
                print("⚠️ Nenhum rosto cadastrado encontrado")
        except Exception as e:
            print(f"❌ Erro ao carregar rostos conhecidos: {str(e)}")

    def run(self):
        """Método principal com tratamento de erros aprimorado"""
        cap = cv2.VideoCapture(self.camera_id)
        if not cap.isOpened():
            print(f"❌ Não foi possível acessar a câmera (ID: {self.camera_id})")
            return

        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, *self.config.DEFAULT_WINDOW_SIZE)

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("⚠️ Erro ao capturar frame")
                    time.sleep(0.1)
                    continue

                frame = self._resize_frame(frame)
                faces = self.face_detector.detect_faces(frame)

                if faces:
                    results = self.face_recognizer.recognize_faces(frame, faces)
                    self._register_accesses(results)
                    frame = self._add_ui_elements(frame, results)
                else:
                    cv2.putText(
                        frame,
                        "Nenhuma face detectada",
                        (50, 50),
                        self.config.FONT,
                        1,
                        self.config.COLORS["warning"],
                        2,
                        cv2.LINE_AA,
                    )

                cv2.imshow(self.window_name, frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                elif key == ord("i"):
                    self.show_info = not self.show_info
                elif key == ord("s"):
                    self._save_current_state()
                elif key == ord("l"):
                    self._load_saved_state()

        except Exception as e:
            print(f"❌ Erro crítico: {str(e)}")
        finally:
            cap.release()
            cv2.destroyAllWindows()
            self.db_ops.close()
            print("✅ Câmera encerrada")

    def _resize_frame(self, frame):
        """Redimensiona o frame para se adequar à tela"""
        try:
            h, w = frame.shape[:2]
            scale = min(1920 / w, 1080 / h)
            if scale < 1:
                frame = cv2.resize(
                    frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA
                )
            return frame
        except Exception as e:
            print(f"⚠️ Erro ao redimensionar frame: {str(e)}")
            return frame

    def _register_accesses(self, results):
        """Registra os acessos no banco de dados"""
        for result in results:
            try:
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

                        if self._register_access(result, "entrada"):
                            self.last_access[user_id] = current_time
                            print(
                                f"👤 Registrado: {result['name']} (ID: {user_id}) - "
                                f"Confiança: {result['confidence']:.2f} - "
                                f"Perigo: {result['danger_level']}"
                            )
            except Exception as e:
                print(f"⚠️ Erro ao registrar acesso: {str(e)}")

    def _register_access(self, result, direction):
        """Registra um acesso individual"""
        try:
            confidence_percent = min(round(result["confidence"] * 100, 2), 100)
            return self.db_ops.register_access(
                user_id=result["user_id"],
                name=result["name"],
                direction=direction,
                confidence=confidence_percent,
                danger_level=result["danger_level"],
            )
        except Exception as e:
            print(f"❌ Erro ao registrar acesso: {e}")
            return False

    def _add_ui_elements(self, frame, results):
        """Adiciona elementos de interface ao frame com melhor desempenho"""
        try:
            h, w = frame.shape[:2]

            # Desenha borda colorida ao redor do frame
            frame = cv2.copyMakeBorder(
                frame,
                self.config.BORDER_SIZE,
                self.config.BORDER_SIZE,
                self.config.BORDER_SIZE,
                self.config.BORDER_SIZE,
                cv2.BORDER_CONSTANT,
                value=self.config.COLORS["primary"],
            )

            for result in results:
                (x1, y1, x2, y2) = result["location"]
                x1 += self.config.BORDER_SIZE
                y1 += self.config.BORDER_SIZE
                x2 += self.config.BORDER_SIZE
                y2 += self.config.BORDER_SIZE

                color = self._get_danger_color(result.get("danger_level", "BAIXO"))
                thickness = max(1, min(4, int(result.get("confidence", 0) * 4)))

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

                if result["name"] != "Desconhecido":
                    label = f"{result['name']} ({result.get('danger_level', 'BAIXO')})"
                    confidence = f"{result['confidence']*100:.1f}%"

                    (text_width, text_height), _ = cv2.getTextSize(
                        label, self.config.FONT, 0.6, 1
                    )

                    cv2.rectangle(
                        frame,
                        (x1, y1 - text_height - 10),
                        (x1 + text_width + 10, y1),
                        color,
                        -1,
                    )

                    cv2.putText(
                        frame,
                        label,
                        (x1 + 5, y1 - 5),
                        self.config.FONT,
                        0.6,
                        self.config.COLORS["light"],
                        1,
                        cv2.LINE_AA,
                    )

                    cv2.putText(
                        frame,
                        confidence,
                        (x1 + 5, y2 + 20),
                        self.config.FONT,
                        0.5,
                        color,
                        1,
                        cv2.LINE_AA,
                    )

            return frame
        except Exception as e:
            print(f"⚠️ Erro ao adicionar elementos de UI: {str(e)}")
            return frame

    def _get_danger_color(self, danger_level):
        """Retorna a cor correspondente ao nível de perigo"""
        try:
            danger_level = danger_level.upper()
            if danger_level == "ALTO":
                return self.config.COLORS["danger"]
            elif danger_level == "MEDIO":
                return self.config.COLORS["orange"]
            else:
                return self.config.COLORS["warning"]
        except:
            return self.config.COLORS["warning"]

    def _save_current_state(self):
        """Salva o estado atual do reconhecimento"""
        try:
            self.face_recognizer.save_model()
            print("💾 Estado do reconhecimento salvo com sucesso")
        except Exception as e:
            print(f"❌ Falha ao salvar estado: {str(e)}")

    def _load_saved_state(self):
        """Carrega um estado previamente salvo"""
        try:
            self.face_recognizer.load_model()
            print("♻️ Estado do reconhecimento carregado com sucesso")
        except Exception as e:
            print(f"❌ Falha ao carregar estado: {str(e)}")


if __name__ == "__main__":
    try:
        CameraEntrada().run()
    except Exception as e:
        print(f"❌ Erro fatal: {str(e)}")
        exit(1)
