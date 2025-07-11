import os

os.environ["QT_QPA_PLATFORM"] = "xcb"  # Solução para problema do Wayland

import cv2
import numpy as np
import sys
import subprocess
from config import Config


def check_and_install_dependencies():
    """Verifica e instala as dependências necessárias"""
    required = {"numpy", "opencv-python", "mysql-connector-python"}
    try:
        import cv2
        import numpy
        import mysql.connector
    except ImportError:
        print("Instalando dependências necessárias...")
        try:
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", "-r", "requirements.txt"]
            )
            print("✅ Dependências instaladas com sucesso!")
        except subprocess.CalledProcessError as e:
            print(f"❌ Erro ao instalar dependências: {e}")
            sys.exit(1)


class App:
    def __init__(self):
        check_and_install_dependencies()
        self.config = Config()
        self.window_name = "Sistema de Reconhecimento Facial"
        self.options = [
            {"text": "1 - Iniciar Reconhecimento", "action": self._start_entrada},
            {"text": "Q - Encerrar Sistema", "action": self._quit},
        ]
        self.selected_index = 0

    def run(self):
        """Método principal para execução do menu"""
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, *self.config.DEFAULT_WINDOW_SIZE)

        while True:
            frame = self._create_menu_frame()
            cv2.imshow(self.window_name, frame)

            key = cv2.waitKey(1) & 0xFF
            self._handle_key_input(key)

    def _create_menu_frame(self):
        """Cria o frame do menu principal"""
        frame = np.full((600, 800, 3), 240, dtype=np.uint8)  # Fundo cinza claro

        # Título principal
        cv2.putText(
            frame,
            "RECONHECIMENTO FACIAL POR NÍVEL DE PERIGO",
            (80, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            self.config.COLORS["dark"],
            2,
            cv2.LINE_AA,
        )

        # Legenda de cores
        cv2.putText(
            frame,
            "LEGENDA:",
            (50, 180),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            self.config.COLORS["dark"],
            1,
            cv2.LINE_AA,
        )

        # Cores de nível de perigo
        colors = [
            ("BAIXO", self.config.COLORS["warning"]),
            ("MÉDIO", self.config.COLORS["orange"]),
            ("ALTO", self.config.COLORS["danger"]),
        ]

        for i, (text, color) in enumerate(colors):
            y_pos = 220 + i * 40
            cv2.rectangle(frame, (50, y_pos - 25), (80, y_pos - 5), color, -1)
            cv2.putText(
                frame,
                f"- {text}",
                (90, y_pos - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                self.config.COLORS["dark"],
                1,
                cv2.LINE_AA,
            )

        # Opções do menu
        for i, option in enumerate(self.options):
            y_pos = 350 + i * 60
            color = (
                self.config.COLORS["primary"]
                if i == self.selected_index
                else self.config.COLORS["dark"]
            )
            cv2.putText(
                frame,
                option["text"],
                (300, y_pos),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                color,
                2,
                cv2.LINE_AA,
            )

        # Instruções
        cv2.putText(
            frame,
            "Pressione ENTER para selecionar | Q para sair",
            (150, 500),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            self.config.COLORS["dark"],
            1,
            cv2.LINE_AA,
        )

        return frame

    def _handle_key_input(self, key):
        """Trata as entradas do teclado"""
        if key == ord("q"):
            self._quit()
        elif key == 13:  # Enter
            self.options[self.selected_index]["action"]()
        elif key == ord("1"):
            self._start_entrada()
        elif key == 82:  # Seta para cima
            self.selected_index = max(0, self.selected_index - 1)
        elif key == 84:  # Seta para baixo
            self.selected_index = min(len(self.options) - 1, self.selected_index + 1)

    def _start_entrada(self):
        """Inicia o módulo de reconhecimento facial"""
        cv2.destroyWindow(self.window_name)
        from camera_entrada import CameraEntrada

        CameraEntrada().run()
        self._reopen_menu()

    def _reopen_menu(self):
        """Reabre o menu principal"""
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, *self.config.DEFAULT_WINDOW_SIZE)

    def _quit(self):
        """Encerra o aplicativo"""
        cv2.destroyAllWindows()
        exit(0)


if __name__ == "__main__":
    App().run()
