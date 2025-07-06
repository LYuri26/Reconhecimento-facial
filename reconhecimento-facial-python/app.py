import cv2
import numpy as np
from camera_entrada import CameraEntrada
from camera_saida import CameraSaida
from config import Config


class App:
    def __init__(self):
        self.config = Config()
        self.window_name = "Menu Principal - Catraca Inteligente"
        self.options = [
            {"text": "1 - Câmera de Entrada", "action": self._start_entrada},
            {"text": "2 - Câmera de Saída", "action": self._start_saida},
            {"text": "Q - Sair", "action": self._quit},
        ]
        self.selected_index = 0

    def run(self):
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, *self.config.DEFAULT_WINDOW_SIZE)

        while True:
            # Cria o frame do menu
            frame = self._create_menu_frame()

            # Exibe o frame
            cv2.imshow(self.window_name, frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == 13:  # Enter
                self.options[self.selected_index]["action"]()
            elif key == ord("1"):
                self._start_entrada()
            elif key == ord("2"):
                self._start_saida()
            elif key == 82:  # Seta para cima
                self.selected_index = max(0, self.selected_index - 1)
            elif key == 84:  # Seta para baixo
                self.selected_index = min(
                    len(self.options) - 1, self.selected_index + 1
                )

        cv2.destroyAllWindows()

    def _create_menu_frame(self):
        """Cria o frame do menu com as opções"""
        frame = np.full((600, 800, 3), 240, dtype=np.uint8)  # Fundo cinza claro

        # Título
        cv2.putText(
            frame,
            "CATRACA INTELIGENTE",
            (250, 100),
            self.config.FONT,
            1.5,
            self.config.COLORS["dark"],
            self.config.FONT_THICKNESS,
            cv2.LINE_AA,
        )

        # Opções
        for i, option in enumerate(self.options):
            y_pos = 200 + i * 100
            color = (
                self.config.COLORS["primary"]
                if i == self.selected_index
                else self.config.COLORS["dark"]
            )
            cv2.putText(
                frame,
                option["text"],
                (300, y_pos),
                self.config.FONT,
                self.config.FONT_SCALE,
                color,
                self.config.FONT_THICKNESS,
                cv2.LINE_AA,
            )

        # Instruções
        cv2.putText(
            frame,
            "Use as setas para navegar e Enter para selecionar",
            (150, 500),
            self.config.FONT,
            0.6,
            self.config.COLORS["dark"],
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            "Ou pressione 1, 2 ou Q diretamente",
            (250, 530),
            self.config.FONT,
            0.6,
            self.config.COLORS["dark"],
            1,
            cv2.LINE_AA,
        )

        return frame

    def _start_entrada(self):
        cv2.destroyWindow(self.window_name)
        camera = CameraEntrada()
        camera.run()
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, *self.config.DEFAULT_WINDOW_SIZE)

    def _start_saida(self):
        cv2.destroyWindow(self.window_name)
        camera = CameraSaida()
        camera.run()
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, *self.config.DEFAULT_WINDOW_SIZE)

    def _quit(self):
        cv2.destroyAllWindows()
        exit()


if __name__ == "__main__":
    app = App()
    app.run()
