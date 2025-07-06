import cv2
from camera_entrada import CameraEntrada
from camera_saida import CameraSaida
from config import Config


class MenuPrincipal:
    def __init__(self):
        self.config = Config()
        self.window_name = "Menu - Catraca Inteligente"
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = 1
        self.font_thickness = 2
        self.selected_option = 0
        self.options = ["1 - Câmera de Entrada", "2 - Câmera de Saída", "Q - Sair"]

    def run(self):
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, 800, 600)

        while True:
            # Fundo branco
            frame = self._create_menu_frame()

            # Exibe o frame
            cv2.imshow(self.window_name, frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("1"):
                self._start_camera_entrada()
            elif key == ord("2"):
                self._start_camera_saida()

        cv2.destroyAllWindows()

    def _create_menu_frame(self):
        """Cria o frame do menu com as opções"""
        frame = 255 * np.ones((600, 800, 3), dtype=np.uint8)  # Fundo branco

        # Título
        cv2.putText(
            frame,
            "CATRACA INTELIGENTE",
            (400 - 250, 100),
            self.font,
            1.5,
            (0, 0, 0),
            self.font_thickness,
            cv2.LINE_AA,
        )

        # Opções
        for i, option in enumerate(self.options):
            y_pos = 200 + i * 100
            color = (0, 165, 255) if i == self.selected_option else (0, 0, 0)
            cv2.putText(
                frame,
                option,
                (400 - 150, y_pos),
                self.font,
                self.font_scale,
                color,
                self.font_thickness,
                cv2.LINE_AA,
            )

        # Instruções
        cv2.putText(
            frame,
            "Selecione uma opção:",
            (50, 500),
            self.font,
            0.8,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )

        return frame

    def _start_camera_entrada(self):
        cv2.destroyWindow(self.window_name)
        camera = CameraEntrada()
        camera.run()
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, 800, 600)

    def _start_camera_saida(self):
        cv2.destroyWindow(self.window_name)
        camera = CameraSaida()
        camera.run()
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, 800, 600)


if __name__ == "__main__":
    import numpy as np

    menu = MenuPrincipal()
    menu.run()
