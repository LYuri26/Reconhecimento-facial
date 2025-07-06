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
            {"text": "1 - Modulo de Entrada", "action": self._start_entrada},
            {"text": "2 - Modulo de Saida", "action": self._start_saida},
            {"text": "Q - Encerrar Sistema", "action": self._quit},
        ]
        self.selected_index = 0

    def run(self):
        """Método principal para execucao do menu."""
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, *self.config.DEFAULT_WINDOW_SIZE)

        while True:
            frame = self._create_menu_frame()
            cv2.imshow(self.window_name, frame)

            key = cv2.waitKey(1) & 0xFF
            self._handle_key_input(key)

        cv2.destroyAllWindows()

    def _create_menu_frame(self):
        """Renderiza o frame do menu com estilizacao."""
        frame = np.full((600, 800, 3), 240, dtype=np.uint8)  # Fundo cinza claro

        # Titulo principal (ajustado para responsividade)
        cv2.putText(
            frame,
            "CONTROLE DE ACESSO POR RECONHECIMENTO FACIAL",
            (50, 80),  # Posicao mais à esquerda e acima
            self.config.FONT,
            0.7,  # Fonte menor
            self.config.COLORS["dark"],
            self.config.FONT_THICKNESS,
            cv2.LINE_AA,
        )

        # Opcoes do menu
        for i, option in enumerate(self.options):
            y_pos = 180 + i * 100
            color = (
                self.config.COLORS["primary"]
                if i == self.selected_index
                else self.config.COLORS["dark"]
            )
            cv2.putText(
                frame,
                option["text"],
                (250, y_pos),
                self.config.FONT,
                self.config.FONT_SCALE,
                color,
                self.config.FONT_THICKNESS,
                cv2.LINE_AA,
            )

        # Instrucões de uso
        cv2.putText(
            frame,
            "Navegacao: Teclas cima/baixo ou numeros 1/2",
            (180, 480),
            self.config.FONT,
            0.6,
            self.config.COLORS["dark"],
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            "Selecao: Tecla Enter | Saida: Tecla Q",
            (200, 510),
            self.config.FONT,
            0.6,
            self.config.COLORS["dark"],
            1,
            cv2.LINE_AA,
        )

        return frame

    def _handle_key_input(self, key):
        """Gerencia as entradas do teclado."""
        if key == ord("q"):
            self._quit()
        elif key == 13:  # Enter
            self.options[self.selected_index]["action"]()
        elif key == ord("1"):
            self._start_entrada()
        elif key == ord("2"):
            self._start_saida()
        elif key == 82:  # Seta para cima
            self.selected_index = max(0, self.selected_index - 1)
        elif key == 84:  # Seta para baixo
            self.selected_index = min(len(self.options) - 1, self.selected_index + 1)

    def _start_entrada(self):
        """Inicia o modulo de entrada."""
        cv2.destroyWindow(self.window_name)
        CameraEntrada().run()
        self._reopen_menu()

    def _start_saida(self):
        """Inicia o modulo de saida."""
        cv2.destroyWindow(self.window_name)
        CameraSaida().run()
        self._reopen_menu()

    def _reopen_menu(self):
        """Reabre o menu apos encerrar um modulo."""
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, *self.config.DEFAULT_WINDOW_SIZE)

    def _quit(self):
        """Encerra o aplicativo com codigo de saida 0."""
        cv2.destroyAllWindows()
        exit(0)


if __name__ == "__main__":
    App().run()
