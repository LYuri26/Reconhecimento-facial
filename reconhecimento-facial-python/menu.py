import cv2
import numpy as np
from camera_entrada import CameraEntrada
from camera_saida import CameraSaida
from config import Config


class MenuPrincipal:
    def __init__(self):
        self.config = Config()
        self.window_name = "Menu Principal - Indentificacao Inteligente"
        self.options = [
            {"text": "1 - Modulo de Entrada", "action": self._start_entrada},
            {"text": "2 - Modulo de Saída", "action": self._start_saida},
            {"text": "Q - Encerrar Sistema", "action": self._quit},
        ]
        self.selected_index = 0

    def run(self):
        """Metodo principal para execucao do menu."""
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, *self.config.DEFAULT_WINDOW_SIZE)

        # Configuracao adicional para suporte a Unicode
        self._configure_unicode_support()

        while True:
            frame = self._create_menu_frame()
            cv2.imshow(self.window_name, frame)

            key = cv2.waitKey(1) & 0xFF
            self._handle_key_input(key)

        cv2.destroyAllWindows()

    def _configure_unicode_support(self):
        """Configuracoes adicionais para suporte a caracteres especiais."""
        # Forca o uso de uma fonte que suporte caracteres Unicode
        try:
            self.font_path = "arial.ttf"  # Tente usar fonte Arial
            self.font = cv2.FONT_HERSHEY_COMPLEX  # Alternativa mais compatível
        except:
            self.font = cv2.FONT_HERSHEY_SIMPLEX  # Fallback padrao

    def _create_menu_frame(self):
        """Renderiza o frame do menu com estilizacao."""
        frame = np.full((600, 800, 3), 240, dtype=np.uint8)  # Fundo cinza claro

        # Título principal
        self._put_text_centered(
            frame,
            "CONTROLE DE ACESSO POR RECONHECIMENTO FACIAL",
            y=100,
            font_scale=0.9,
            thickness=2,
        )

        # Opcoes do menu
        for i, option in enumerate(self.options):
            color = (
                self.config.COLORS["primary"]
                if i == self.selected_index
                else self.config.COLORS["dark"]
            )
            self._put_text_centered(
                frame, option["text"], y=200 + i * 100, color=color, thickness=2
            )

        # Instrucoes de uso
        self._put_text_centered(
            frame, "Navegacao: Teclas ↑/↓ ou números 1/2", y=500, font_scale=0.6
        )
        self._put_text_centered(
            frame, "Selecao: Tecla Enter | Saída: Tecla Q", y=530, font_scale=0.6
        )

        return frame

    def _put_text_centered(
        self, frame, text, y, font_scale=0.7, thickness=1, color=None
    ):
        """Metodo auxiliar para colocar texto centralizado com suporte a Unicode."""
        if color is None:
            color = self.config.COLORS["dark"]

        # Calcula a largura do texto
        text_size = cv2.getTextSize(text, self.font, font_scale, thickness)[0]

        # Calcula a posicao X para centralizar
        text_x = (frame.shape[1] - text_size[0]) // 2

        # Desenha o texto
        cv2.putText(
            frame,
            text,
            (text_x, y),
            self.font,
            font_scale,
            color,
            thickness,
            cv2.LINE_AA,
        )

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
        """Inicia o Modulo de entrada."""
        cv2.destroyWindow(self.window_name)
        CameraEntrada().run()
        self._reopen_menu()

    def _start_saida(self):
        """Inicia o Modulo de saída."""
        cv2.destroyWindow(self.window_name)
        CameraSaida().run()
        self._reopen_menu()

    def _reopen_menu(self):
        """Reabre o menu apos encerrar um Modulo."""
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, *self.config.DEFAULT_WINDOW_SIZE)

    def _quit(self):
        """Encerra o aplicativo com codigo de saída 0."""
        cv2.destroyAllWindows()
        exit(0)


if __name__ == "__main__":
    # Configura o encoding para UTF-8
    import sys
    import io

    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

    MenuPrincipal().run()
