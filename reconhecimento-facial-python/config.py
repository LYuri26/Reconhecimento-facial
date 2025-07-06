import os
import cv2


class Config:
    def __init__(self):
        # Configuração do banco de dados
        self.DB_CONFIG = {
            "host": "localhost",
            "database": "Catraca",
            "user": "root",
            "password": "",
            "port": 3306,
        }

        # Configurações de reconhecimento facial
        self.FACE_RECOGNITION_THRESHOLD = 0.6
        self.FACE_IMAGE_SIZE = (100, 100)

        # Configurações das câmeras
        self.SHOW_PREVIEW = True
        self.MIN_ACCESS_INTERVAL = 30  # Segundos entre acessos do mesmo usuário
        self.BORDER_SIZE = 5  # Tamanho da borda decorativa
        self.INFO_DISPLAY_TIME = 5  # Segundos que as informações são exibidas

        # Configuração de interface
        self.COLORS = {
            "primary": (0, 165, 255),  # Laranja (entrada)
            "secondary": (0, 215, 255),  # Dourado (saída)
            "dark": (0, 0, 0),  # Preto
            "light": (255, 255, 255),  # Branco
            "success": (0, 255, 0),  # Verde para reconhecido
            "danger": (0, 0, 255),  # Vermelho para desconhecido
            "info_bg": (50, 50, 50),  # Fundo das informações
        }

        self.FONT = cv2.FONT_HERSHEY_SIMPLEX
        self.FONT_SCALE = 0.7
        self.FONT_THICKNESS = 2
        self.WINDOW_TITLE = "Sistema de Catraca Inteligente"
        self.DEFAULT_WINDOW_SIZE = (800, 600)  # Largura, Altura
        self.WINDOW_SIZE = self.DEFAULT_WINDOW_SIZE  # Mantido para compatibilidade

        # Configuração do caminho base
        self.BASE_DIR = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "cadastro-web-php")
        )

    def get_full_path(self, relative_path):
        """Converte caminho relativo para absoluto"""
        if relative_path.startswith("imagens/"):
            relative_path = relative_path[8:]
        return os.path.join(self.BASE_DIR, "imagens", relative_path)

    def get_window_center(self, frame_width, frame_height):
        """Calcula a posição central para a janela"""
        screen_width = 1920  # Ajuste conforme a resolução do monitor
        screen_height = 1080
        x = (screen_width - frame_width) // 2
        y = (screen_height - frame_height) // 2
        return x, y
