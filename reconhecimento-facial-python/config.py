class Config:
    def __init__(self):
        # Configurações do Banco de Dados
        self.DB_CONFIG = {
            "host": "localhost",
            "user": "root",
            "password": "",
            "database": "indentificacao",
            "port": 3306,
        }

        # Configurações de Interface
        self.WINDOW_TITLE = "Reconhecimento Facial"
        self.DEFAULT_WINDOW_SIZE = (800, 600)
        self.BORDER_SIZE = 10
        self.COLORS = {
            "primary": (0, 119, 200),
            "secondary": (0, 180, 216),
            "success": (0, 200, 83),
            "warning": (0, 200, 200),
            "orange": (0, 165, 255),
            "danger": (0, 0, 255),
            "dark": (50, 50, 50),
            "light": (240, 240, 240),
        }

        # Configurações de Fonte
        self._font = None
        self.FONT_SCALE = 0.7
        self.FONT_THICKNESS = 2

        # Configurações de Detecção
        self.FACE_IMAGE_SIZE = (160, 160)  # Tamanho ideal para Facenet
        self.MIN_FACE_SIZE = (30, 30)  # Tamanho mínimo para detecção
        self.DETECTION_CONFIDENCE = 0.85  # Limiar mais rigoroso

        # Configurações de Reconhecimento
        self.RECOGNITION_MODEL = "hog"  # ou "cnn" se tiver GPU poderosa
        self.FACE_RECOGNITION_THRESHOLD = 0.6
        self.MIN_ACCESS_INTERVAL = 5

        # Caminhos para Modelos
        self.DNN_MODEL_PATH = "models/res10_300x300_ssd_iter_140000_fp16.caffemodel"
        self.DNN_CONFIG_PATH = "models/deploy.prototxt"

    @property
    def FONT(self):
        if self._font is None:
            import cv2

            self._font = cv2.FONT_HERSHEY_SIMPLEX
        return self._font

    @FONT.setter
    def FONT(self, value):
        self._font = value

    def get_full_path(self, folder):
        """Retorna o caminho completo para a pasta de armazenamento de rostos"""
        return f"../cadastro-web-php/{folder}"
