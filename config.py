import os
from pathlib import Path

# Diretório base do projeto
BASE_DIR = Path(__file__).parent

# Configurações do banco de dados MySQL
# Configurações do banco de dados MySQL
DB_CONFIG = {
    "host": "localhost",
    "user": "root",
    "password": "",
    "database": "reconhecimento_facial",
    "raise_on_warnings": True,
}

# Configurações do modelo de treinamento
MODEL_CONFIG = {
    "model_path": str(BASE_DIR / "face_recognition/models/face_model_v2.dat"),
    "min_images_per_person": 5,
    "face_detection_method": "hog",  # "hog" (mais rápido) ou "cnn" (mais preciso)
    "num_jitters": 5,  # Número de variações para aumentar robustez
    "encoding_model": "small",  # "small" (rápido) ou "large" (mais preciso)
}

# Configurações de reconhecimento
RECOGNITION_SETTINGS = {
    "recognition_threshold": 0.6,  # Limiar de confiança (0-1)
    "min_face_size": 100,  # Tamanho mínimo da face em pixels
    "max_frame_skip": 5,  # Quadros para pular durante processamento de vídeo
}

# Configurações de segurança
SECURITY_SETTINGS = {
    "enable_alarm": True,
    "alarm_sound": str(BASE_DIR / "assets/sounds/alarm.wav"),
    "enable_screen_lock": True,
    "unknown_face_timeout": 30,  # Segundos antes de ações de segurança
}
