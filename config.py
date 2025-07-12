import os
import mysql.connector
from mysql.connector import Error


class Config:
    # Configurações do banco de dados
    DB_HOST = "localhost"
    DB_USER = "root"
    DB_PASSWORD = ""
    DB_NAME = "reconhecimento_facial"

    # Configurações do servidor
    SECRET_KEY = "segredo-muito-seguro"
    UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), "uploads", "faces")
    TEMP_FOLDER = os.path.join(os.path.dirname(__file__), "uploads", "temp")
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB

    @classmethod
    def ensure_database_exists(cls):
        """Garante que o banco de dados existe antes de qualquer conexão"""
        try:
            # Primeiro conecta sem especificar o banco
            conn = mysql.connector.connect(
                host=cls.DB_HOST, user=cls.DB_USER, password=cls.DB_PASSWORD
            )
            cursor = conn.cursor()

            # Cria o banco se não existir
            cursor.execute(f"CREATE DATABASE IF NOT EXISTS {cls.DB_NAME}")
            print(f"✅ Banco de dados '{cls.DB_NAME}' verificado/criado")

            cursor.close()
            conn.close()
            return True

        except Error as e:
            print(f"❌ Falha crítica ao criar banco de dados: {e}")
            return False

    @staticmethod
    def init_app(app):
        """Configuração do aplicativo Flask"""
        os.makedirs(Config.UPLOAD_FOLDER, exist_ok=True)
        os.makedirs(Config.TEMP_FOLDER, exist_ok=True)
