import mysql.connector
from mysql.connector import Error
from .config import DB_CONFIG


def get_db_connection():
    try:
        connection = mysql.connector.connect(**DB_CONFIG)
        return connection
    except Error as e:
        print(f"Erro ao conectar ao MySQL: {e}")
        return None


def create_database():
    try:
        # Conectar sem especificar o banco de dados
        conn = mysql.connector.connect(
            host=DB_CONFIG["host"],
            user=DB_CONFIG["user"],
            password=DB_CONFIG["password"],
        )
        cursor = conn.cursor()

        # Criar banco de dados se não existir
        cursor.execute(
            """
        CREATE DATABASE IF NOT EXISTS reconhecimento_facial 
        CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci
        """
        )

        # Usar o banco de dados
        cursor.execute("USE reconhecimento_facial")

        # Criar tabela de cadastros
        cursor.execute(
            """
        CREATE TABLE IF NOT EXISTS cadastros (
            id INT AUTO_INCREMENT PRIMARY KEY,
            nome VARCHAR(100) NOT NULL,
            sobrenome VARCHAR(100) NOT NULL,
            apelido VARCHAR(100),
            observacoes TEXT,
            data_cadastro DATETIME NOT NULL
        ) ENGINE = InnoDB DEFAULT CHARSET = utf8mb4 COLLATE = utf8mb4_unicode_ci
        """
        )

        # Criar tabela de imagens
        cursor.execute(
            """
        CREATE TABLE IF NOT EXISTS imagens_cadastro (
            id INT AUTO_INCREMENT PRIMARY KEY,
            cadastro_id INT NOT NULL,
            caminho_imagem VARCHAR(255) NOT NULL,
            data_upload DATETIME NOT NULL,
            CONSTRAINT fk_cadastro_id FOREIGN KEY (cadastro_id) 
            REFERENCES cadastros(id) ON DELETE CASCADE
        ) ENGINE = InnoDB DEFAULT CHARSET = utf8mb4 COLLATE = utf8mb4_unicode_ci
        """
        )

        conn.commit()
        print("Banco de dados e tabelas criados com sucesso.")

    except Error as e:
        print(f"Erro ao criar banco de dados: {e}")
    finally:
        if conn and conn.is_connected():
            cursor.close()
            conn.close()
