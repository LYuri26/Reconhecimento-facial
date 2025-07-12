import mysql.connector
from mysql.connector import Error
from datetime import datetime
import os
import numpy as np
from config import Config

try:
    from app.utils import train_classifier
except ImportError:
    from utils import train_classifier


class Database:
    def __init__(self):
        self.connection = self._create_connection()

    def _create_connection(self):
        """Estabelece conexão com o banco de dados"""
        try:
            conn = mysql.connector.connect(
                host=Config.DB_HOST,
                user=Config.DB_USER,
                password=Config.DB_PASSWORD,
                database=Config.DB_NAME,
            )
            return conn
        except Error as e:
            print(f"Erro ao conectar ao MySQL: {e}")
            raise

    def initialize(self):
        """Cria todas as tabelas necessárias"""
        cursor = self.connection.cursor()

        try:
            # Tabela de pessoas
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS pessoas (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    nome VARCHAR(100) NOT NULL,
                    sobrenome VARCHAR(100) NOT NULL,
                    observacoes TEXT,
                    data_cadastro DATETIME DEFAULT CURRENT_TIMESTAMP
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
                """
            )

            # Tabela de imagens
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS imagens (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    pessoa_id INT NOT NULL,
                    caminho VARCHAR(255) NOT NULL,
                    encodings TEXT,
                    data_cadastro DATETIME DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (pessoa_id) REFERENCES pessoas(id) ON DELETE CASCADE
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
                """
            )

            # Tabela de logs/reconhecimentos
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS logs (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    pessoa_id INT,
                    data_hora DATETIME DEFAULT CURRENT_TIMESTAMP,
                    confianca FLOAT,
                    FOREIGN KEY (pessoa_id) REFERENCES pessoas(id) ON DELETE SET NULL
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
                """
            )

            # Tabela de configurações do sistema
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS config (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    chave VARCHAR(50) UNIQUE NOT NULL,
                    valor TEXT,
                    data_atualizacao DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
                """
            )

            self.connection.commit()
            print("✅ Tabelas criadas com sucesso!")

            # Insere configurações padrão
            self._insert_default_config()

        except Error as e:
            print(f"❌ Erro ao criar tabelas: {e}")
            raise
        finally:
            cursor.close()

    def _insert_default_config(self):
        """Insere configurações padrão no banco de dados"""
        default_configs = {
            "limite_confianca": "80",
            "ultimo_treinamento": None,
            "modelo_primario": "LBPH",
        }

        cursor = self.connection.cursor()
        try:
            for chave, valor in default_configs.items():
                cursor.execute(
                    """
                    INSERT INTO config (chave, valor)
                    VALUES (%s, %s)
                    ON DUPLICATE KEY UPDATE valor = %s
                    """,
                    (chave, str(valor), str(valor)),
                )
            self.connection.commit()
        except Error as e:
            print(f"Erro ao inserir configurações padrão: {e}")
        finally:
            cursor.close()

    def train_classifiers(self):
        """Treina os classificadores com todas as imagens do banco"""
        cursor = self.connection.cursor(dictionary=True)

        try:
            cursor.execute(
                """
                SELECT i.pessoa_id, i.encodings 
                FROM imagens i
                JOIN pessoas p ON i.pessoa_id = p.id
                WHERE i.encodings IS NOT NULL
                """
            )

            images = []
            labels = []

            for row in cursor.fetchall():
                try:
                    encodings = np.fromstring(row["encodings"][1:-1], sep=" ")
                    images.append(encodings.reshape(220, 220))
                    labels.append(row["pessoa_id"])
                except Exception as e:
                    print(f"Erro ao processar encoding: {e}")
                    continue

            if not images:
                print("⚠️ Nenhuma imagem válida para treinamento")
                return False

            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            lbph_path = os.path.join(base_dir, "app/classifier/classificadorLBPH.yml")
            eigen_path = os.path.join(base_dir, "app/classifier/classificadorEigen.yml")

            os.makedirs(os.path.dirname(lbph_path), exist_ok=True)

            train_classifier(images, labels, lbph_path, "LBPH")
            train_classifier(images, labels, eigen_path, "Eigen")

            cursor.execute(
                "UPDATE config SET valor = NOW() WHERE chave = 'ultimo_treinamento'"
            )
            self.connection.commit()

            print("✅ Classificadores treinados com sucesso!")
            return True

        except Exception as e:
            print(f"❌ Erro durante o treinamento: {e}")
            return False
        finally:
            cursor.close()

    # Métodos para pessoas
    def inserir_pessoa(self, nome, sobrenome, observacoes=None):
        """Cadastra uma nova pessoa"""
        query = """
            INSERT INTO pessoas (nome, sobrenome, observacoes)
            VALUES (%s, %s, %s)
        """
        cursor = self.connection.cursor()
        try:
            cursor.execute(query, (nome, sobrenome, observacoes))
            self.connection.commit()
            return cursor.lastrowid
        except Error as e:
            print(f"Erro ao inserir pessoa: {e}")
            raise
        finally:
            cursor.close()

    def obter_pessoa(self, pessoa_id):
        """Obtém dados de uma pessoa específica"""
        query = """
            SELECT p.*, 
                   (SELECT COUNT(*) FROM imagens WHERE pessoa_id = p.id) as total_imagens,
                   (SELECT COUNT(*) FROM logs WHERE pessoa_id = p.id) as total_reconhecimentos
            FROM pessoas p
            WHERE p.id = %s
        """
        cursor = self.connection.cursor(dictionary=True)
        try:
            cursor.execute(query, (pessoa_id,))
            return cursor.fetchone()
        finally:
            cursor.close()

    def listar_pessoas(self, limit=100):
        """Lista todas as pessoas cadastradas"""
        query = """
            SELECT p.*, 
                   (SELECT COUNT(*) FROM imagens WHERE pessoa_id = p.id) as total_imagens,
                   (SELECT COUNT(*) FROM logs WHERE pessoa_id = p.id) as total_reconhecimentos
            FROM pessoas p
            ORDER BY p.data_cadastro DESC
            LIMIT %s
        """
        cursor = self.connection.cursor(dictionary=True)
        try:
            cursor.execute(query, (limit,))
            return cursor.fetchall()
        finally:
            cursor.close()

    def atualizar_pessoa(self, pessoa_id, nome, sobrenome, observacoes=None):
        """Atualiza dados de uma pessoa"""
        query = """
            UPDATE pessoas
            SET nome = %s, sobrenome = %s, observacoes = %s
            WHERE id = %s
        """
        cursor = self.connection.cursor()
        try:
            cursor.execute(query, (nome, sobrenome, observacoes, pessoa_id))
            self.connection.commit()
            return cursor.rowcount > 0
        except Error as e:
            print(f"Erro ao atualizar pessoa: {e}")
            raise
        finally:
            cursor.close()

    def deletar_pessoa(self, pessoa_id):
        """Remove uma pessoa e seus dados relacionados"""
        try:
            imagens = self.listar_imagens(pessoa_id)
            queries = [
                "DELETE FROM logs WHERE pessoa_id = %s",
                "DELETE FROM imagens WHERE pessoa_id = %s",
                "DELETE FROM pessoas WHERE id = %s",
            ]

            cursor = self.connection.cursor()
            for query in queries:
                cursor.execute(query, (pessoa_id,))
            self.connection.commit()

            for img in imagens:
                img_path = os.path.join(Config.UPLOAD_FOLDER, img["caminho"])
                if os.path.exists(img_path):
                    os.remove(img_path)

            return True
        except Exception as e:
            print(f"Erro ao deletar pessoa: {e}")
            self.connection.rollback()
            return False
        finally:
            cursor.close()

    # Métodos para imagens
    def inserir_imagem(self, pessoa_id, caminho_imagem, encoding_facial):
        """Cadastra uma nova imagem facial"""
        query = """
            INSERT INTO imagens (pessoa_id, caminho, encodings)
            VALUES (%s, %s, %s)
        """
        cursor = self.connection.cursor()
        try:
            cursor.execute(query, (pessoa_id, caminho_imagem, str(encoding_facial)))
            self.connection.commit()
            return True
        except Error as e:
            print(f"Erro ao inserir imagem: {e}")
            return False
        finally:
            cursor.close()

    def listar_imagens(self, pessoa_id):
        """Lista todas as imagens de uma pessoa"""
        query = """
            SELECT i.*, p.nome, p.sobrenome
            FROM imagens i
            JOIN pessoas p ON i.pessoa_id = p.id
            WHERE i.pessoa_id = %s
            ORDER BY i.data_cadastro DESC
        """
        cursor = self.connection.cursor(dictionary=True)
        try:
            cursor.execute(query, (pessoa_id,))
            return cursor.fetchall()
        except Error as e:
            print(f"Erro ao listar imagens: {e}")
            return []
        finally:
            cursor.close()

    def obter_imagem(self, imagem_id):
        """Obtém dados de uma imagem específica"""
        query = "SELECT * FROM imagens WHERE id = %s"
        cursor = self.connection.cursor(dictionary=True)
        try:
            cursor.execute(query, (imagem_id,))
            return cursor.fetchone()
        except Error as e:
            print(f"Erro ao obter imagem: {e}")
            return None
        finally:
            cursor.close()

    def deletar_imagem(self, imagem_id):
        """Remove uma imagem específica"""
        try:
            # Obtém o caminho da imagem antes de deletar
            query = "SELECT caminho FROM imagens WHERE id = %s"
            cursor = self.connection.cursor(dictionary=True)
            cursor.execute(query, (imagem_id,))
            imagem = cursor.fetchone()

            if not imagem:
                return False

            # Remove do banco de dados
            cursor.execute("DELETE FROM imagens WHERE id = %s", (imagem_id,))
            self.connection.commit()

            # Remove o arquivo físico
            img_path = os.path.join(Config.UPLOAD_FOLDER, imagem["caminho"])
            if os.path.exists(img_path):
                os.remove(img_path)

            return True
        except Exception as e:
            print(f"Erro ao deletar imagem: {e}")
            self.connection.rollback()
            return False
        finally:
            cursor.close()

    # Métodos para reconhecimento
    def obter_encodings(self, pessoa_id=None):
        """Obtém todos os encodings faciais cadastrados"""
        query = """
            SELECT i.pessoa_id, i.encodings, p.nome, p.sobrenome
            FROM imagens i
            JOIN pessoas p ON i.pessoa_id = p.id
        """
        params = ()

        if pessoa_id:
            query += " WHERE i.pessoa_id = %s"
            params = (pessoa_id,)

        cursor = self.connection.cursor(dictionary=True)
        try:
            cursor.execute(query, params)
            return cursor.fetchall()
        except Error as e:
            print(f"Erro ao obter encodings: {e}")
            return []
        finally:
            cursor.close()

    def registrar_reconhecimento(self, pessoa_id, confianca=None):
        """Registra uma tentativa de reconhecimento"""
        query = """
            INSERT INTO logs (pessoa_id, confianca)
            VALUES (%s, %s)
        """
        cursor = self.connection.cursor()
        try:
            cursor.execute(query, (pessoa_id, confianca))
            self.connection.commit()
            return cursor.lastrowid
        except Error as e:
            print(f"Erro ao registrar reconhecimento: {e}")
            raise
        finally:
            cursor.close()

    def listar_logs(self, limit=100, pessoa_id=None):
        """Lista os registros de reconhecimento"""
        query = """
            SELECT l.*, p.nome, p.sobrenome 
            FROM logs l
            LEFT JOIN pessoas p ON l.pessoa_id = p.id
            {where}
            ORDER BY l.data_hora DESC
            LIMIT %s
        """.format(
            where="WHERE l.pessoa_id = %s" if pessoa_id else ""
        )

        params = (pessoa_id, limit) if pessoa_id else (limit,)

        cursor = self.connection.cursor(dictionary=True)
        try:
            cursor.execute(query, params)
            return cursor.fetchall()
        finally:
            cursor.close()

    def contar_reconhecimentos(self, pessoa_id=None):
        """Conta o total de reconhecimentos"""
        query = "SELECT COUNT(*) as total FROM logs"
        params = ()

        if pessoa_id:
            query += " WHERE pessoa_id = %s"
            params = (pessoa_id,)

        cursor = self.connection.cursor(dictionary=True)
        try:
            cursor.execute(query, params)
            return cursor.fetchone()["total"]
        finally:
            cursor.close()

    def __del__(self):
        """Fecha a conexão quando o objeto é destruído"""
        if (
            hasattr(self, "connection")
            and self.connection
            and self.connection.is_connected()
        ):
            self.connection.close()
