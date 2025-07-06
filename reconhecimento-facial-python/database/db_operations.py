import mysql.connector
from mysql.connector import Error
from .models import Person


class DBOperations:
    def __init__(self, db_config):
        self.db_config = db_config
        self.connection = None

    def connect(self):
        """Estabelece conexão com o banco de dados"""
        try:
            self.connection = mysql.connector.connect(**self.db_config)
            if self.connection.is_connected():
                print("✅ Conexão com o banco de dados estabelecida")
                return True
        except Error as e:
            print(f"❌ Erro de conexão: {e}")
        return False

    def get_all_known_faces(self):
        """Obtém todos os rostos conhecidos do banco de dados"""
        try:
            cursor = self.connection.cursor(dictionary=True)
            cursor.execute(
                "SELECT ID, Nome, Pasta FROM Pessoas WHERE Pasta IS NOT NULL AND Ativo = 1"
            )
            pessoas = cursor.fetchall()
            cursor.close()

            if not pessoas:
                print("⚠️ Nenhuma pessoa ativa cadastrada no banco de dados")
                return None

            return pessoas
        except Exception as e:
            print(f"❌ Erro ao carregar rostos: {e}")
            return None

    def register_access(self, user_id, name, direction, confidence):
        """Registra um acesso no banco de dados"""
        try:
            cursor = self.connection.cursor()

            # Registra na tabela Acessos
            cursor.execute(
                """
                INSERT INTO Acessos 
                (user_id, nome_pessoa, tipo_acesso, confianca, metodo_autenticacao) 
                VALUES (%s, %s, %s, %s, %s)
                """,
                (user_id, name, direction, confidence, "facial"),
            )

            # Registra na tabela LogsSeguranca
            cursor.execute(
                """
                INSERT INTO LogsSeguranca 
                (tipo_evento, descricao, user_id, severidade) 
                VALUES (%s, %s, %s, %s)
                """,
                (
                    "acesso",
                    f"Acesso de {direction} registrado para {name}",
                    user_id,
                    "info",
                ),
            )

            self.connection.commit()
            cursor.close()
            print(
                f"✅ Acesso registrado: {direction.upper()} - {name} (Confiança: {confidence:.2f}%)"
            )
            return True
        except Error as e:
            print(f"❌ Erro ao registrar acesso: {e}")
            return False

    def close(self):
        """Fecha a conexão com o banco de dados"""
        if self.connection and self.connection.is_connected():
            self.connection.close()
            print("✅ Conexão com o banco de dados encerrada")
