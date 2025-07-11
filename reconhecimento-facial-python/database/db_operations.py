import mysql.connector
from mysql.connector import Error


class Person:
    def __init__(self, id, name, folder, danger_level):
        self.id = id
        self.name = name
        self.folder = folder
        self.danger_level = danger_level

    def __repr__(self):
        return f"Person(ID={self.id}, Name='{self.name}', Danger='{self.danger_level}', Folder='{self.folder}')"


class DBOperations:
    def __init__(self, db_config):
        self.db_config = db_config
        self.connection = None

    def connect(self):
        try:
            self.connection = mysql.connector.connect(**self.db_config)
            if self.connection.is_connected():
                print("✅ Conexão com o banco de dados estabelecida")
                return True
        except Error as e:
            print(f"❌ Erro de conexão: {e}")
        return False

    def get_all_known_faces(self):
        try:
            cursor = self.connection.cursor(dictionary=True)
            cursor.execute("SELECT id, nome, pasta, nivel_perigo FROM Pessoas")
            pessoas = cursor.fetchall()
            cursor.close()

            if not pessoas:
                print("⚠️ Nenhuma pessoa cadastrada no banco de dados")
                return None

            return [
                Person(p["id"], p["nome"], p["pasta"], p["nivel_perigo"])
                for p in pessoas
            ]
        except Exception as e:
            print(f"❌ Erro ao carregar rostos: {e}")
            return None

    def register_access(self, user_id, name, direction, confidence, danger_level):
        try:
            cursor = self.connection.cursor()

            # Registra na tabela Reconhecimentos
            cursor.execute(
                """
                INSERT INTO Reconhecimentos 
                (pessoa_id, nivel_confianca, nivel_perigo) 
                VALUES (%s, %s, %s)
                """,
                (user_id, confidence, danger_level),
            )

            # Registra na tabela Logs
            cursor.execute(
                """
                INSERT INTO Logs 
                (acao, detalhes) 
                VALUES (%s, %s)
                """,
                (
                    "reconhecimento",
                    f"Reconhecimento {direction} - {name} (ID: {user_id}) - Perigo: {danger_level} - Confiança: {confidence}%",
                ),
            )

            self.connection.commit()
            cursor.close()
            print(
                f"✅ Reconhecimento registrado: {direction.upper()} - {name} (ID: {user_id})"
            )
            return True
        except Error as e:
            print(f"❌ Erro ao registrar reconhecimento: {e}")
            return False

    def close(self):
        if self.connection and self.connection.is_connected():
            self.connection.close()
            print("✅ Conexão com o banco de dados encerrada")
