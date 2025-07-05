import mysql.connector
from mysql.connector import errorcode


class MySQLUtils:
    def __init__(self):
        self.config = {
            "user": "root",
            "password": "123456",
            "host": "127.0.0.1",
            "database": "Formulario",
        }
        self.connection = None

    def connect(self):
        try:
            self.connection = mysql.connector.connect(**self.config)
            return self.connection
        except mysql.connector.Error as err:
            if err.errno == errorcode.ER_ACCESS_DENIED_ERROR:
                print("Erro de autenticação no banco de dados")
            elif err.errno == errorcode.ER_BAD_DB_ERROR:
                print("Database não existe")
            else:
                print(err)
            return None

    def get_user_by_face(self, face_encoding):
        cursor = self.connection.cursor(dictionary=True)
        query = "SELECT * FROM Pessoas WHERE Foto IS NOT NULL"
        cursor.execute(query)

        # Aqui você implementaria a comparação dos encodings faciais
        # Este é um placeholder - você precisará adaptar para sua lógica de reconhecimento
        for user in cursor:
            # Comparar face_encoding com o encoding armazenado
            pass

        cursor.close()
        return None

    def register_access(self, user_id, direction):
        cursor = self.connection.cursor()
        query = (
            "INSERT INTO Acessos (user_id, direction, timestamp) VALUES (%s, %s, NOW())"
        )
        cursor.execute(query, (user_id, direction))
        self.connection.commit()
        cursor.close()

    def close(self):
        if self.connection:
            self.connection.close()
