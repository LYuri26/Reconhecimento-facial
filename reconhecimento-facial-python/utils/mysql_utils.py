import mysql.connector
from mysql.connector import Error, errorcode


class MySQLUtils:
    def __init__(self):
        self.config = {
            "user": "root",
            "password": "",
            "host": "localhost",
            "database": "Catraca",
        }
        self.connection = None

    def connect(self):
        try:
            self.connection = mysql.connector.connect(**self.config)
            if self.connection.is_connected():
                print("Conexão ao MySQL estabelecida com sucesso")
                self.create_backup_tables()
            return self.connection
        except mysql.connector.Error as err:
            if err.errno == errorcode.ER_ACCESS_DENIED_ERROR:
                print("Erro de autenticação no banco de dados")
            elif err.errno == errorcode.ER_BAD_DB_ERROR:
                print("Database não existe, tentando criar...")
                self.create_database_and_tables()
            else:
                print(f"Erro ao conectar ao MySQL: {err}")
            return None

    def create_database_and_tables(self):
        try:
            # Conectar sem especificar o banco de dados
            temp_conn = mysql.connector.connect(
                user=self.config["user"],
                password=self.config["password"],
                host=self.config["host"],
            )
            cursor = temp_conn.cursor()

            # Criar banco de dados
            cursor.execute(
                f"CREATE DATABASE IF NOT EXISTS {self.config['database']} CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci"
            )
            print(f"Banco de dados {self.config['database']} criado com sucesso")

            cursor.close()
            temp_conn.close()

            # Reconectar ao banco específico e criar tabelas
            self.connection = mysql.connector.connect(**self.config)
            self.create_backup_tables()

        except Error as e:
            print(f"Erro ao criar banco de dados: {e}")

    def create_backup_tables(self):
        try:
            cursor = self.connection.cursor()

            # Tabela Pessoas (backup da tabela PHP)
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS `Pessoas` (
                    `ID` int NOT NULL AUTO_INCREMENT,
                    `Nome` varchar(50) NOT NULL,
                    `cpf` varchar(14) NOT NULL,
                    `Email` varchar(200) NOT NULL,
                    `Telefone` varchar(15) NOT NULL,
                    `Foto` varchar(255) DEFAULT NULL,
                    `Pasta` varchar(255) DEFAULT NULL,
                    `Data` TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (`ID`),
                    UNIQUE KEY `cpf_UNIQUE` (`cpf`)
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
            """
            )

            # Tabela Acessos (backup da tabela PHP)
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS `Acessos` (
                    `id` int NOT NULL AUTO_INCREMENT,
                    `user_id` int NOT NULL,
                    `direction` enum('entrada','saida') NOT NULL,
                    `timestamp` TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (`id`),
                    FOREIGN KEY (`user_id`) REFERENCES `Pessoas` (`ID`)
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
            """
            )

            self.connection.commit()
            cursor.close()
            print("Tabelas de backup criadas/verificadas com sucesso")
        except Error as e:
            print(f"Erro ao criar tabelas de backup: {e}")

    def get_user_by_face(self, face_encoding):
        cursor = self.connection.cursor(dictionary=True)
        query = "SELECT * FROM Pessoas WHERE Foto IS NOT NULL"
        cursor.execute(query)
        users = cursor.fetchall()
        cursor.close()
        return users

    def register_access(self, user_id, direction):
        try:
            cursor = self.connection.cursor()
            query = "INSERT INTO Acessos (user_id, direction) VALUES (%s, %s)"
            cursor.execute(query, (user_id, direction))
            self.connection.commit()
            cursor.close()
            return True
        except Error as e:
            print(f"Erro ao registrar acesso: {e}")
            return False

    def close(self):
        if self.connection and self.connection.is_connected():
            self.connection.close()
            print("Conexão ao MySQL encerrada")
