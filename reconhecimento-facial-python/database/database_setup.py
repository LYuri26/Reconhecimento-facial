import mysql.connector
from mysql.connector import Error, errorcode
from config import Config
from datetime import datetime


class DatabaseSetup:
    def __init__(self):
        self.config = Config()
        self.db_config = self.config.DB_CONFIG
        self.connection = None

    def setup_database(self):
        """Configura o banco de dados e tabelas com estrutura aprimorada"""
        try:
            # Conectar sem especificar o banco de dados
            temp_conn = mysql.connector.connect(
                user=self.db_config["user"],
                password=self.db_config["password"],
                host=self.db_config["host"],
                port=self.db_config["port"],
            )
            cursor = temp_conn.cursor()

            # Criar banco de dados se não existir
            self._create_database(cursor)
            cursor.close()
            temp_conn.close()

            # Conectar ao banco específico e criar tabelas
            self.connection = mysql.connector.connect(**self.db_config)
            self._create_tables()
            print("✅ Banco de dados e tabelas configurados com sucesso")
            return True

        except Error as e:
            print(f"❌ Erro ao configurar banco de dados: {e}")
            return False

    def _create_database(self, cursor):
        """Cria o banco de dados se não existir"""
        try:
            cursor.execute(
                f"CREATE DATABASE IF NOT EXISTS {self.db_config['database']} "
                f"CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci"
            )
            print(f"✅ Banco de dados {self.db_config['database']} verificado/criado")
        except Error as e:
            print(f"❌ Erro ao criar banco de dados: {e}")
            raise

    def _create_tables(self):
        """Cria as tabelas com estrutura aprimorada para controle de acesso"""
        try:
            cursor = self.connection.cursor()

            # Tabela Pessoas (com campos adicionais para segurança)
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS `Pessoas` (
                    `ID` int NOT NULL AUTO_INCREMENT,
                    `Nome` varchar(100) NOT NULL,
                    `cpf` varchar(14) NOT NULL,
                    `Email` varchar(200) NOT NULL,
                    `Telefone` varchar(15) NOT NULL,
                    `Pasta` varchar(255) DEFAULT NULL COMMENT 'Caminho para as imagens de referência',
                    `Ativo` tinyint(1) DEFAULT 1 COMMENT '0 = Inativo, 1 = Ativo',
                    `DataCadastro` TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    `UltimaAtualizacao` TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                    `CriadoPor` varchar(50) DEFAULT 'Sistema',
                    `AtualizadoPor` varchar(50) DEFAULT 'Sistema',
                    PRIMARY KEY (`ID`),
                    UNIQUE KEY `cpf_UNIQUE` (`cpf`),
                    KEY `idx_ativo` (`Ativo`)
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='Cadastro de pessoas autorizadas';
            """
            )

            # Tabela Acessos (com logs detalhados)
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS `Acessos` (
                    `id` int NOT NULL AUTO_INCREMENT,
                    `user_id` int NOT NULL,
                    `nome_pessoa` varchar(100) NOT NULL COMMENT 'Cópia do nome no momento do acesso',
                    `tipo_acesso` enum('entrada','saida') NOT NULL,
                    `data_hora` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    `confianca` decimal(5,2) DEFAULT NULL COMMENT 'Nível de confiança do reconhecimento (0-100)',
                    `metodo_autenticacao` enum('facial','cartao','manual') NOT NULL DEFAULT 'facial',
                    `ip_dispositivo` varchar(45) DEFAULT NULL,
                    `nome_dispositivo` varchar(100) DEFAULT NULL,
                    `foto_captura` varchar(255) DEFAULT NULL COMMENT 'Caminho para foto da captura',
                    `observacoes` text DEFAULT NULL,
                    PRIMARY KEY (`id`),
                    FOREIGN KEY (`user_id`) REFERENCES `Pessoas` (`ID`),
                    KEY `idx_data_hora` (`data_hora`),
                    KEY `idx_user_id` (`user_id`),
                    KEY `idx_tipo_acesso` (`tipo_acesso`)
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='Registro de acessos ao sistema';
            """
            )

            # Tabela de Logs de Segurança
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS `LogsSeguranca` (
                    `id` int NOT NULL AUTO_INCREMENT,
                    `data_hora` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    `tipo_evento` enum('acesso','tentativa_falha','configuracao','alerta') NOT NULL,
                    `descricao` text NOT NULL,
                    `user_id` int DEFAULT NULL COMMENT 'ID do usuário relacionado, se aplicável',
                    `ip_origem` varchar(45) DEFAULT NULL,
                    `dispositivo` varchar(255) DEFAULT NULL,
                    `severidade` enum('info','aviso','erro','critico') DEFAULT 'info',
                    PRIMARY KEY (`id`),
                    KEY `idx_data_hora` (`data_hora`),
                    KEY `idx_tipo_evento` (`tipo_evento`),
                    KEY `idx_severidade` (`severidade`)
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='Logs de segurança do sistema';
            """
            )

            self.connection.commit()
            cursor.close()
            print("✅ Tabelas de segurança criadas com sucesso")

        except Error as e:
            print(f"❌ Erro ao criar tabelas de segurança: {e}")
            raise

    def close(self):
        """Fecha a conexão com o banco de dados"""
        if self.connection and self.connection.is_connected():
            self.connection.close()
            print("✅ Conexão com o banco de dados encerrada")


if __name__ == "__main__":
    db_setup = DatabaseSetup()
    if db_setup.setup_database():
        db_setup.close()
