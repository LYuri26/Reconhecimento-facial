<?php
require_once 'config.php';

class Database
{
    private static $pdo = null;

    public static function getConnection()
    {
        if (self::$pdo === null) {
            try {
                // Primeiro conecta sem especificar o banco de dados para criá-lo se necessário
                $dsn = "mysql:host=" . DB_HOST . ";charset=utf8mb4";
                $options = [
                    PDO::ATTR_ERRMODE => PDO::ERRMODE_EXCEPTION,
                    PDO::ATTR_DEFAULT_FETCH_MODE => PDO::FETCH_ASSOC,
                    PDO::ATTR_EMULATE_PREPARES => false,
                ];

                self::$pdo = new PDO($dsn, DB_USER, DB_PASS, $options);

                // Verifica e cria o banco de dados se não existir
                self::createDatabase();

                // Agora conecta ao banco de dados específico
                $dsn = "mysql:host=" . DB_HOST . ";dbname=" . DB_NAME . ";charset=utf8mb4";
                self::$pdo = new PDO($dsn, DB_USER, DB_PASS, $options);

                // Cria as tabelas se não existirem
                self::createTables();
            } catch (PDOException $e) {
                die("Erro na conexão com o banco de dados: " . $e->getMessage());
            }
        }
        return self::$pdo;
    }

    private static function createDatabase()
    {
        try {
            // Verifica se o banco de dados já existe
            $stmt = self::$pdo->query("SELECT SCHEMA_NAME FROM INFORMATION_SCHEMA.SCHEMATA WHERE SCHEMA_NAME = '" . DB_NAME . "'");

            if ($stmt->rowCount() == 0) {
                // Cria o banco de dados
                self::$pdo->exec("CREATE DATABASE " . DB_NAME . " CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci");
                echo "Banco de dados criado com sucesso!<br>";
            }

            // Seleciona o banco de dados
            self::$pdo->exec("USE " . DB_NAME);
        } catch (PDOException $e) {
            die("Erro ao criar o banco de dados: " . $e->getMessage());
        }
    }

    private static function createTables()
    {
        global $database_structure;

        try {
            foreach ($database_structure as $table => $structure) {
                // Verifica se a tabela já existe
                $stmt = self::$pdo->query("SHOW TABLES LIKE '$table'");

                if ($stmt->rowCount() == 0) {
                    // Monta a query de criação da tabela
                    $columns = [];
                    foreach ($structure['columns'] as $name => $definition) {
                        $columns[] = "$name $definition";
                    }

                    $query = "CREATE TABLE $table (" . implode(', ', $columns);

                    // Adiciona chaves estrangeiras se existirem
                    if (isset($structure['foreign_keys'])) {
                        foreach ($structure['foreign_keys'] as $column => $fk) {
                            $query .= ", FOREIGN KEY ($column) REFERENCES {$fk['references']}";
                            if (isset($fk['on_delete'])) {
                                $query .= " ON DELETE {$fk['on_delete']}";
                            }
                        }
                    }

                    $query .= ") {$structure['options']}";

                    // Executa a criação da tabela
                    self::$pdo->exec($query);
                    echo "Tabela $table criada com sucesso!<br>";
                }
            }
        } catch (PDOException $e) {
            die("Erro ao criar tabelas: " . $e->getMessage());
        }
    }

    // Método para fechar a conexão (útil para testes)
    public static function closeConnection()
    {
        self::$pdo = null;
    }
}

// Obtém a conexão PDO
$pdo = Database::getConnection();
