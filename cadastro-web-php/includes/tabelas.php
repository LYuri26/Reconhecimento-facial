<?php
function criarTabelas($pdo)
{
    try {
        // Tabela Pessoas com campo para pasta
        $pdo->exec("
            CREATE TABLE IF NOT EXISTS `Pessoas` (
                `ID` int NOT NULL AUTO_INCREMENT,
                `Nome` varchar(50) NOT NULL,
                `cpf` varchar(14) NOT NULL,
                `Email` varchar(200) NOT NULL,
                `Telefone` varchar(15) NOT NULL,
                `Pasta` varchar(255) DEFAULT NULL,
                `Data` TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (`ID`),
                UNIQUE KEY `cpf_UNIQUE` (`cpf`)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
        ");

        // Tabela Acessos
        $pdo->exec("
            CREATE TABLE IF NOT EXISTS `Acessos` (
                `id` int NOT NULL AUTO_INCREMENT,
                `user_id` int NOT NULL,
                `direction` enum('entrada','saida') NOT NULL,
                `timestamp` TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (`id`),
                FOREIGN KEY (`user_id`) REFERENCES `Pessoas` (`ID`)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
        ");
    } catch (\PDOException $e) {
        die("Erro ao criar tabelas: " . $e->getMessage());
    }
}
