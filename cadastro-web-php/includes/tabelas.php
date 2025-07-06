<?php
function criarTabelas($pdo)
{
    try {
        // Tabela Pessoas simplificada
        $pdo->exec("
            CREATE TABLE IF NOT EXISTS `Pessoas` (
                `ID` int NOT NULL AUTO_INCREMENT,
                `Nome` varchar(100) NOT NULL,
                `cpf` varchar(14) NOT NULL,
                `Email` varchar(200) NOT NULL,
                `Telefone` varchar(15) NOT NULL,
                `Pasta` varchar(255) DEFAULT NULL,
                `Ativo` tinyint(1) DEFAULT 1,
                `DataCadastro` TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (`ID`),
                UNIQUE KEY `cpf_UNIQUE` (`cpf`),
                KEY `idx_ativo` (`Ativo`)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
        ");

        // Tabela Acessos simplificada
        $pdo->exec("
            CREATE TABLE IF NOT EXISTS `Acessos` (
                `id` int NOT NULL AUTO_INCREMENT,
                `user_id` int NOT NULL,
                `nome_pessoa` varchar(100) NOT NULL,
                `tipo_acesso` enum('entrada','saida') NOT NULL,
                `data_hora` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
                `confianca` decimal(5,2) DEFAULT NULL,
                PRIMARY KEY (`id`),
                FOREIGN KEY (`user_id`) REFERENCES `Pessoas` (`ID`),
                KEY `idx_data_hora` (`data_hora`),
                KEY `idx_user_id` (`user_id`)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
        ");

        // Tabela Logs simplificada
        $pdo->exec("
            CREATE TABLE IF NOT EXISTS `LogsSeguranca` (
                `id` int NOT NULL AUTO_INCREMENT,
                `data_hora` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
                `tipo_evento` enum('acesso','tentativa_falha') NOT NULL,
                `descricao` text NOT NULL,
                `user_id` int DEFAULT NULL,
                PRIMARY KEY (`id`),
                KEY `idx_data_hora` (`data_hora`)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
        ");
    } catch (\PDOException $e) {
        die("Erro ao criar tabelas: " . $e->getMessage());
    }
}
