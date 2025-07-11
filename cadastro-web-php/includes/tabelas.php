<?php
function criarTabelas($pdo)
{
    try {
        // Tabela Pessoas
        $pdo->exec("
            CREATE TABLE IF NOT EXISTS `Pessoas` (
                `id` int NOT NULL AUTO_INCREMENT,
                `nome` varchar(100) NOT NULL,
                `sobrenome` varchar(100) NOT NULL,
                `apelido` varchar(50),
                `foto` varchar(255) NOT NULL,
                `pasta` varchar(255) NOT NULL,
                `nivel_perigo` enum('ALTO','MEDIO','BAIXO') NOT NULL DEFAULT 'BAIXO',
                `foragido` boolean NOT NULL DEFAULT false,
                `observacoes` text,
                `data_cadastro` TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (`id`),
                UNIQUE KEY `idx_nome_unico` (`nome`, `sobrenome`),
                KEY `idx_perigo` (`nivel_perigo`),
                KEY `idx_foragido` (`foragido`)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
        ");

        // Tabela de Logs (corrigido o nome)
        $pdo->exec("
            CREATE TABLE IF NOT EXISTS `Logs` (
                `id` int NOT NULL AUTO_INCREMENT,
                `data_hora` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
                `acao` varchar(50) NOT NULL,
                `detalhes` text,
                PRIMARY KEY (`id`),
                KEY `idx_data_hora` (`data_hora`)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
        ");

        // Tabela de Reconhecimentos
        $pdo->exec("
            CREATE TABLE IF NOT EXISTS `Reconhecimentos` (
                `id` int NOT NULL AUTO_INCREMENT,
                `pessoa_id` int NOT NULL,
                `data_hora` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
                `nivel_confianca` decimal(5,2) NOT NULL,
                `local` varchar(100),
                `observacoes` text,
                PRIMARY KEY (`id`),
                FOREIGN KEY (`pessoa_id`) REFERENCES `Pessoas` (`id`),
                KEY `idx_data_hora` (`data_hora`)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
        ");
    } catch (\PDOException $e) {
        die("Erro ao criar tabelas: " . $e->getMessage());
    }
}
