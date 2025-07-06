<?php
function criarTabelas($pdo)
{
    try {
        // Tabela Pessoas com campos adicionais
        $pdo->exec("
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
        ");

        // Tabela Acessos com logs detalhados
        $pdo->exec("
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
        ");

        // Tabela de Logs de Segurança
        $pdo->exec("
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
        ");

    } catch (\PDOException $e) {
        die("Erro ao criar tabelas: " . $e->getMessage());
    }
}