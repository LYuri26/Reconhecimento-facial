-- Criação do banco de dados
CREATE DATABASE IF NOT EXISTS reconhecimento_facial CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;

-- Seleciona o banco de dados
USE reconhecimento_facial;

-- Tabela de cadastros
CREATE TABLE IF NOT EXISTS cadastros (
    id INT AUTO_INCREMENT PRIMARY KEY,
    nome VARCHAR(100) NOT NULL,
    sobrenome VARCHAR(100) NOT NULL,
    apelido VARCHAR(100),
    observacoes TEXT,
    data_cadastro DATETIME NOT NULL
) ENGINE = InnoDB DEFAULT CHARSET = utf8mb4 COLLATE = utf8mb4_unicode_ci;

-- Tabela de imagens originais
CREATE TABLE IF NOT EXISTS imagens_cadastro (
    id INT AUTO_INCREMENT PRIMARY KEY,
    cadastro_id INT NOT NULL,
    caminho_imagem VARCHAR(255) NOT NULL,
    data_upload DATETIME NOT NULL,
    CONSTRAINT fk_cadastro_id FOREIGN KEY (cadastro_id) REFERENCES cadastros(id) ON DELETE CASCADE
) ENGINE = InnoDB DEFAULT CHARSET = utf8mb4 COLLATE = utf8mb4_unicode_ci;

-- Nova tabela para treinamento
CREATE TABLE IF NOT EXISTS treinamentos (
    id INT AUTO_INCREMENT PRIMARY KEY,
    cadastro_id INT NOT NULL,
    nome VARCHAR(100) NOT NULL,
    sobrenome VARCHAR(100) NOT NULL,
    caminho_imagem VARCHAR(255) NOT NULL,
    data_processamento DATETIME NOT NULL,
    CONSTRAINT fk_treinamento_cadastro_id FOREIGN KEY (cadastro_id) REFERENCES cadastros(id) ON DELETE CASCADE
) ENGINE = InnoDB DEFAULT CHARSET = utf8mb4 COLLATE = utf8mb4_unicode_ci;