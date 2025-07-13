<?php

// Configurações do sistema de banco de dados
define('DB_HOST', 'localhost');
define('DB_USER', 'root');
define('DB_PASS', '');
define('DB_NAME', 'reconhecimento_facial');

// Configurações de tabelas
define('TABELA_CADASTROS', 'cadastros');
define('TABELA_IMAGENS', 'imagens_cadastro');
define('TABELA_TREINAMENTOS', 'treinamentos');

// Estrutura do banco de dados
$database_structure = [
    TABELA_CADASTROS => [
        'columns' => [
            'id' => 'INT AUTO_INCREMENT PRIMARY KEY',
            'nome' => 'VARCHAR(100) NOT NULL',
            'sobrenome' => 'VARCHAR(100) NOT NULL',
            'apelido' => 'VARCHAR(100)',
            'observacoes' => 'TEXT',
            'data_cadastro' => 'DATETIME NOT NULL'
        ],
        'options' => 'ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci'
    ],
    TABELA_IMAGENS => [
        'columns' => [
            'id' => 'INT AUTO_INCREMENT PRIMARY KEY',
            'cadastro_id' => 'INT NOT NULL',
            'caminho_imagem' => 'VARCHAR(255) NOT NULL',
            'data_upload' => 'DATETIME NOT NULL'
        ],
        'foreign_keys' => [
            'cadastro_id' => [
                'references' => TABELA_CADASTROS . '(id)',
                'on_delete' => 'CASCADE'
            ]
        ],
        'options' => 'ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci'
    ],
    TABELA_TREINAMENTOS => [
        'columns' => [
            'id' => 'INT AUTO_INCREMENT PRIMARY KEY',
            'cadastro_id' => 'INT NOT NULL',
            'nome' => 'VARCHAR(100) NOT NULL',
            'sobrenome' => 'VARCHAR(100) NOT NULL',
            'caminho_imagem' => 'VARCHAR(255) NOT NULL',
            'data_processamento' => 'DATETIME NOT NULL'
        ],
        'foreign_keys' => [
            'cadastro_id' => [
                'references' => TABELA_CADASTROS . '(id)',
                'on_delete' => 'CASCADE'
            ]
        ],
        'options' => 'ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci'
    ]
];
