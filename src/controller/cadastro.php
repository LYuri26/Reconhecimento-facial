<?php
// src/controller/cadastro.php

require_once __DIR__ . '/../database/config.php';
require_once __DIR__ . '/../database/conexao.php';

// Constantes para uploads
define('MAX_FILE_SIZE', 2 * 1024 * 1024); // 2 MB
define('MAX_IMAGES', 20);
define('ALLOWED_TYPES', ['image/jpeg', 'image/png', 'image/gif']);

if ($_SERVER['REQUEST_METHOD'] !== 'POST') {
    header('Location: ../../index.html');
    exit();
}

try {
    // Validação dos campos enviados
    $nome = filter_input(INPUT_POST, 'nome', FILTER_SANITIZE_STRING);
    $sobrenome = filter_input(INPUT_POST, 'sobrenome', FILTER_SANITIZE_STRING);
    $apelido = filter_input(INPUT_POST, 'apelido', FILTER_SANITIZE_STRING);
    $observacoes = filter_input(INPUT_POST, 'observacoes', FILTER_SANITIZE_STRING);

    if (!$nome || !$sobrenome) {
        throw new Exception('Nome e sobrenome são obrigatórios.');
    }

    if (empty($_FILES['imagens']) || !is_array($_FILES['imagens']['name']) || count($_FILES['imagens']['name']) == 0) {
        throw new Exception('Nenhuma imagem enviada.');
    }

    $pdo->beginTransaction();

    // Inserção do cadastro principal
    $stmt = $pdo->prepare("INSERT INTO " . TABELA_CADASTROS . " (nome, sobrenome, apelido, observacoes, data_cadastro) VALUES (:nome, :sobrenome, :apelido, :observacoes, NOW())");
    $stmt->execute([
        ':nome' => $nome,
        ':sobrenome' => $sobrenome,
        ':apelido' => $apelido ?: null,
        ':observacoes' => $observacoes ?: null
    ]);

    $cadastroId = $pdo->lastInsertId();

    // Cria pasta principal de uploads se não existir
    $uploadBaseDir = __DIR__ . '/../../uploads/';
    if (!file_exists($uploadBaseDir)) {
        mkdir($uploadBaseDir, 0777, true);
    }

    // Cria pasta específica para o usuário no formato nome_sobrenome
    $userDirName = sanitizeFileName($nome) . '_' . sanitizeFileName($sobrenome);
    $userDir = $uploadBaseDir . $userDirName . '/';

    // Verifica se pasta já existe e adiciona sufixo se necessário
    $originalDir = $userDir;
    $counter = 1;
    while (file_exists($userDir)) {
        $userDir = $originalDir . '_' . $counter . '/';
        $counter++;
    }

    if (!file_exists($userDir)) {
        mkdir($userDir, 0777, true);
    }

    $stmtImg = $pdo->prepare("INSERT INTO " . TABELA_IMAGENS . " (cadastro_id, caminho_imagem, data_upload) VALUES (:cadastro_id, :caminho_imagem, NOW())");

    $imagensValidadas = 0;

    for ($i = 0; $i < count($_FILES['imagens']['name']); $i++) {
        $error = $_FILES['imagens']['error'][$i];
        if ($error !== UPLOAD_ERR_OK) continue;

        $tipo = $_FILES['imagens']['type'][$i];
        $tamanho = $_FILES['imagens']['size'][$i];

        if (!in_array($tipo, ALLOWED_TYPES)) {
            throw new Exception('Tipo de arquivo não permitido: ' . $_FILES['imagens']['name'][$i]);
        }

        if ($tamanho > MAX_FILE_SIZE) {
            throw new Exception('Arquivo muito grande: ' . $_FILES['imagens']['name'][$i]);
        }

        // Gera nome do arquivo com nome do usuário e número sequencial
        $nomeArquivo = sanitizeFileName($nome) . '_' . sanitizeFileName($sobrenome) . '_' . ($i + 1) . '.' . pathinfo($_FILES['imagens']['name'][$i], PATHINFO_EXTENSION);
        $destino = $userDir . $nomeArquivo;

        if (move_uploaded_file($_FILES['imagens']['tmp_name'][$i], $destino)) {
            $stmtImg->execute([
                ':cadastro_id' => $cadastroId,
                ':caminho_imagem' => $userDirName . '/' . $nomeArquivo
            ]);
            $imagensValidadas++;
        }
    }

    if ($imagensValidadas < 1) {
        throw new Exception('Pelo menos uma imagem válida deve ser enviada.');
    }

    if ($imagensValidadas > MAX_IMAGES) {
        throw new Exception('Número máximo de imagens excedido (' . MAX_IMAGES . ').');
    }

    $pdo->commit();

    header('Location: ../../pages/sucesso.html');
    exit();
} catch (Exception $e) {
    if ($pdo->inTransaction()) {
        $pdo->rollBack();
    }

    $msg = urlencode($e->getMessage());
    header("Location: ../../pages/erro.html?msg=$msg");
    exit();
}

// Função para sanitizar nomes de arquivo
function sanitizeFileName($name)
{
    $name = iconv('UTF-8', 'ASCII//TRANSLIT', $name);
    $name = preg_replace('/[^a-zA-Z0-9-_]/', '_', $name);
    $name = strtolower($name);
    $name = preg_replace('/_+/', '_', $name);
    $name = trim($name, '_');
    return $name;
}
