<?php
require 'includes/conexao.php';

session_start();

// Verificar se há dados na sessão
if (!isset($_SESSION['cadastro_temp'])) {
    header('Location: index.php');
    exit;
}

// Recuperar dados da sessão
$dados = $_SESSION['cadastro_temp'];
$nome = $dados['nome'];
$cpf = $dados['cpf'];
$email = $dados['email'];
$telefone = $dados['telefone'];
$pastaUsuario = $dados['pasta'];

// Processar fotos
$fotos = json_decode($_POST['fotos'] ?? '[]', true);
$fotosSalvas = [];

try {
    // Salvar fotos na pasta do usuário
    foreach ($fotos as $index => $fotoBase64) {
        $imageData = base64_decode(preg_replace('#^data:image/\w+;base64,#i', '', $fotoBase64));
        $nomeImagem = $index . '_' . md5(uniqid()) . '.jpg';
        file_put_contents($pastaUsuario . '/' . $nomeImagem, $imageData);
        $fotosSalvas[] = $nomeImagem;
    }

    // Inserir no banco de dados
    $stmt = $pdo->prepare("INSERT INTO Pessoas (Nome, CPF, Email, Telefone, Pasta) VALUES (?, ?, ?, ?, ?)");
    $stmt->execute([$nome, $cpf, $email, $telefone, $pastaUsuario]);

    // Limpar sessão
    unset($_SESSION['cadastro_temp']);

    // Redirecionar para página de sucesso
    header('Location: cadastro-sucesso.php?pasta=' . urlencode($pastaUsuario) . '&fotos=' . count($fotosSalvas));
    exit;
} catch (Exception $e) {
    // Em caso de erro, limpar a pasta criada
    if (file_exists($pastaUsuario)) {
        array_map('unlink', glob("$pastaUsuario/*"));
        rmdir($pastaUsuario);
    }

    die("Erro ao finalizar cadastro: " . $e->getMessage());
}
