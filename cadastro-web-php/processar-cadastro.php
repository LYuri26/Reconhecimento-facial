<?php
require 'includes/conexao.php';

header('Content-Type: application/json');

if ($_SERVER['REQUEST_METHOD'] === 'POST') {
    try {
        // Validar dados obrigatórios
        $required = ['nome', 'sobrenome', 'nivel_perigo', 'foragido'];
        foreach ($required as $field) {
            if (empty($_POST[$field])) {
                throw new Exception("O campo $field é obrigatório");
            }
        }

        // Verificar se pessoa já existe
        $stmt = $pdo->prepare("SELECT id FROM Pessoas WHERE nome = ? AND sobrenome = ?");
        $stmt->execute([$_POST['nome'], $_POST['sobrenome']]);

        if ($stmt->fetch()) {
            throw new Exception('Esta pessoa já está cadastrada no sistema');
        }

        // Processar upload da foto
        if (!isset($_FILES['foto']) || $_FILES['foto']['error'] !== UPLOAD_ERR_OK) {
            throw new Exception('Erro no upload da foto');
        }

        // Validar imagem
        $fileInfo = getimagesize($_FILES['foto']['tmp_name']);
        if (!$fileInfo) {
            throw new Exception('O arquivo enviado não é uma imagem válida');
        }

        // Criar nome da pasta
        $nomePasta = formatarNomePasta($_POST['nome'], $_POST['sobrenome']);
        $uploadDir = 'uploads/rostos/' . $nomePasta . '/';

        // Criar pasta se não existir
        if (!file_exists($uploadDir)) {
            mkdir($uploadDir, 0755, true);
        }

        // Gerar nome da foto
        $nomeImagem = 'foto_' . $nomePasta . '.jpg';
        $caminhoCompleto = $uploadDir . $nomeImagem;

        // Mover arquivo
        if (!move_uploaded_file($_FILES['foto']['tmp_name'], $caminhoCompleto)) {
            throw new Exception('Falha ao salvar a imagem');
        }

        // Inserir no banco
        $stmt = $pdo->prepare("INSERT INTO Pessoas 
                              (nome, sobrenome, apelido, foto, pasta, nivel_perigo, foragido, observacoes) 
                              VALUES (?, ?, ?, ?, ?, ?, ?, ?)");

        $stmt->execute([
            $_POST['nome'],
            $_POST['sobrenome'],
            $_POST['apelido'] ?? null,
            $nomeImagem,
            $uploadDir,
            $_POST['nivel_perigo'],
            (bool)$_POST['foragido'],
            $_POST['observacoes'] ?? null
        ]);

        // Registrar log
        $pdo->prepare("INSERT INTO Logs (acao, detalhes) VALUES (?, ?)")
            ->execute([
                'CADASTRO',
                "Novo indivíduo cadastrado: {$_POST['nome']} {$_POST['sobrenome']}"
            ]);

        echo json_encode([
            'success' => true,
            'message' => 'Cadastro realizado com sucesso!',
            'redirect' => 'cadastro-sucesso.php?id=' . $pdo->lastInsertId()
        ]);
    } catch (Exception $e) {
        // Limpeza em caso de erro
        if (isset($caminhoCompleto) && file_exists($caminhoCompleto)) {
            unlink($caminhoCompleto);
        }
        if (isset($uploadDir) && file_exists($uploadDir)) {
            @rmdir($uploadDir);
        }

        echo json_encode([
            'success' => false,
            'message' => $e->getMessage()
        ]);
    }
} else {
    echo json_encode([
        'success' => false,
        'message' => 'Método não permitido'
    ]);
}

function formatarNomePasta($nome, $sobrenome)
{
    $nome = preg_replace('/[^A-Za-z0-9]/', '', iconv('UTF-8', 'ASCII//TRANSLIT', $nome));
    $sobrenome = preg_replace('/[^A-Za-z0-9]/', '', iconv('UTF-8', 'ASCII//TRANSLIT', $sobrenome));
    return strtolower($nome . '_' . $sobrenome);
}
