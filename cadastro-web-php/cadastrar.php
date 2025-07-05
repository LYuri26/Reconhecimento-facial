<?php
require 'includes/conexao.php';

header('Content-Type: application/json');

if ($_SERVER['REQUEST_METHOD'] === 'POST') {
    $nome = $_POST['nome'] ?? '';
    $cpf = $_POST['cpf'] ?? '';
    $email = $_POST['email'] ?? '';
    $telefone = $_POST['telefone'] ?? '';
    $fotos = json_decode($_POST['fotos'] ?? '[]', true);

    try {
        // Validações básicas
        if (empty($nome) || empty($cpf) || empty($fotos)) {
            throw new Exception('Todos os campos obrigatórios devem ser preenchidos');
        }

        // Verificar se CPF já existe
        $stmt = $pdo->prepare("SELECT ID FROM Pessoas WHERE cpf = ?");
        $stmt->execute([$cpf]);
        if ($stmt->fetch()) {
            throw new Exception('CPF já cadastrado no sistema');
        }

        // Criar pasta para o usuário
        $pastaUsuario = 'imagens/rostos_salvos/' . md5($cpf);
        if (!file_exists($pastaUsuario)) {
            mkdir($pastaUsuario, 0777, true);
        }

        // Salvar múltiplas fotos
        $fotosSalvas = [];
        foreach ($fotos as $index => $fotoBase64) {
            $imageData = base64_decode(preg_replace('#^data:image/\w+;base64,#i', '', $fotoBase64));
            $nomeImagem = $index . '_' . md5(uniqid()) . '.jpg';
            file_put_contents($pastaUsuario . '/' . $nomeImagem, $imageData);
            $fotosSalvas[] = $nomeImagem;
        }

        // Inserir no banco
        $stmt = $pdo->prepare("INSERT INTO Pessoas (Nome, CPF, Email, Telefone, Pasta) VALUES (?, ?, ?, ?, ?)");
        $stmt->execute([$nome, $cpf, $email, $telefone, $pastaUsuario]);

        echo json_encode([
            'success' => true,
            'message' => 'Cadastro realizado com sucesso!',
            'pasta' => $pastaUsuario,
            'fotos' => $fotosSalvas
        ]);
    } catch (Exception $e) {
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
