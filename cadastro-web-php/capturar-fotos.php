<?php
require 'includes/conexao.php';

// Verificar se os dados foram enviados
if ($_SERVER['REQUEST_METHOD'] !== 'POST') {
    header('Location: index.php');
    exit;
}

// Recuperar dados do formulário
$nome = $_POST['nome'] ?? '';
$cpf = $_POST['cpf'] ?? '';
$email = $_POST['email'] ?? '';
$telefone = $_POST['telefone'] ?? '';

// Verificar se o CPF já existe
$stmt = $pdo->prepare("SELECT ID FROM Pessoas WHERE cpf = ?");
$stmt->execute([$cpf]);
if ($stmt->fetch()) {
    die('CPF já cadastrado no sistema');
}

// Criar pasta temporária
$pastaUsuario = 'imagens/rostos_salvos/' . preg_replace('/[^a-zA-Z0-9]/', '_', $nome);
if (!file_exists($pastaUsuario)) {
    mkdir($pastaUsuario, 0777, true);
}

// Armazenar dados em sessão
session_start();
$_SESSION['cadastro_temp'] = [
    'nome' => $nome,
    'cpf' => $cpf,
    'email' => $email,
    'telefone' => $telefone,
    'pasta' => $pastaUsuario
];
?>
<!DOCTYPE html>
<html lang="pt-BR">

<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Captura de Fotos - <?= htmlspecialchars($nome) ?></title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">
    <link href="css/capturar-fotos.css" rel="stylesheet">
</head>

<body>
    <div class="container py-5">
        <div class="row justify-content-center">
            <div class="col-lg-8">
                <div class="card">
                    <div class="card-header text-center">
                        <h3><i class="fas fa-camera me-2"></i> Captura de Fotos</h3>
                    </div>
                    <div class="card-body">
                        <div class="row mb-4">
                            <div class="col-md-6">
                                <div class="camera-container mb-3">
                                    <video id="video" width="100%" height="240" autoplay></video>
                                    <canvas id="canvas" width="320" height="240" style="display:none;"></canvas>
                                </div>
                                <div class="d-flex justify-content-between">
                                    <button type="button" id="capturar" class="btn btn-primary">
                                        <i class="fas fa-camera me-2"></i>Capturar
                                    </button>
                                    <button type="button" id="iniciarCaptura" class="btn btn-primary">
                                        <i class="fas fa-video me-2"></i>Captura Automática
                                    </button>
                                    <button type="button" id="reiniciar" class="btn btn-secondary">
                                        <i class="fas fa-redo me-2"></i>Reiniciar
                                    </button>
                                </div>
                                <div class="progress-container mt-2" id="progressContainer" style="display:none;">
                                    <div>Capturando fotos... <span id="contador">0</span>/20</div>
                                    <div class="progress mt-1">
                                        <div id="progressBar" class="progress-bar" role="progressbar" style="width: 0%">
                                        </div>
                                    </div>
                                </div>
                            </div>
                            <div class="col-md-6">
                                <div id="fotoPreview" class="foto-preview">
                                    <p class="text-muted mb-0">Nenhuma foto capturada</p>
                                </div>
                                <div class="mt-2 text-center">
                                    <small class="text-muted">Capturas realizadas: <span
                                            id="totalFotos">0</span></small>
                                </div>
                            </div>
                        </div>

                        <form id="formFotos" action="finalizar-cadastro.php" method="post">
                            <input type="hidden" id="fotos" name="fotos">
                            <div class="d-grid gap-2">
                                <button type="submit" id="btnFinalizar" class="btn btn-success btn-lg" disabled>
                                    <i class="fas fa-save me-2"></i>Finalizar Cadastro
                                </button>
                            </div>
                        </form>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
    <script src="js/capturar-fotos.js"></script>
</body>

</html>