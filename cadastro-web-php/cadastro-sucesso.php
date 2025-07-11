<?php
require 'includes/conexao.php';

$id = $_GET['id'] ?? null;

if (!$id) {
    header('Location: index.php');
    exit;
}

$stmt = $pdo->prepare("SELECT * FROM Pessoas WHERE id = ?");
$stmt->execute([$id]);
$individuo = $stmt->fetch();

if (!$individuo) {
    header('Location: index.php');
    exit;
}

// Mapear cores para níveis de perigo
$coresPerigo = [
    'ALTO' => 'danger',
    'MEDIO' => 'warning',
    'BAIXO' => 'success'
];
?>
<!DOCTYPE html>
<html lang="pt-BR">

<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Visualizar Indivíduo</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">
    <link href="css/estilo.css" rel="stylesheet">
</head>

<body>
    <div class="container py-5">
        <div class="row justify-content-center">
            <div class="col-lg-8">
                <div class="card shadow-lg">
                    <div class="card-header bg-<?= $coresPerigo[$individuo['nivel_perigo']] ?> text-white">
                        <h3 class="mb-0">
                            <i class="fas fa-user me-2"></i>
                            <?= htmlspecialchars($individuo['nome'] . ' ' . $individuo['sobrenome']) ?>
                            <?php if ($individuo['foragido']): ?>
                                <span class="badge bg-dark float-end">FORAGIDO</span>
                            <?php endif; ?>
                        </h3>
                    </div>
                    <div class="card-body">
                        <div class="row">
                            <div class="col-md-5 text-center">
                                <img src="<?= htmlspecialchars($individuo['pasta']) . htmlspecialchars($individuo['foto']) ?>"
                                    class="img-fluid rounded mb-3 border border-3 border-<?= $coresPerigo[$individuo['nivel_perigo']] ?>"
                                    alt="Foto cadastrada">

                                <div class="alert alert-<?= $coresPerigo[$individuo['nivel_perigo']] ?>">
                                    <strong>Nível de Perigo:</strong>
                                    <?= $individuo['nivel_perigo'] ?>
                                </div>
                            </div>
                            <div class="col-md-7">
                                <h4>Informações</h4>
                                <ul class="list-group mb-4">
                                    <li class="list-group-item">
                                        <strong>Apelido:</strong>
                                        <?= $individuo['apelido'] ? htmlspecialchars($individuo['apelido']) : 'Nenhum' ?>
                                    </li>
                                    <li class="list-group-item">
                                        <strong>Status:</strong>
                                        <?= $individuo['foragido'] ? 'Foragido' : 'Não foragido' ?>
                                    </li>
                                    <li class="list-group-item">
                                        <strong>Data Cadastro:</strong>
                                        <?= date('d/m/Y H:i', strtotime($individuo['data_cadastro'])) ?>
                                    </li>
                                </ul>

                                <h4>Observações</h4>
                                <div class="card mb-4">
                                    <div class="card-body">
                                        <?= $individuo['observacoes'] ? nl2br(htmlspecialchars($individuo['observacoes'])) : 'Nenhuma observação cadastrada' ?>
                                    </div>
                                </div>

                                <div class="d-grid gap-2">
                                    <a href="index.php" class="btn btn-primary">
                                        <i class="fas fa-plus me-2"></i>Novo Cadastro
                                    </a>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
</body>

</html>