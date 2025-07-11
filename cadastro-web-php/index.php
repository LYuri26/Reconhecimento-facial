<?php
require 'includes/conexao.php';

// Inicializa as variáveis para evitar warnings
$error = $_GET['error'] ?? '';
$success = $_GET['success'] ?? '';
?>
<!DOCTYPE html>
<html lang="pt-BR">

<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Sistema de Reconhecimento Facial</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">
    <link href="css/index.css" rel="stylesheet">
</head>

<body>
    <div class="container py-5">
        <?php if (!empty($error)): ?>
            <div class="alert alert-danger alert-dismissible fade show mb-4">
                <i class="fas fa-exclamation-circle me-2"></i>
                <?= htmlspecialchars($error) ?>
                <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
            </div>
        <?php endif; ?>

        <?php if (!empty($success)): ?>
            <div class="alert alert-success alert-dismissible fade show mb-4">
                <i class="fas fa-check-circle me-2"></i>
                <?= htmlspecialchars($success) ?>
                <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
            </div>
        <?php endif; ?>

        <div class="row justify-content-center">
            <div class="col-lg-8">
                <div class="card shadow-lg">
                    <div class="card-header text-center py-3">
                        <h3 class="mb-0"><i class="fas fa-fingerprint me-2"></i> Cadastro Biométrico</h3>
                    </div>
                    <div class="card-body p-4">
                        <form id="formCadastro" action="processar-cadastro.php" method="post"
                            enctype="multipart/form-data">
                            <div class="row mb-4">
                                <div class="col-md-4 mb-3">
                                    <label for="nome" class="form-label fw-bold">Nome *</label>
                                    <input type="text" class="form-control" id="nome" name="nome" required>
                                </div>
                                <div class="col-md-4 mb-3">
                                    <label for="sobrenome" class="form-label fw-bold">Sobrenome *</label>
                                    <input type="text" class="form-control" id="sobrenome" name="sobrenome" required>
                                </div>
                                <div class="col-md-4 mb-3">
                                    <label for="apelido" class="form-label">Apelido</label>
                                    <input type="text" class="form-control" id="apelido" name="apelido">
                                </div>
                            </div>

                            <div class="row mb-4">
                                <div class="col-md-6 mb-3">
                                    <label for="nivel_perigo" class="form-label fw-bold">Nível de Perigo *</label>
                                    <select class="form-select" id="nivel_perigo" name="nivel_perigo" required>
                                        <option value="ALTO">Alto Perigo</option>
                                        <option value="MEDIO">Médio Perigo</option>
                                        <option value="BAIXO" selected>Baixo Perigo</option>
                                    </select>
                                </div>
                                <div class="col-md-6 mb-3">
                                    <label for="foragido" class="form-label fw-bold">Status *</label>
                                    <select class="form-select" id="foragido" name="foragido" required>
                                        <option value="0">Não foragido</option>
                                        <option value="1">Foragido</option>
                                    </select>
                                </div>
                            </div>

                            <div class="mb-4">
                                <label for="foto" class="form-label fw-bold">Foto para Reconhecimento *</label>
                                <input class="form-control" type="file" id="foto" name="foto" accept="image/*" required>
                                <div class="form-text">Formatos aceitos: JPG, PNG (Máx. 5MB)</div>

                                <div class="mt-3 text-center">
                                    <img id="preview" class="preview-img" alt="Pré-visualização da imagem">
                                </div>
                            </div>

                            <div class="mb-4">
                                <label for="observacoes" class="form-label">Observações</label>
                                <textarea class="form-control" id="observacoes" name="observacoes" rows="3"
                                    placeholder="Marcas distintivas, características, etc."></textarea>
                            </div>

                            <div class="d-grid gap-2 mt-4">
                                <button type="submit" class="btn btn-primary btn-lg py-3">
                                    <i class="fas fa-save me-2"></i>Cadastrar Indivíduo
                                </button>
                            </div>
                        </form>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
    <script src="js/index.js"></script>
</body>

</html>