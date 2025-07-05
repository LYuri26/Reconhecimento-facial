<?php require 'includes/conexao.php'; ?>
<!DOCTYPE html>
<html lang="pt-BR">

<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Cadastro Facial - Dados Pessoais</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">
    <link href="css/index.css" rel="stylesheet">
</head>

<body>
    <div class="container py-5">
        <div class="row justify-content-center">
            <div class="col-lg-8">
                <div class="card">
                    <div class="card-header text-center">
                        <h3><i class="fas fa-user-plus me-2"></i> Cadastro Biométrico</h3>
                    </div>
                    <div class="card-body">
                        <form id="formDados" action="capturar-fotos.php" method="post">
                            <div class="row mb-4">
                                <div class="col-md-6 mb-3">
                                    <label for="nome" class="form-label fw-bold">Nome Completo</label>
                                    <input type="text" class="form-control border-dark" id="nome" name="nome" required>
                                </div>
                                <div class="col-md-6 mb-3">
                                    <label for="cpf" class="form-label fw-bold">CPF</label>
                                    <input type="text" class="form-control border-dark" id="cpf" name="cpf" required>
                                </div>
                            </div>

                            <div class="row mb-4">
                                <div class="col-md-6 mb-3">
                                    <label for="email" class="form-label fw-bold">E-mail</label>
                                    <input type="email" class="form-control border-dark" id="email" name="email"
                                        required>
                                </div>
                                <div class="col-md-6 mb-3">
                                    <label for="telefone" class="form-label fw-bold">Telefone</label>
                                    <input type="tel" class="form-control border-dark" id="telefone" name="telefone"
                                        required>
                                </div>
                            </div>

                            <div class="d-grid gap-2 mt-4">
                                <button type="submit" class="btn btn-primary btn-lg">
                                    <i class="fas fa-arrow-right me-2"></i>Próxima Etapa
                                </button>
                            </div>
                        </form>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/jquery/3.6.0/jquery.min.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/jquery.mask/1.14.16/jquery.mask.min.js"></script>
    <script src="js/validacoes.js"></script>
    <script src="js/index.js"></script>
</body>

</html>