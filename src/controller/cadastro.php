<?php
// Ativar exibição de erros para debug (remova em produção)
ini_set('display_errors', 1);
ini_set('display_startup_errors', 1);
error_reporting(E_ALL);

// Iniciar buffer de saída para capturar qualquer saída acidental
ob_start();

require_once __DIR__ . '/../database/config.php';
require_once __DIR__ . '/../database/conexao.php';

define('MAX_FILE_SIZE', 5 * 1024 * 1024); // 5MB
define('MAX_TOTAL_FILES', 10);
define('ALLOWED_TYPES', [
    'image/jpeg' => 'jpg',
    'image/png' => 'png',
    'image/webp' => 'webp'
]);

// Função para log de erros
function logError($message)
{
    $logFile = __DIR__ . '/../../logs/erros.log';
    $dir = dirname($logFile);
    if (!is_dir($dir)) {
        mkdir($dir, 0777, true);
    }
    file_put_contents($logFile, '[' . date('Y-m-d H:i:s') . '] ' . $message . PHP_EOL, FILE_APPEND);
}

function sanitizeFilename($name)
{
    $name = iconv('UTF-8', 'ASCII//TRANSLIT', $name);
    $name = preg_replace('/[^a-zA-Z0-9-_]/', '_', $name);
    $name = strtolower($name);
    $name = preg_replace('/_+/', '_', $name);
    $name = trim($name, '_');
    return substr($name, 0, 50);
}

try {
    if ($_SERVER['REQUEST_METHOD'] !== 'POST') {
        throw new Exception("Método não permitido", 405);
    }

    // Validação dos campos obrigatórios
    $required_fields = ['nome', 'sobrenome'];
    foreach ($required_fields as $field) {
        if (empty($_POST[$field])) {
            throw new Exception("O campo '$field' é obrigatório", 400);
        }
    }

    $nome = trim($_POST['nome']);
    $sobrenome = trim($_POST['sobrenome']);
    $apelido = isset($_POST['apelido']) ? trim($_POST['apelido']) : null;
    $observacoes = isset($_POST['observacoes']) ? trim($_POST['observacoes']) : null;

    // Validação das imagens
    if (empty($_FILES['imagens']) || !is_array($_FILES['imagens']['name'])) {
        throw new Exception("Nenhuma imagem enviada", 400);
    }

    if (count($_FILES['imagens']['name']) < 1) {
        throw new Exception("Pelo menos uma imagem deve ser enviada", 400);
    }

    if (count($_FILES['imagens']['name']) > MAX_TOTAL_FILES) {
        throw new Exception("Máximo de " . MAX_TOTAL_FILES . " imagens permitidas", 400);
    }

    $pdo->beginTransaction();

    // Inserir pessoa
    $stmt_person = $pdo->prepare("INSERT INTO cadastros 
        (nome, sobrenome, apelido, observacoes, data_cadastro) 
        VALUES (:nome, :sobrenome, :apelido, :observacoes, NOW())");

    $stmt_person->execute([
        ':nome' => $nome,
        ':sobrenome' => $sobrenome,
        ':apelido' => $apelido,
        ':observacoes' => $observacoes
    ]);

    $person_id = $pdo->lastInsertId();

    // Criar diretório para as imagens
    $upload_base = __DIR__ . '/../../uploads/';
    if (!file_exists($upload_base)) {
        mkdir($upload_base, 0755, true);
    }

    $user_dirname = sanitizeFilename($nome) . '_' . sanitizeFilename($sobrenome);
    $user_dir = $upload_base . $user_dirname . '/';

    $counter = 1;
    $original_dir = $user_dir;
    while (file_exists($user_dir)) {
        $user_dir = $original_dir . $counter . '/';
        $counter++;
    }
    if (!mkdir($user_dir, 0755, true)) {
        throw new Exception("Não foi possível criar o diretório para as imagens");
    }

    // Preparar query para inserir imagens
    $stmt_image = $pdo->prepare("INSERT INTO imagens_cadastro 
        (cadastro_id, caminho_imagem, data_upload) 
        VALUES (:cadastro_id, :caminho_imagem, NOW())");

    $uploaded_images = 0;

    for ($i = 0; $i < count($_FILES['imagens']['name']); $i++) {
        // Verifica erro no upload
        if ($_FILES['imagens']['error'][$i] !== UPLOAD_ERR_OK) {
            if ($_FILES['imagens']['error'][$i] === UPLOAD_ERR_NO_FILE) {
                continue;
            }
            throw new Exception("Erro no upload da imagem: " . $_FILES['imagens']['name'][$i]);
        }

        // Verifica tipo
        $file_type = $_FILES['imagens']['type'][$i];
        if (!array_key_exists($file_type, ALLOWED_TYPES)) {
            throw new Exception("Tipo de arquivo não suportado: " . $_FILES['imagens']['name'][$i]);
        }

        // Verifica tamanho
        if ($_FILES['imagens']['size'][$i] > MAX_FILE_SIZE) {
            throw new Exception("Arquivo muito grande: " . $_FILES['imagens']['name'][$i]);
        }

        // Obtém a extensão correta a partir do tipo MIME
        $extensao = ALLOWED_TYPES[$file_type];

        // Gera nome único mantendo a extensão original
        $filename = $user_dirname . '_' . time() . '_' . $i . '.' . $extensao;
        $output_path = $user_dir . $filename;

        // Move o arquivo temporário para o destino final (sem processamento)
        if (!move_uploaded_file($_FILES['imagens']['tmp_name'][$i], $output_path)) {
            throw new Exception("Falha ao salvar imagem: " . $_FILES['imagens']['name'][$i]);
        }

        // Salva no banco com o caminho relativo
        $relative_path = $user_dirname . '/' . $filename;
        $stmt_image->execute([
            ':cadastro_id' => $person_id,
            ':caminho_imagem' => $relative_path
        ]);

        $uploaded_images++;
    }

    if ($uploaded_images < 1) {
        throw new Exception("Nenhuma imagem válida foi salva", 400);
    }

    $pdo->commit();

    // Limpa o buffer de saída antes do redirecionamento
    ob_end_clean();

    header('Location: ../../pages/sucesso.html');
    exit();

} catch (Exception $e) {
    // Desfaz a transação se estiver ativa
    if (isset($pdo) && $pdo->inTransaction()) {
        $pdo->rollBack();
    }

    // Remove o diretório de imagens se foi criado
    if (isset($user_dir) && file_exists($user_dir)) {
        array_map('unlink', glob("$user_dir/*"));
        rmdir($user_dir);
    }

    // Log do erro
    logError('Erro no cadastro: ' . $e->getMessage() . ' em ' . $e->getFile() . ':' . $e->getLine());

    // Limpa o buffer e redireciona para página de erro
    ob_end_clean();

    $msg = urlencode($e->getMessage());
    header("Location: ../../pages/erro.html?msg=$msg");
    exit();
}