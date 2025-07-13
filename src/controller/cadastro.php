<?php
// src/controller/cadastro.php

require_once __DIR__ . '/../database/config.php';
require_once __DIR__ . '/../database/conexao.php';

define('MAX_FILE_SIZE', 5 * 1024 * 1024); // 5MB
define('MAX_TOTAL_FILES', 10); // Máximo de 10 imagens por cadastro
define('ALLOWED_TYPES', [
    'image/jpeg' => 'jpg',
    'image/png' => 'png',
    'image/webp' => 'webp'
]);
define('IMAGE_QUALITY', 90);
define('TARGET_SIZE', 500); // Size for processed images

function sanitizeFilename($name)
{
    $name = iconv('UTF-8', 'ASCII//TRANSLIT', $name);
    $name = preg_replace('/[^a-zA-Z0-9-_]/', '_', $name);
    $name = strtolower($name);
    $name = preg_replace('/_+/', '_', $name);
    $name = trim($name, '_');
    return substr($name, 0, 50);
}

function processSingleImage($source_path, $output_path)
{
    $source_type = exif_imagetype($source_path);
    switch ($source_type) {
        case IMAGETYPE_JPEG:
            $source = imagecreatefromjpeg($source_path);
            break;
        case IMAGETYPE_PNG:
            $source = imagecreatefrompng($source_path);
            break;
        case IMAGETYPE_WEBP:
            $source = imagecreatefromwebp($source_path);
            break;
        default:
            return false;
    }

    if (!$source) return false;

    $width = imagesx($source);
    $height = imagesy($source);
    $size = min($width, $height);
    $x = ($width - $size) / 2;
    $y = ($height - $size) / 2;

    $image = imagecreatetruecolor(TARGET_SIZE, TARGET_SIZE);
    imagecopyresampled($image, $source, 0, 0, $x, $y, TARGET_SIZE, TARGET_SIZE, $size, $size);

    imagejpeg($image, $output_path, IMAGE_QUALITY);

    imagedestroy($image);
    imagedestroy($source);

    return true;
}

try {
    if ($_SERVER['REQUEST_METHOD'] !== 'POST') {
        throw new Exception("Método não permitido", 405);
    }

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

    // Verifica se há pelo menos uma imagem e não excede o máximo
    if (empty($_FILES['imagens']) || !is_array($_FILES['imagens']['name']) || count($_FILES['imagens']['name']) < 1) {
        throw new Exception("Pelo menos uma imagem deve ser enviada", 400);
    }

    if (count($_FILES['imagens']['name']) > MAX_TOTAL_FILES) {
        throw new Exception("Máximo de " . MAX_TOTAL_FILES . " imagens permitidas", 400);
    }

    $pdo->beginTransaction();

    // Insere os dados da pessoa
    $stmt_person = $pdo->prepare("INSERT INTO cadastros 
        (nome, sobrenome, apelido, observacoes, data_cadastro) 
        VALUES (:nome, :sobrenome, :apelido, :observacoes, NOW())");

    $stmt_person->execute([
        ':nome' => $nome,
        ':sobrenome' => $sobrenome,
        ':apelido' => empty($apelido) ? null : $apelido,
        ':observacoes' => empty($observacoes) ? null : $observacoes
    ]);

    $person_id = $pdo->lastInsertId();

    // Prepara o diretório para as imagens
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
    mkdir($user_dir, 0755, true);

    // Prepara a query para inserir as imagens
    $stmt_image = $pdo->prepare("INSERT INTO imagens_cadastro 
        (cadastro_id, caminho_imagem, data_upload) 
        VALUES (:cadastro_id, :caminho_imagem, NOW())");

    // Processa cada imagem enviada
    $total_images = count($_FILES['imagens']['name']);
    $uploaded_images = 0;

    for ($i = 0; $i < $total_images; $i++) {
        // Verifica se há erro no upload
        if ($_FILES['imagens']['error'][$i] !== UPLOAD_ERR_OK) {
            if ($_FILES['imagens']['error'][$i] === UPLOAD_ERR_NO_FILE) {
                continue; // Pula arquivos não enviados (quando há campos vazios no múltiplo upload)
            }
            throw new Exception("Erro no upload da imagem: " . $_FILES['imagens']['name'][$i]);
        }

        // Verifica tipo e tamanho do arquivo
        $file_type = $_FILES['imagens']['type'][$i];
        if (!array_key_exists($file_type, ALLOWED_TYPES)) {
            throw new Exception("Tipo de arquivo não suportado: " . $_FILES['imagens']['name'][$i]);
        }

        if ($_FILES['imagens']['size'][$i] > MAX_FILE_SIZE) {
            throw new Exception("Arquivo muito grande: " . $_FILES['imagens']['name'][$i]);
        }

        // Gera nome único para o arquivo
        $filename = $user_dirname . '_' . time() . '_' . $i;
        $output_path = $user_dir . $filename . '.jpg';

        // Move o arquivo temporário para o destino final
        if (!move_uploaded_file($_FILES['imagens']['tmp_name'][$i], $output_path)) {
            throw new Exception("Falha ao salvar imagem: " . $_FILES['imagens']['name'][$i]);
        }

        // Processa a imagem (redimensiona e converte para JPEG)
        if (!processSingleImage($output_path, $output_path)) {
            throw new Exception("Erro ao processar imagem: " . $_FILES['imagens']['name'][$i]);
        }

        // Salva o caminho relativo no banco de dados
        $relative_path = $user_dirname . '/' . basename($output_path);
        $stmt_image->execute([
            ':cadastro_id' => $person_id,
            ':caminho_imagem' => $relative_path
        ]);

        $uploaded_images++;
    }

    // Verifica se pelo menos uma imagem foi enviada com sucesso
    if ($uploaded_images < 1) {
        throw new Exception("Nenhuma imagem válida foi enviada", 400);
    }

    $pdo->commit();
    header('Location: ../../pages/sucesso.html');
    exit();
} catch (Exception $e) {
    if (isset($pdo) && $pdo->inTransaction()) {
        $pdo->rollBack();
    }

    if (isset($user_dir) && file_exists($user_dir)) {
        array_map('unlink', glob("$user_dir/*"));
        rmdir($user_dir);
    }

    $msg = urlencode($e->getMessage());
    header("Location: ../../pages/erro.html?msg=$msg");
    exit();
}
