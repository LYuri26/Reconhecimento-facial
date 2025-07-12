<?php
// src/controller/cadastro.php

require_once __DIR__ . '/../database/config.php';
require_once __DIR__ . '/../database/conexao.php';

// Enhanced configuration
define('MAX_FILE_SIZE', 5 * 1024 * 1024); // 5MB
define('MIN_IMAGES', 1);  // Minimum 1 image required
define('MAX_IMAGES', 1);  // Maximum 1 image allowed (we'll generate variations)
define('ALLOWED_TYPES', [
    'image/jpeg' => 'jpg',
    'image/png' => 'png',
    'image/webp' => 'webp'
]);
define('IMAGE_QUALITY', 90);
define('TARGET_SIZE', 500); // Size for processed images

// Utility function to create image variations
function createImageVariations($source_path, $output_dir, $base_filename)
{
    $variations = [];

    try {
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

        // Original processed
        $original_path = $output_dir . $base_filename . '_original.jpg';
        imagejpeg($image, $original_path, IMAGE_QUALITY);
        $variations[] = $original_path;

        // 1. Flipped
        $flipped = imagecreatetruecolor(TARGET_SIZE, TARGET_SIZE);
        for ($i = 0; $i < TARGET_SIZE; $i++) {
            imagecopy($flipped, $image, $i, 0, TARGET_SIZE - $i - 1, 0, 1, TARGET_SIZE);
        }
        $flipped_path = $output_dir . $base_filename . '_flipped.jpg';
        imagejpeg($flipped, $flipped_path, IMAGE_QUALITY);
        $variations[] = $flipped_path;
        imagedestroy($flipped);

        // 2. Rotated -10 degrees
        $rotated1 = imagerotate($image, -10, 0);
        $rotated1_path = $output_dir . $base_filename . '_rotated1.jpg';
        imagejpeg($rotated1, $rotated1_path, IMAGE_QUALITY);
        $variations[] = $rotated1_path;
        imagedestroy($rotated1);

        // 3. Rotated +10 degrees
        $rotated2 = imagerotate($image, 10, 0);
        $rotated2_path = $output_dir . $base_filename . '_rotated2.jpg';
        imagejpeg($rotated2, $rotated2_path, IMAGE_QUALITY);
        $variations[] = $rotated2_path;
        imagedestroy($rotated2);

        // 4. Brightness adjusted
        $bright = imagecreatetruecolor(TARGET_SIZE, TARGET_SIZE);
        imagecopy($bright, $image, 0, 0, 0, 0, TARGET_SIZE, TARGET_SIZE);
        imagefilter($bright, IMG_FILTER_BRIGHTNESS, -30);
        $bright_path = $output_dir . $base_filename . '_bright.jpg';
        imagejpeg($bright, $bright_path, IMAGE_QUALITY);
        $variations[] = $bright_path;
        imagedestroy($bright);

        imagedestroy($image);
        imagedestroy($source);

        return $variations;
    } catch (Exception $e) {
        error_log("Error generating image variations: " . $e->getMessage());
        return false;
    }
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

// MAIN

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

    if (empty($_FILES['imagens']) || !is_array($_FILES['imagens']['name']) || count($_FILES['imagens']['name']) < MIN_IMAGES) {
        throw new Exception("Pelo menos uma imagem deve ser enviada", 400);
    }

    $pdo->beginTransaction();

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

    $stmt_image = $pdo->prepare("INSERT INTO imagens_cadastro 
        (cadastro_id, caminho_imagem, data_upload) 
        VALUES (:cadastro_id, :caminho_imagem, NOW())");

    $processed_images = 0;

    foreach ($_FILES['imagens']['tmp_name'] as $i => $tmp_path) {
        if ($processed_images > 0) break; // only process first image

        if ($_FILES['imagens']['error'][$i] !== UPLOAD_ERR_OK) {
            throw new Exception("Erro no upload da imagem: " . $_FILES['imagens']['name'][$i]);
        }

        $file_type = $_FILES['imagens']['type'][$i];
        if (!array_key_exists($file_type, ALLOWED_TYPES)) {
            throw new Exception("Tipo de arquivo não suportado: " . $_FILES['imagens']['name'][$i]);
        }

        if ($_FILES['imagens']['size'][$i] > MAX_FILE_SIZE) {
            throw new Exception("Arquivo muito grande: " . $_FILES['imagens']['name'][$i]);
        }

        $filename = $user_dirname . '_' . time();
        $original_path = $user_dir . $filename . '.' . ALLOWED_TYPES[$file_type];

        if (!move_uploaded_file($tmp_path, $original_path)) {
            throw new Exception("Falha ao salvar imagem: " . $_FILES['imagens']['name'][$i]);
        }

        $variations = createImageVariations($original_path, $user_dir, $filename);
        if (!$variations || count($variations) < 4) {
            throw new Exception("Falha ao gerar variações da imagem");
        }

        foreach ($variations as $variation_path) {
            $relative_path = $user_dirname . '/' . basename($variation_path);
            $stmt_image->execute([
                ':cadastro_id' => $person_id,
                ':caminho_imagem' => $relative_path
            ]);
            $processed_images++;
        }
    }

    if ($processed_images < 4) {
        throw new Exception("Falha no processamento das imagens");
    }

    $pdo->commit();

    // Redirecionar para sucesso.html
    header('Location: ../../pages/sucesso.html');
    exit();
} catch (Exception $e) {
    if (isset($pdo) && $pdo->inTransaction()) {
        $pdo->rollBack();
    }

    // Limpar pasta criada em caso de erro
    if (isset($user_dir) && file_exists($user_dir)) {
        array_map('unlink', glob("$user_dir/*"));
        rmdir($user_dir);
    }

    // Redirecionar para erro.html com mensagem na query string
    $msg = urlencode($e->getMessage());
    header("Location: ../../pages/erro.html?msg=$msg");
    exit();
}
