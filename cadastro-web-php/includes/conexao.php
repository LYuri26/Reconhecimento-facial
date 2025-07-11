<?php
error_reporting(E_ALL);
ini_set('display_errors', 1);

$host = 'localhost';
$user = 'root';
$pass = '';
$db = 'indentificacao';
$charset = 'utf8mb4';

try {
    // Conexão inicial
    $dsn = "mysql:host=$host;charset=$charset";
    $options = [
        PDO::ATTR_ERRMODE => PDO::ERRMODE_EXCEPTION,
        PDO::ATTR_DEFAULT_FETCH_MODE => PDO::FETCH_ASSOC,
        PDO::ATTR_EMULATE_PREPARES => false,
    ];

    $pdo = new PDO($dsn, $user, $pass, $options);

    // Cria o banco de dados
    $pdo->exec("CREATE DATABASE IF NOT EXISTS `$db` 
                CHARACTER SET utf8mb4 
                COLLATE utf8mb4_unicode_ci");

    // Conexão definitiva
    $pdo = new PDO("mysql:host=$host;dbname=$db;charset=$charset", $user, $pass, $options);

    // Configurações seguras (sem timezone problemático)
    $pdo->exec("SET NAMES utf8mb4");
    // $pdo->exec("SET time_zone = '-03:00'"); // Opcional: usar offset se timezone falhar

    require_once __DIR__ . '/tabelas.php';
    criarTabelas($pdo);
} catch (\PDOException $e) {
    die("Erro de conexão: " . $e->getMessage());
}
