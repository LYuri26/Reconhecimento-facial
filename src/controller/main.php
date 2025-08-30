<?php
// src/controller/main.php

header('Content-Type: application/json');
header('Access-Control-Allow-Origin: *');
header('Access-Control-Allow-Methods: POST, GET, OPTIONS');
header('Access-Control-Allow-Headers: Content-Type');

// Resposta padrão
$response = [
    'success' => false,
    'error' => '',
    'output' => '',
    'real_time' => false
];

// Verificar se é uma requisição GET para tempo real
if ($_SERVER['REQUEST_METHOD'] == 'GET' && isset($_GET['real_time']) && $_GET['real_time'] == '1') {
    $acao = $_GET['acao'] ?? '';

    if (!empty($acao)) {
        // Determinar o caminho base do projeto
        $projectRoot = realpath(dirname(__FILE__) . '/../..');
        $scriptPath = $projectRoot . '/main.py';

        // Tentar encontrar o Python disponível
        $pythonCommand = encontrarPython();

        if ($pythonCommand && file_exists($scriptPath)) {
            executarComandoTempoReal($pythonCommand, $scriptPath, $acao, $projectRoot);
            exit;
        }
    }

    // Se chegou aqui, houve erro
    header('Content-Type: text/event-stream');
    echo "data: " . json_encode(['error' => 'Parâmetros inválidos']) . "\n\n";
    exit;
}

// Verificar se a requisição é POST (para o método original)
if ($_SERVER['REQUEST_METHOD'] == 'POST') {
    // Ler os dados JSON da requisição
    $input = json_decode(file_get_contents('php://input'), true);

    if (isset($input['acao'])) {
        $acao = $input['acao'];

        // Determinar o caminho base do projeto
        $projectRoot = realpath(dirname(__FILE__) . '/../..');
        $scriptPath = $projectRoot . '/main.py';

        // Verificar se o script Python existe
        if (!file_exists($scriptPath)) {
            $response['error'] = 'Script Python não encontrado em: ' . $scriptPath;
            echo json_encode($response);
            exit;
        }

        // Tentar encontrar o Python disponível
        $pythonCommand = encontrarPython();
        if (!$pythonCommand) {
            $response['error'] = 'Python não encontrado no sistema. Instale Python 3.10+ primeiro.';
            echo json_encode($response);
            exit;
        }

        // Verificar se o Python funciona
        if (!testarPython($pythonCommand)) {
            $response['error'] = 'Python encontrado mas não funciona corretamente.';
            echo json_encode($response);
            exit;
        }

        // PRIMEIRO: Configurar ambiente virtual e instalar dependências
        $setupComando = escapeshellarg($pythonCommand) . ' ' . escapeshellarg($scriptPath) . ' 2>&1';

        try {
            // Mudar para o diretório do projeto
            $oldCwd = getcwd();
            chdir($projectRoot);

            // Executar configuração do ambiente primeiro
            $setupOutput = [];
            $setupReturn = 0;
            exec($setupComando, $setupOutput, $setupReturn);

            // Voltar ao diretório original
            chdir($oldCwd);

            if ($setupReturn !== 0) {
                $response['error'] = 'Erro ao configurar ambiente virtual';
                $response['output'] = implode("\n", $setupOutput);
                echo json_encode($response);
                exit;
            }

            // SE a ação for apenas setup, retornar sucesso
            if ($acao === 'setup_ambiente') {
                $response['success'] = true;
                $response['output'] = implode("\n", $setupOutput);
                echo json_encode($response);
                exit;
            }

            // Definir o comando baseado na ação
            $comando = '';
            $arg = '';

            if ($acao === 'treinamento_ia') {
                $arg = '--treinamento';
            } elseif ($acao === 'iniciar_cameras') {
                $arg = '--reconhecimento';
            } else {
                $response['error'] = 'Ação não reconhecida: ' . $acao;
                echo json_encode($response);
                exit;
            }

            // Construir comando seguro usando o Python do venv
            $venvPythonPath = encontrarPythonVenv($projectRoot);
            if ($venvPythonPath) {
                $pythonCommand = $venvPythonPath;
            }

            $comando = escapeshellarg($pythonCommand) . ' ' . escapeshellarg($scriptPath) . ' ' . $arg . ' 2>&1';

            // Executar o comando específico da ação
            $output = [];
            $return_var = 0;

            // Mudar para o diretório do projeto novamente
            $oldCwd = getcwd();
            chdir($projectRoot);

            // Executar comando
            exec($comando, $output, $return_var);

            // Voltar ao diretório original
            chdir($oldCwd);

            // Processar saída
            $outputStr = implode("\n", $output);

            if ($return_var === 0) {
                $response['success'] = true;
                $response['output'] = $outputStr;
            } else {
                $response['error'] = 'Erro ao executar (código: ' . $return_var . ')';
                $response['output'] = $outputStr;

                // Log detalhado para debugging
                error_log("Comando falhou: " . $comando);
                error_log("Código de retorno: " . $return_var);
                error_log("Saída: " . $outputStr);
            }
        } catch (Exception $e) {
            $response['error'] = 'Exceção: ' . $e->getMessage();
        }
    } else {
        $response['error'] = 'Parâmetro "acao" não especificado';
    }
} else {
    $response['error'] = 'Método não permitido. Use POST.';
}

// Retornar a resposta como JSON
echo json_encode($response);

function executarComandoTempoReal($pythonCommand, $scriptPath, $acao, $projectRoot)
{
    // Primeiro, tentar encontrar o Python do venv
    $venvPythonPath = encontrarPythonVenv($projectRoot);
    if ($venvPythonPath) {
        $pythonCommand = $venvPythonPath;
    }

    // Resto do código permanece igual...
    $descriptorspec = array(
        0 => array("pipe", "r"),  // stdin
        1 => array("pipe", "w"),  // stdout
        2 => array("pipe", "w")   // stderr
    );

    // Definir o comando baseado na ação
    $arg = '';
    if ($acao === 'treinamento_ia') {
        $arg = '--treinamento';
    } elseif ($acao === 'iniciar_cameras') {
        $arg = '--reconhecimento';
    } elseif ($acao === 'setup_ambiente') {
        $arg = '';
    }

    $comando = escapeshellarg($pythonCommand) . ' ' . escapeshellarg($scriptPath);
    if (!empty($arg)) {
        $comando .= ' ' . $arg;
    }
    $comando .= ' 2>&1';

    $process = proc_open($comando, $descriptorspec, $pipes, $projectRoot);

    if (is_resource($process)) {
        // Fechar stdin já que não vamos usar
        fclose($pipes[0]);

        // Configurar headers para streaming
        header('Content-Type: text/event-stream');
        header('Cache-Control: no-cache');
        header('X-Accel-Buffering: no'); // Desabilitar buffering do Nginx
        header('Access-Control-Allow-Origin: *');
        header('Access-Control-Allow-Methods: GET');

        // Ler a saída em tempo real
        while (!feof($pipes[1])) {
            $output = fgets($pipes[1]);
            if ($output !== false) {
                // Enviar como evento SSE
                echo "data: " . json_encode(['output' => $output]) . "\n\n";
                ob_flush();
                flush();

                // Pequena pausa para não sobrecarregar
                usleep(10000); // 10ms
            }
        }

        // Fechar pipes e processo
        fclose($pipes[1]);
        fclose($pipes[2]);
        $returnCode = proc_close($process);

        // Enviar evento de finalização
        echo "data: " . json_encode(['finished' => true, 'return_code' => $returnCode]) . "\n\n";
        ob_flush();
        flush();
    } else {
        // Erro ao abrir processo
        header('Content-Type: text/event-stream');
        echo "data: " . json_encode(['error' => 'Erro ao iniciar processo']) . "\n\n";
        ob_flush();
        flush();
    }
}

function encontrarPython()
{
    $comandos = ['python3.10', 'python3', 'python', 'py', 'python3.11', 'python3.12', 'python3.13'];

    foreach ($comandos as $cmd) {
        // Linux/Mac
        $output = [];
        $return_var = 0;
        exec("which " . escapeshellarg($cmd) . " 2>/dev/null", $output, $return_var);

        if ($return_var === 0 && !empty($output) && file_exists(trim($output[0]))) {
            return trim($output[0]);
        }

        // Windows
        $output = [];
        $return_var = 0;
        exec("where " . escapeshellarg($cmd) . " 2>nul", $output, $return_var);

        if ($return_var === 0 && !empty($output) && file_exists(trim($output[0]))) {
            return trim($output[0]);
        }
    }

    return null;
}

function encontrarPythonVenv($projectRoot)
{
    $venvPath = $projectRoot . '/venv';

    if (is_dir($venvPath)) {
        if (strtoupper(substr(PHP_OS, 0, 3)) === 'WIN') {
            $pythonPath = $venvPath . '/Scripts/python.exe';
        } else {
            $pythonPath = $venvPath . '/bin/python';
        }

        if (file_exists($pythonPath)) {
            return $pythonPath;
        }
    }

    return null;
}

function testarPython($pythonPath)
{
    $output = [];
    $return_var = 0;

    // Testar se o Python funciona e é versão 3.10+
    exec(escapeshellarg($pythonPath) . ' --version 2>&1', $output, $return_var);

    if ($return_var === 0 && !empty($output)) {
        $versionOutput = $output[0];
        if (preg_match('/Python (\d+)\.(\d+)\./', $versionOutput, $matches)) {
            $major = (int)$matches[1];
            $minor = (int)$matches[2];

            // Verificar se é Python 3.10 ou superior
            if ($major === 3 && $minor >= 10) {
                return true;
            }
        }
    }

    return false;
}
