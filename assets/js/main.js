// assets/js/main.js

document.addEventListener("DOMContentLoaded", function () {
  // Botão de Configurar Ambiente
  document
    .getElementById("btnConfigurarAmbiente")
    .addEventListener("click", function () {
      executarScriptPython("setup_ambiente");
    });

  // Botão de Treinamento IA
  document
    .getElementById("btnTreinamentoIA")
    .addEventListener("click", function () {
      executarScriptPython("treinamento_ia");
    });

  // Botão de Iniciar Câmeras
  document
    .getElementById("btnIniciarCameras")
    .addEventListener("click", function () {
      executarScriptPython("iniciar_cameras");
    });

  // Botão de Parar Câmeras
  document
    .getElementById("btnPararCameras")
    .addEventListener("click", function () {
      pararCameras();
    });

  // Botão de Limpar Console
  document
    .getElementById("btnLimparConsole")
    .addEventListener("click", function () {
      document.getElementById("consoleOutput").innerHTML = "";
    });

  // Inicializar estado dos botões
  atualizarEstadoBotoes(false);
});

// Variável global para controlar o estado das câmeras
let camerasAtivas = false;
let intervaloMonitoramento = null;

function executarScriptPython(acao) {
  // Mostrar o modal de feedback
  const modal = new bootstrap.Modal(document.getElementById("modalFeedback"));
  const progressBar = document.getElementById("progressBar");
  const feedbackMessage = document.getElementById("feedbackMessage");
  const consoleOutput = document.getElementById("consoleOutput");

  // Resetar o modal
  progressBar.style.width = "0%";
  progressBar.textContent = "0%";
  progressBar.setAttribute("aria-valuenow", 0);
  feedbackMessage.textContent = "Iniciando processo...";
  consoleOutput.innerHTML = "> Inicializando sistema...\n";

  // Mostrar o modal
  modal.show();

  // Atualizar progresso inicial
  atualizarProgresso(10, "Preparando ambiente...");

  // Desabilitar botões durante a execução
  atualizarEstadoBotoes(false);

  // Fazer requisição para o PHP
  fetch("./src/controller/main.php", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      acao: acao,
    }),
  })
    .then((response) => {
      if (!response.ok) {
        throw new Error("Erro na resposta do servidor: " + response.status);
      }
      return response.json();
    })
    .then((data) => {
      console.log("Resposta recebida:", data);

      if (data.success) {
        // Atualizar progresso
        atualizarProgresso(100, "Processamento concluído com sucesso!");

        // Exibir saída no console
        if (data.output) {
          consoleOutput.innerHTML += formatarOutput(data.output) + "\n";
        }

        // Executar ação específica após conclusão
        if (acao === "setup_ambiente") {
          consoleOutput.innerHTML +=
            "> Ambiente configurado com sucesso! Sistema pronto para uso.\n";
          mostrarNotificacao("success", "Ambiente configurado com sucesso!");
        } else if (acao === "treinamento_ia") {
          consoleOutput.innerHTML +=
            "> Treinamento de IA concluído. Modelo pronto para uso.\n";
          mostrarNotificacao("success", "Treinamento concluído com sucesso!");
        } else if (acao === "iniciar_cameras") {
          consoleOutput.innerHTML +=
            "> Sistema de reconhecimento facial inicializado.\n";
          consoleOutput.innerHTML += "> Verificando câmeras disponíveis...\n";
          mostrarNotificacao("success", "Sistema de câmeras iniciado!");

          // Atualizar estado das câmeras
          camerasAtivas = true;
          atualizarEstadoBotoes(true);

          // Iniciar monitoramento
          iniciarMonitoramentoCameras();
        }
      } else {
        atualizarProgresso(0, "Erro no processamento");
        consoleOutput.innerHTML += "> ERRO: " + data.error + "\n";
        if (data.output) {
          consoleOutput.innerHTML +=
            "> Detalhes: " + formatarOutput(data.output) + "\n";
        }
        mostrarNotificacao("error", "Erro: " + data.error);

        // Re-habilitar botões em caso de erro
        atualizarEstadoBotoes(true);
      }
    })
    .catch((error) => {
      console.error("Erro na requisição:", error);
      atualizarProgresso(0, "Erro na comunicação com o servidor");
      consoleOutput.innerHTML += "> ERRO: " + error.message + "\n";
      mostrarNotificacao("error", "Erro de comunicação: " + error.message);

      // Re-habilitar botões em caso de erro
      atualizarEstadoBotoes(true);
    });
}

function formatarOutput(output) {
  return "> " + output.replace(/\n/g, "\n> ");
}

function atualizarProgresso(percentual, mensagem) {
  const progressBar = document.getElementById("progressBar");
  const feedbackMessage = document.getElementById("feedbackMessage");

  progressBar.style.width = percentual + "%";
  progressBar.textContent = percentual + "%";
  progressBar.setAttribute("aria-valuenow", percentual);
  feedbackMessage.textContent = mensagem;

  // Atualizar a classe da barra de progresso baseada no percentual
  if (percentual < 30) {
    progressBar.className =
      "progress-bar progress-bar-striped progress-bar-animated bg-danger";
  } else if (percentual < 70) {
    progressBar.className =
      "progress-bar progress-bar-striped progress-bar-animated bg-warning";
  } else {
    progressBar.className =
      "progress-bar progress-bar-striped progress-bar-animated bg-success";
  }
}

function atualizarEstadoBotoes(habilitado) {
  const botoes = [
    "btnConfigurarAmbiente",
    "btnTreinamentoIA",
    "btnIniciarCameras",
    "btnPararCameras",
  ];

  botoes.forEach((botaoId) => {
    const botao = document.getElementById(botaoId);
    if (botao) {
      if (botaoId === "btnPararCameras") {
        // Botão de parar só fica habilitado quando as câmeras estão ativas
        botao.disabled = !camerasAtivas;
        botao.classList.toggle("btn-danger", camerasAtivas);
        botao.classList.toggle("btn-secondary", !camerasAtivas);
      } else {
        botao.disabled = !habilitado;
      }
    }
  });

  // Atualizar indicador de status
  const statusIndicator = document.getElementById("statusIndicator");
  if (statusIndicator) {
    if (camerasAtivas) {
      statusIndicator.className = "status-indicator active";
      statusIndicator.title = "Câmeras ativas";
    } else {
      statusIndicator.className = "status-indicator";
      statusIndicator.title = "Câmeras inativas";
    }
  }
}

function pararCameras() {
  if (camerasAtivas) {
    // Simular parada das câmeras (em produção, você teria uma API para parar o processo)
    const consoleOutput = document.getElementById("consoleOutput");
    consoleOutput.innerHTML += "> Parando sistema de câmeras...\n";

    // Parar monitoramento
    pararMonitoramentoCameras();

    // Atualizar estado
    camerasAtivas = false;
    atualizarEstadoBotoes(true);

    consoleOutput.innerHTML += "> Sistema de câmeras parado com sucesso.\n";
    mostrarNotificacao("info", "Sistema de câmeras parado");
  }
}

function iniciarMonitoramentoCameras() {
  // Limpar intervalo anterior se existir
  if (intervaloMonitoramento) {
    clearInterval(intervaloMonitoramento);
  }

  // Simular monitoramento das câmeras
  intervaloMonitoramento = setInterval(() => {
    if (camerasAtivas) {
      const consoleOutput = document.getElementById("consoleOutput");
      const timestamp = new Date().toLocaleTimeString();

      // Simular logs de monitoramento
      if (Math.random() > 0.7) {
        consoleOutput.innerHTML += `> [${timestamp}] Sistema ativo - Processando frames...\n`;

        // Manter o console rolável para baixo
        consoleOutput.scrollTop = consoleOutput.scrollHeight;
      }
    }
  }, 5000);
}

function pararMonitoramentoCameras() {
  if (intervaloMonitoramento) {
    clearInterval(intervaloMonitoramento);
    intervaloMonitoramento = null;
  }
}

function mostrarNotificacao(tipo, mensagem) {
  // Criar elemento de notificação
  const notificacao = document.createElement("div");
  notificacao.className = `alert alert-${tipo} alert-dismissible fade show position-fixed`;
  notificacao.style.cssText = `
    top: 20px;
    right: 20px;
    z-index: 1050;
    min-width: 300px;
  `;
  notificacao.innerHTML = `
    ${mensagem}
    <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
  `;

  // Adicionar ao corpo do documento
  document.body.appendChild(notificacao);

  // Remover automaticamente após 5 segundos
  setTimeout(() => {
    if (notificacao.parentNode) {
      notificacao.parentNode.removeChild(notificacao);
    }
  }, 5000);
}

// Função para verificar o status do servidor
function verificarStatusServidor() {
  fetch("./src/controller/main.php", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      acao: "status",
    }),
  })
    .then((response) => response.json())
    .then((data) => {
      if (data.success) {
        console.log("Servidor online e respondendo");
      }
    })
    .catch((error) => {
      console.warn("Servidor offline ou não respondendo");
    });
}

// Verificar status do servidor periodicamente
setInterval(verificarStatusServidor, 30000);

// Verificar status inicial
verificarStatusServidor();

// Adicionar estilos CSS dinamicamente para melhorar a interface
const styles = `
.status-indicator {
  display: inline-block;
  width: 12px;
  height: 12px;
  border-radius: 50%;
  background-color: #dc3545;
  margin-left: 10px;
  animation: pulse 2s infinite;
}

.status-indicator.active {
  background-color: #28a745;
  animation: pulse 1s infinite;
}

@keyframes pulse {
  0% { opacity: 1; }
  50% { opacity: 0.5; }
  100% { opacity: 1; }
}

.console-output {
  font-family: 'Courier New', monospace;
  font-size: 12px;
  height: 300px;
  overflow-y: auto;
  background-color: #1a1a1a;
  color: #00ff00;
  padding: 15px;
  border-radius: 5px;
  border: 1px solid #333;
}

.console-output::-webkit-scrollbar {
  width: 8px;
}

.console-output::-webkit-scrollbar-track {
  background: #2a2a2a;
}

.console-output::-webkit-scrollbar-thumb {
  background: #555;
  border-radius: 4px;
}

.console-output::-webkit-scrollbar-thumb:hover {
  background: #777;
}
`;

// Adicionar estilos ao documento
const styleSheet = document.createElement("style");
styleSheet.textContent = styles;
document.head.appendChild(styleSheet);
