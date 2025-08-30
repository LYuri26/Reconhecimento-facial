// assets/js/main.js

document.addEventListener("DOMContentLoaded", function () {
  // Verificar se os botões existem antes de adicionar event listeners
  const btnConfigurarAmbiente = document.getElementById(
    "btnConfigurarAmbiente"
  );
  const btnTreinamentoIA = document.getElementById("btnTreinamentoIA");
  const btnIniciarCameras = document.getElementById("btnIniciarCameras");
  const btnPararCameras = document.getElementById("btnPararCameras");
  const btnLimparConsole = document.getElementById("btnLimparConsole");

  // Botão de Configurar Ambiente
  if (btnConfigurarAmbiente) {
    btnConfigurarAmbiente.addEventListener("click", function () {
      executarScriptPython("setup_ambiente");
    });
  }

  // Botão de Treinamento IA
  if (btnTreinamentoIA) {
    btnTreinamentoIA.addEventListener("click", function () {
      executarScriptPython("treinamento_ia");
    });
  }

  // Botão de Iniciar Câmeras
  if (btnIniciarCameras) {
    btnIniciarCameras.addEventListener("click", function () {
      executarScriptPython("iniciar_cameras");
    });
  }

  // Botão de Parar Câmeras
  if (btnPararCameras) {
    btnPararCameras.addEventListener("click", function () {
      pararCameras();
    });
  }

  // Botão de Limpar Console
  if (btnLimparConsole) {
    btnLimparConsole.addEventListener("click", function () {
      const consoleOutput = document.getElementById("consoleOutput");
      if (consoleOutput) {
        consoleOutput.innerHTML = "> Console limpo...\n";
      }
    });
  }

  // Inicializar estado dos botões - COMEÇAM HABILITADOS
  atualizarEstadoBotoes(true);
});

// Variável global para controlar o estado das câmeras
let camerasAtivas = false;
let intervaloMonitoramento = null;
let eventSource = null;

function executarScriptPython(acao) {
  // Mostrar o modal de feedback
  const modalElement = document.getElementById("modalFeedback");
  if (!modalElement) {
    console.error("Modal de feedback não encontrado");
    mostrarNotificacao("error", "Elemento modal não encontrado");
    return;
  }

  const modal = new bootstrap.Modal(modalElement);
  const progressBar = document.getElementById("progressBar");
  const feedbackMessage = document.getElementById("feedbackMessage");
  const consoleOutput = document.getElementById("consoleOutput");

  // Verificar se elementos existem
  if (!progressBar || !feedbackMessage || !consoleOutput) {
    console.error("Elementos do modal não encontrados");
    mostrarNotificacao("error", "Elementos do modal não configurados");
    return;
  }

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

  // Usar Server-Sent Events para tempo real
  iniciarConexaoTempoReal(
    acao,
    progressBar,
    feedbackMessage,
    consoleOutput,
    modal
  );
}

function iniciarConexaoTempoReal(
  acao,
  progressBar,
  feedbackMessage,
  consoleOutput,
  modal
) {
  // Fechar conexão anterior se existir
  if (eventSource) {
    eventSource.close();
  }

  // URL correta para o PHP - usando POST em vez de GET
  const url = "./src/controller/main.php";

  // Criar nova conexão SSE usando POST
  eventSource = new EventSource(`${url}?acao=${acao}&real_time=1`);

  eventSource.onmessage = function (event) {
    try {
      const data = JSON.parse(event.data);

      if (data.output) {
        // Adicionar saída ao console
        consoleOutput.innerHTML += data.output;
        consoleOutput.scrollTop = consoleOutput.scrollHeight;

        // Atualizar progresso baseado na saída
        if (data.output.includes("✓")) {
          const currentProgress = parseInt(
            progressBar.getAttribute("aria-valuenow")
          );
          atualizarProgresso(Math.min(currentProgress + 20, 100), data.output);
        }

        // Verificar se é mensagem de conclusão
        if (
          data.output.includes("concluído") ||
          data.output.includes("sucesso")
        ) {
          atualizarProgresso(100, "Processamento concluído com sucesso!");
        }
      }

      if (data.finished) {
        // Processo finalizado
        finalizarProcesso(acao, consoleOutput, modal);
      }
    } catch (e) {
      console.error("Erro ao processar evento:", e);
    }
  };

  eventSource.onerror = function (error) {
    console.error("Erro na conexão SSE:", error);
    atualizarProgresso(0, "Erro na comunicação");
    consoleOutput.innerHTML += "> ERRO: Conexão perdida com o servidor\n";
    mostrarNotificacao("error", "Erro de comunicação com o servidor");

    // Re-habilitar botões
    atualizarEstadoBotoes(true);

    // Fechar conexão
    if (eventSource) {
      eventSource.close();
      eventSource = null;
    }
  };
}

function finalizarProcesso(acao, consoleOutput, modal) {
  // Fechar conexão SSE
  if (eventSource) {
    eventSource.close();
    eventSource = null;
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
  }

  // Re-habilitar botões
  atualizarEstadoBotoes(true);
}

function formatarOutput(output) {
  return "> " + output.replace(/\n/g, "\n> ");
}

function atualizarProgresso(percentual, mensagem) {
  const progressBar = document.getElementById("progressBar");
  const feedbackMessage = document.getElementById("feedbackMessage");

  if (!progressBar || !feedbackMessage) return;

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
    // Simular parada das câmeras
    const consoleOutput = document.getElementById("consoleOutput");
    if (consoleOutput) {
      consoleOutput.innerHTML += "> Parando sistema de câmeras...\n";
    }

    // Parar monitoramento
    pararMonitoramentoCameras();

    // Atualizar estado
    camerasAtivas = false;
    atualizarEstadoBotoes(true);

    if (consoleOutput) {
      consoleOutput.innerHTML += "> Sistema de câmeras parado com sucesso.\n";
    }
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
      if (Math.random() > 0.7 && consoleOutput) {
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
  height: 150px;
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
