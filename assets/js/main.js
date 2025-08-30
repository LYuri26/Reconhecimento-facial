// assets/js/main.js

document.addEventListener("DOMContentLoaded", function () {
  // Verificar se os botões existem antes de adicionar event listeners
  const btnTreinamentoIA = document.getElementById("btnTreinamentoIA");
  const btnIniciarCameras = document.getElementById("btnIniciarCameras");
  const btnLimparConsole = document.getElementById("btnLimparConsole");

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
  atualizarProgresso(5, "Iniciando sistema...", "iniciando");

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
    eventSource = null;
  }

  // URL com timestamp para evitar cache
  const url = `./src/controller/main.php?acao=${acao}&real_time=1&t=${Date.now()}`;

  console.log(`Iniciando conexão SSE: ${url}`);

  eventSource = new EventSource(url);

  // Adicionar timeout para reconexão
  let connectionTimeout = setTimeout(() => {
    if (eventSource && eventSource.readyState !== EventSource.CLOSED) {
      console.error("Timeout na conexão SSE");
      eventSource.close();
      eventSource = null;
      mostrarNotificacao("error", "Timeout de conexão com o servidor");
      atualizarEstadoBotoes(true);

      // Adicionar mensagem no console
      if (consoleOutput) {
        consoleOutput.innerHTML +=
          "> ERRO: Timeout na conexão com o servidor\n";
        consoleOutput.scrollTop = consoleOutput.scrollHeight;
      }
    }
  }, 30000); // 30 segundos

  eventSource.onopen = function () {
    console.log("Conexão SSE estabelecida com sucesso");
    clearTimeout(connectionTimeout);

    // Resetar timeout para 5 minutos após conexão estabelecida
    connectionTimeout = setTimeout(() => {
      if (eventSource && eventSource.readyState !== EventSource.CLOSED) {
        console.error("Timeout após conexão estabelecida");
        eventSource.close();
        eventSource = null;
        mostrarNotificacao("error", "Processo está demorando muito");

        if (consoleOutput) {
          consoleOutput.innerHTML += "> AVISO: Processo ainda em execução...\n";
          consoleOutput.scrollTop = consoleOutput.scrollHeight;
        }
      }
    }, 300000); // 5 minutos
  };

  eventSource.onmessage = function (event) {
    try {
      clearTimeout(connectionTimeout); // Resetar timeout a cada mensagem

      const data = JSON.parse(event.data);

      // Verificar se há saída para o console
      if (data.output && consoleOutput) {
        consoleOutput.innerHTML += data.output + "\n";
        consoleOutput.scrollTop = consoleOutput.scrollHeight;
      }

      // Atualizar progresso baseado nos dados recebidos
      if (data.progress !== undefined && progressBar && feedbackMessage) {
        atualizarProgresso(
          data.progress,
          getStageMessage(data.stage),
          data.stage
        );
      }

      // Verificar se é mensagem de erro
      if (data.error) {
        console.error("Erro recebido:", data.error);
        if (feedbackMessage) {
          feedbackMessage.textContent = "Erro: " + data.error;
        }
        if (consoleOutput) {
          consoleOutput.innerHTML += "> ERRO: " + data.error + "\n";
          consoleOutput.scrollTop = consoleOutput.scrollHeight;
        }
        mostrarNotificacao("error", "Erro durante a execução: " + data.error);

        // Re-habilitar botões em caso de erro
        atualizarEstadoBotoes(true);

        // Fechar conexão
        if (eventSource) {
          eventSource.close();
          eventSource = null;
        }
      }

      // Verificar se é mensagem de conclusão
      if (data.finished) {
        console.log("Processo finalizado com código:", data.return_code);

        // Fechar conexão SSE
        if (eventSource) {
          eventSource.close();
          eventSource = null;
        }

        clearTimeout(connectionTimeout);

        // Executar ação específica após conclusão
        finalizarProcesso(acao, consoleOutput, modal, data.return_code);
      }
    } catch (e) {
      console.error("Erro ao processar evento SSE:", e);
      // Tentar processar como texto simples se não for JSON
      if (event.data && consoleOutput) {
        consoleOutput.innerHTML += "> " + event.data + "\n";
        consoleOutput.scrollTop = consoleOutput.scrollHeight;
      }
    }
  };

  eventSource.onerror = function (error) {
    console.error("Erro na conexão SSE:", error);
    clearTimeout(connectionTimeout);

    // Verificar estado da conexão
    if (eventSource) {
      console.log("Estado da conexão SSE:", eventSource.readyState);

      // Só tratar como erro se a conexão foi fechada inesperadamente
      if (eventSource.readyState === EventSource.CLOSED) {
        // Tentar reconectar após 3 segundos se o modal ainda estiver aberto
        setTimeout(() => {
          const modalElement = document.getElementById("modalFeedback");
          if (modalElement && modalElement.classList.contains("show")) {
            console.log("Tentando reconexão automática...");

            if (consoleOutput) {
              consoleOutput.innerHTML += "> Reconectando com servidor...\n";
              consoleOutput.scrollTop = consoleOutput.scrollHeight;
            }

            iniciarConexaoTempoReal(
              acao,
              progressBar,
              feedbackMessage,
              consoleOutput,
              modal
            );
          }
        }, 3000);
      }
    }
  };

  // Adicionar event listener para fechamento do modal
  const modalElement = document.getElementById("modalFeedback");
  if (modalElement) {
    modalElement.addEventListener("hidden.bs.modal", function () {
      // Fechar conexão SSE quando o modal for fechado
      if (eventSource) {
        eventSource.close();
        eventSource = null;
      }
      clearTimeout(connectionTimeout);
    });
  }
}

function getStageMessage(stage) {
  const stageMessages = {
    iniciando: "Iniciando sistema...",
    criando_venv: "Criando ambiente virtual...",
    instalando_dependencias: "Instalando dependências...",
    processando_imagens: "Processando imagens...",
    gerando_embeddings: "Gerando embeddings faciais...",
    salvando_modelo: "Salvando modelo treinado...",
    concluido: "Processamento concluído!",
    finalizado: "Processo finalizado",
    erro: "Erro durante o processo",
  };

  return stageMessages[stage] || "Processando...";
}

function finalizarProcesso(acao, consoleOutput, modal, returnCode) {
  // Fechar conexão SSE
  if (eventSource) {
    eventSource.close();
    eventSource = null;
  }

  // Executar ação específica após conclusão
  if (acao === "treinamento_ia") {
    if (returnCode === 0) {
      consoleOutput.innerHTML +=
        "> Treinamento de IA concluído. Modelo pronto para uso.\n";
      mostrarNotificacao("success", "Treinamento concluído com sucesso!");
    } else {
      consoleOutput.innerHTML +=
        "> Erro durante o treinamento. Verifique os logs.\n";
      mostrarNotificacao("error", "Falha no treinamento!");
    }
  } else if (acao === "iniciar_cameras") {
    if (returnCode === 0) {
      consoleOutput.innerHTML +=
        "> Sistema de reconhecimento facial inicializado.\n";
      consoleOutput.innerHTML += "> Verificando câmeras disponíveis...\n";
      mostrarNotificacao("success", "Sistema de câmeras iniciado!");

      // Atualizar estado das câmeras
      camerasAtivas = true;
    } else {
      consoleOutput.innerHTML +=
        "> Erro ao iniciar câmeras. Verifique os logs.\n";
      mostrarNotificacao("error", "Falha ao iniciar câmeras!");
    }
  }

  // Re-habilitar botões
  atualizarEstadoBotoes(true);
}

function formatarOutput(output) {
  return "> " + output.replace(/\n/g, "\n> ");
}

function atualizarProgresso(percentual, mensagem, stage) {
  const progressBar = document.getElementById("progressBar");
  const feedbackMessage = document.getElementById("feedbackMessage");

  if (!progressBar || !feedbackMessage) return;

  progressBar.style.width = percentual + "%";
  progressBar.textContent = percentual + "%";
  progressBar.setAttribute("aria-valuenow", percentual);
  feedbackMessage.textContent = mensagem;

  // Atualizar a classe da barra de progresso baseada no estágio
  if (stage === "erro") {
    progressBar.className = "progress-bar bg-danger";
  } else if (percentual < 30) {
    progressBar.className =
      "progress-bar progress-bar-striped progress-bar-animated bg-danger";
  } else if (percentual < 70) {
    progressBar.className =
      "progress-bar progress-bar-striped progress-bar-animated bg-warning";
  } else if (percentual < 100) {
    progressBar.className =
      "progress-bar progress-bar-striped progress-bar-animated bg-info";
  } else {
    progressBar.className = "progress-bar bg-success";
  }
}
function atualizarEstadoBotoes(habilitado) {
  const botoes = ["btnTreinamentoIA", "btnIniciarCameras"];

  botoes.forEach((botaoId) => {
    const botao = document.getElementById(botaoId);
    if (botao) {
      botao.disabled = !habilitado;
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
