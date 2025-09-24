// assets/js/main.js - ATUALIZADO (SEM CONTROLE DE CÂMERA ATIVA)

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

  // Inicializar estado dos botões
  atualizarEstadoBotoes(true);
});

// Variável global para controlar o estado das câmeras (REMOVER CONTROLE)
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
      mostrarNotificacao(
        "warning",
        "Conexão está demorando mais que o normal..."
      );

      if (consoleOutput) {
        consoleOutput.innerHTML +=
          "> AVISO: Aguardando resposta do servidor...\n";
        consoleOutput.scrollTop = consoleOutput.scrollHeight;
      }
    }
  }, 30000);

  eventSource.onopen = function () {
    console.log("Conexão SSE estabelecida com sucesso");
    clearTimeout(connectionTimeout);
  };

  eventSource.onmessage = function (event) {
    try {
      clearTimeout(connectionTimeout);

      const data = JSON.parse(event.data);

      // Verificar se há saída para o console
      if (data.output && consoleOutput) {
        // Formatar melhor a saída
        const timestamp = new Date().toLocaleTimeString();
        consoleOutput.innerHTML += `<span class="text-muted">[${timestamp}]</span> ${data.output}\n`;
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
          feedbackMessage.className = "text-danger";
        }

        mostrarNotificacao("error", "Erro durante a execução");

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
    }
  };

  eventSource.onerror = function (error) {
    console.error("Erro na conexão SSE:", error);
    clearTimeout(connectionTimeout);

    if (eventSource && eventSource.readyState === EventSource.CLOSED) {
      // Não mostrar erro se foi fechado intencionalmente
      if (!modal._element.classList.contains("show")) {
        return;
      }

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
  };

  // Adicionar event listener para fechamento do modal
  modal._element.addEventListener("hidden.bs.modal", function () {
    // Fechar conexão SSE quando o modal for fechado
    if (eventSource) {
      eventSource.close();
      eventSource = null;
    }
    clearTimeout(connectionTimeout);

    // Re-habilitar botões quando o modal for fechado
    atualizarEstadoBotoes(true);
  });
}

function getStageMessage(stage) {
  const stageMessages = {
    iniciando: "Iniciando sistema...",
    criando_venv: "Configurando ambiente...",
    instalando_dependencias: "Instalando dependências...",
    processando_imagens: "Processando imagens...",
    gerando_embeddings: "Gerando embeddings faciais...",
    salvando_modelo: "Salvando modelo treinado...",
    concluido: "Processamento concluído!",
    finalizado: "Processo finalizado",
  };

  return stageMessages[stage] || "Processando...";
}

function finalizarProcesso(acao, consoleOutput, modal, returnCode) {
  // Fechar conexão SSE
  if (eventSource) {
    eventSource.close();
    eventSource = null;
  }

  const timestamp = new Date().toLocaleTimeString();

  // Executar ação específica após conclusão
  if (acao === "treinamento_ia") {
    if (returnCode === 0) {
      consoleOutput.innerHTML += `<span class="text-muted">[${timestamp}]</span> <span class="text-success">Treinamento concluído com sucesso!</span>\n`;
      mostrarNotificacao("success", "Modelo treinado e pronto para uso!");
    } else {
      consoleOutput.innerHTML += `<span class="text-muted">[${timestamp}]</span> <span class="text-danger">Erro durante o treinamento</span>\n`;
      mostrarNotificacao("error", "Verifique os logs para detalhes");
    }

    // Fechar modal após 2 segundos
    setTimeout(() => {
      modal.hide();
      atualizarEstadoBotoes(true);
    }, 2000);
  } else if (acao === "iniciar_cameras") {
    if (returnCode === 0) {
      consoleOutput.innerHTML += `<span class="text-muted">[${timestamp}]</span> <span class="text-success">Reconhecimento facial executado com sucesso!</span>\n`;
      mostrarNotificacao("success", "Processo de reconhecimento concluído!");
    } else {
      consoleOutput.innerHTML += `<span class="text-muted">[${timestamp}]</span> <span class="text-warning">Reconhecimento facial finalizado</span>\n`;
      mostrarNotificacao("info", "Processo de reconhecimento finalizado");
    }

    // Fechar modal após 2 segundos para câmeras também
    setTimeout(() => {
      modal.hide();
      atualizarEstadoBotoes(true);
    }, 2000);
  }

  // Re-habilitar botões
  atualizarEstadoBotoes(true);
}

function atualizarProgresso(percentual, mensagem, stage) {
  const progressBar = document.getElementById("progressBar");
  const feedbackMessage = document.getElementById("feedbackMessage");

  if (!progressBar || !feedbackMessage) return;

  progressBar.style.width = percentual + "%";
  progressBar.textContent = percentual + "%";
  progressBar.setAttribute("aria-valuenow", percentual);
  feedbackMessage.textContent = mensagem;

  // Atualizar cores baseado no progresso
  if (stage === "erro") {
    progressBar.className = "progress-bar bg-danger";
    feedbackMessage.className = "text-danger";
  } else if (percentual === 100) {
    progressBar.className = "progress-bar bg-success";
    feedbackMessage.className = "text-success";
  } else {
    progressBar.className =
      "progress-bar progress-bar-striped progress-bar-animated bg-primary";
    feedbackMessage.className = "text-primary";
  }
}

// ATUALIZADA: Função simplificada para controlar estado dos botões
function atualizarEstadoBotoes(habilitado) {
  const btnTreinamentoIA = document.getElementById("btnTreinamentoIA");
  const btnIniciarCameras = document.getElementById("btnIniciarCameras");

  // Controle geral dos botões
  if (btnTreinamentoIA) {
    btnTreinamentoIA.disabled = !habilitado;
    btnTreinamentoIA.innerHTML = habilitado
      ? '<i class="fas fa-robot"></i> Treinar IA'
      : '<i class="fas fa-spinner fa-spin"></i> Processando...';
  }

  if (btnIniciarCameras) {
    btnIniciarCameras.disabled = !habilitado;
    btnIniciarCameras.innerHTML = habilitado
      ? '<i class="fas fa-camera"></i> Iniciar Reconhecimento'
      : '<i class="fas fa-spinner fa-spin"></i> Executando...';
  }

  // Remover qualquer indicador de status de câmera ativa
  const statusIndicator = document.getElementById("statusIndicator");
  if (statusIndicator) {
    statusIndicator.style.display = "none";
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
    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
  `;

  const icons = {
    success: '<i class="fas fa-check-circle me-2"></i>',
    error: '<i class="fas fa-exclamation-circle me-2"></i>',
    warning: '<i class="fas fa-exclamation-triangle me-2"></i>',
    info: '<i class="fas fa-info-circle me-2"></i>',
  };

  notificacao.innerHTML = `
    <div class="d-flex align-items-center">
      ${icons[tipo] || '<i class="fas fa-bell me-2"></i>'}
      <span class="flex-grow-1">${mensagem}</span>
      <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
    </div>
  `;

  // Adicionar ao corpo do documento
  document.body.appendChild(notificacao);

  // Remover automaticamente após 5 segundos (exceto errors)
  setTimeout(() => {
    if (notificacao.parentNode && tipo !== "error") {
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
.console-output {
  font-family: 'Fira Code', 'Courier New', monospace;
  font-size: 13px;
  height: 200px;
  overflow-y: auto;
  background: #1e1e1e;
  color: #00ff00;
  padding: 15px;
  border-radius: 10px;
  border: 2px solid #333;
  line-height: 1.4;
}

.console-output .text-muted {
  color: #6c757d !important;
}

.console-output .text-success {
  color: #28a745 !important;
  font-weight: bold;
}

.console-output .text-danger {
  color: #dc3545 !important;
  font-weight: bold;
}

.console-output .text-warning {
  color: #ffc107 !important;
  font-weight: bold;
}

.console-output::-webkit-scrollbar {
  width: 8px;
}

.console-output::-webkit-scrollbar-track {
  background: #2a2a2a;
  border-radius: 4px;
}

.console-output::-webkit-scrollbar-thumb {
  background: #555;
  border-radius: 4px;
}

.console-output::-webkit-scrollbar-thumb:hover {
  background: #777;
}

/* Progress bar melhorada */
.progress {
  height: 20px;
  border-radius: 10px;
  background: #e9ecef;
  overflow: hidden;
  box-shadow: inset 0 1px 3px rgba(0,0,0,0.2);
}

.progress-bar {
  border-radius: 10px;
  transition: width 0.6s ease;
  font-weight: 600;
  font-size: 12px;
}

/* Botões com ícones */
.btn i {
  margin-right: 8px;
}

.btn:disabled {
  opacity: 0.6;
  cursor: not-allowed;
}

.btn:disabled:hover {
  transform: none !important;
}

/* Notificações melhoradas */
.alert {
  border-radius: 10px;
  border: none;
  font-weight: 500;
}

.alert-success {
  background: linear-gradient(135deg, #28a745, #20c997);
  color: white;
}

.alert-error {
  background: linear-gradient(135deg, #dc3545, #c82333);
  color: white;
}

.alert-warning {
  background: linear-gradient(135deg, #ffc107, #e0a800);
  color: #000;
}

.alert-info {
  background: linear-gradient(135deg, #17a2b8, #138496);
  color: white;
}
`;

// Adicionar estilos ao documento
const styleSheet = document.createElement("style");
styleSheet.textContent = styles;
document.head.appendChild(styleSheet);
