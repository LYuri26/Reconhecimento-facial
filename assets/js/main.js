// assets/js/main.js

document.addEventListener("DOMContentLoaded", function () {
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
});

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
          consoleOutput.innerHTML +=
            "> " + data.output.replace(/\n/g, "\n> ") + "\n";
        }

        // Executar ação específica após conclusão
        if (acao === "treinamento_ia") {
          consoleOutput.innerHTML +=
            "> Treinamento de IA concluído. Modelo pronto para uso.\n";
        } else if (acao === "iniciar_cameras") {
          consoleOutput.innerHTML +=
            "> Câmeras inicializadas. Sistema de reconhecimento ativo.\n";
        }
      } else {
        atualizarProgresso(0, "Erro no processamento");
        consoleOutput.innerHTML += "> ERRO: " + data.error + "\n";
        if (data.output) {
          consoleOutput.innerHTML +=
            "> Detalhes: " + data.output.replace(/\n/g, "\n> ") + "\n";
        }
      }
    })
    .catch((error) => {
      console.error("Erro na requisição:", error);
      atualizarProgresso(0, "Erro na comunicação com o servidor");
      consoleOutput.innerHTML += "> ERRO: " + error.message + "\n";
    });
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
    progressBar.className = "progress-bar bg-danger";
  } else if (percentual < 70) {
    progressBar.className = "progress-bar bg-warning";
  } else {
    progressBar.className = "progress-bar bg-success";
  }
}
