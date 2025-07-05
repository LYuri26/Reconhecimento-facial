document.addEventListener("DOMContentLoaded", function () {
  // Elementos DOM
  const video = document.getElementById("video");
  const canvas = document.getElementById("canvas");
  const capturarBtn = document.getElementById("capturar");
  const iniciarCapturaBtn = document.getElementById("iniciarCaptura");
  const reiniciarBtn = document.getElementById("reiniciar");
  const fotoPreview = document.getElementById("fotoPreview");
  const formFotos = document.getElementById("formFotos");
  const fotosInput = document.getElementById("fotos");
  const totalFotosSpan = document.getElementById("totalFotos");
  const progressContainer = document.getElementById("progressContainer");
  const contadorSpan = document.getElementById("contador");
  const progressBar = document.getElementById("progressBar");
  const btnFinalizar = document.getElementById("btnFinalizar");

  // Variáveis de estado
  let fotosCapturadas = [];
  let intervaloCaptura;
  let capturaAtiva = false;
  let streamAtivo = null;
  const TOTAL_FOTOS = 20; // Total de fotos a serem capturadas

  // Iniciar câmera
  function iniciarCamera() {
    if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
      navigator.mediaDevices
        .getUserMedia({
          video: {
            width: { ideal: 1280 },
            height: { ideal: 720 },
            facingMode: "user",
          },
        })
        .then(function (stream) {
          streamAtivo = stream;
          video.srcObject = stream;
          video.play();
        })
        .catch(function (error) {
          console.error("Erro ao acessar a câmera: ", error);
          alert(
            "Não foi possível acessar a câmera. Por favor, verifique as permissões."
          );
        });
    }
  }

  // Encerrar câmera
  function encerrarCamera() {
    if (streamAtivo) {
      streamAtivo.getTracks().forEach((track) => track.stop());
      streamAtivo = null;
    }
  }

  // Capturar foto
  function capturarFoto() {
    if (fotosCapturadas.length >= TOTAL_FOTOS) {
      alert(`Máximo de ${TOTAL_FOTOS} fotos atingido`);
      return;
    }

    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    canvas.getContext("2d").drawImage(video, 0, 0, canvas.width, canvas.height);

    const imageData = canvas.toDataURL("image/jpeg", 0.8);
    fotosCapturadas.push(imageData);

    atualizarPreview();
  }

  // Atualizar preview
  function atualizarPreview() {
    if (fotosCapturadas.length > 0) {
      const ultimaFoto = fotosCapturadas[fotosCapturadas.length - 1];
      fotoPreview.innerHTML = `
                <img src="${ultimaFoto}" class="img-fluid" alt="Foto capturada">
                <div class="photo-counter">${fotosCapturadas.length}</div>
            `;

      // Atualizar progresso
      const progresso = Math.round(
        (fotosCapturadas.length / TOTAL_FOTOS) * 100
      );
      progressBar.style.width = `${progresso}%`;
      contadorSpan.textContent = fotosCapturadas.length;

      // Habilitar botão de finalizar se tiver pelo menos 10 fotos
      if (fotosCapturadas.length >= 10) {
        btnFinalizar.disabled = false;
      }
    } else {
      fotoPreview.innerHTML =
        '<p class="text-muted mb-0">Nenhuma foto capturada</p>';
      btnFinalizar.disabled = true;
      progressBar.style.width = "0%";
    }
    totalFotosSpan.textContent = fotosCapturadas.length;
  }

  // Iniciar captura automática
  function iniciarCapturaAutomatica() {
    if (fotosCapturadas.length >= TOTAL_FOTOS) {
      alert(`Máximo de ${TOTAL_FOTOS} fotos atingido`);
      return;
    }

    capturaAtiva = true;
    progressContainer.style.display = "block";
    iniciarCapturaBtn.innerHTML =
      '<i class="fas fa-stop me-2"></i>Parar Captura';

    // Limpar qualquer intervalo existente
    if (intervaloCaptura) {
      clearInterval(intervaloCaptura);
    }

    // Calcular quantas fotos faltam
    const fotosRestantes = TOTAL_FOTOS - fotosCapturadas.length;
    let fotosCapturadasNestaSessao = 0;

    intervaloCaptura = setInterval(() => {
      if (
        fotosCapturadasNestaSessao >= fotosRestantes ||
        fotosCapturadas.length >= TOTAL_FOTOS
      ) {
        pararCapturaAutomatica();
        return;
      }

      capturarFoto();
      fotosCapturadasNestaSessao++;
    }, 500); // Intervalo de 500ms entre fotos
  }

  // Parar captura automática
  function pararCapturaAutomatica() {
    capturaAtiva = false;
    clearInterval(intervaloCaptura);
    iniciarCapturaBtn.innerHTML =
      '<i class="fas fa-video me-2"></i>Captura Automática';
  }

  // Event Listeners
  capturarBtn.addEventListener("click", capturarFoto);

  iniciarCapturaBtn.addEventListener("click", function () {
    if (capturaAtiva) {
      pararCapturaAutomatica();
    } else {
      iniciarCapturaAutomatica();
    }
  });

  reiniciarBtn.addEventListener("click", function () {
    fotosCapturadas = [];
    atualizarPreview();

    if (capturaAtiva) {
      pararCapturaAutomatica();
    }
  });

  formFotos.addEventListener("submit", function (e) {
    e.preventDefault();

    if (fotosCapturadas.length < 10) {
      alert("Por favor, capture pelo menos 10 fotos");
      return;
    }

    fotosInput.value = JSON.stringify(fotosCapturadas);
    this.submit();
  });

  // Iniciar a câmera quando a página carrega
  iniciarCamera();

  // Encerrar a câmera quando a página é fechada
  window.addEventListener("beforeunload", encerrarCamera);
});
