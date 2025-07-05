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
    if (fotosCapturadas.length >= 20) {
      alert("Máximo de 20 fotos atingido");
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

      // Habilitar botão de finalizar se tiver pelo menos 10 fotos
      if (fotosCapturadas.length >= 10) {
        btnFinalizar.disabled = false;
      }
    } else {
      fotoPreview.innerHTML =
        '<p class="text-muted mb-0">Nenhuma foto capturada</p>';
      btnFinalizar.disabled = true;
    }
    totalFotosSpan.textContent = fotosCapturadas.length;
  }

  // Iniciar captura automática
  function iniciarCapturaAutomatica() {
    progressContainer.style.display = "block";
    let fotosRestantes = 20 - fotosCapturadas.length;
    let fotosCapturadasNestaSessao = 0;
    const totalParaCapturar = Math.min(20, fotosRestantes);

    intervaloCaptura = setInterval(() => {
      if (fotosCapturadasNestaSessao >= totalParaCapturar) {
        pararCapturaAutomatica();
        iniciarCapturaBtn.innerHTML =
          '<i class="fas fa-video me-2"></i>Captura Automática';
        capturaAtiva = false;
        return;
      }

      capturarFoto();
      fotosCapturadasNestaSessao++;

      // Atualizar progresso
      const progresso = Math.round(
        (fotosCapturadasNestaSessao / totalParaCapturar) * 100
      );
      contadorSpan.textContent = fotosCapturadasNestaSessao;
      progressBar.style.width = `${progresso}%`;
    }, 500);
  }

  // Parar captura automática
  function pararCapturaAutomatica() {
    clearInterval(intervaloCaptura);
    progressContainer.style.display = "none";
    progressBar.style.width = "0%";
  }

  // Event Listeners
  capturarBtn.addEventListener("click", capturarFoto);

  iniciarCapturaBtn.addEventListener("click", function () {
    if (capturaAtiva) {
      pararCapturaAutomatica();
      iniciarCapturaBtn.innerHTML =
        '<i class="fas fa-video me-2"></i>Captura Automática';
    } else {
      if (fotosCapturadas.length >= 20) {
        alert("Máximo de 20 fotos atingido");
        return;
      }

      iniciarCapturaAutomatica();
      iniciarCapturaBtn.innerHTML =
        '<i class="fas fa-stop me-2"></i>Parar Captura';
    }
    capturaAtiva = !capturaAtiva;
  });

  reiniciarBtn.addEventListener("click", function () {
    fotosCapturadas = [];
    atualizarPreview();

    if (capturaAtiva) {
      pararCapturaAutomatica();
      iniciarCapturaBtn.innerHTML =
        '<i class="fas fa-video me-2"></i>Captura Automática';
      capturaAtiva = false;
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
