document.addEventListener("DOMContentLoaded", function () {
  // Preview das imagens selecionadas
  const inputImagens = document.getElementById("imagens");
  const previewContainer = document.getElementById("preview-container");

  if (inputImagens && previewContainer) {
    inputImagens.addEventListener("change", function (e) {
      previewContainer.innerHTML = "";
      const files = e.target.files;

      if (files.length < 1) {
        alert("Selecione pelo menos 1 imagem!");
        return;
      }

      for (let i = 0; i < files.length; i++) {
        const reader = new FileReader();
        reader.onload = function (event) {
          const img = document.createElement("img");
          img.src = event.target.result;
          img.className = "img-thumbnail";
          img.style.maxHeight = "150px";
          previewContainer.appendChild(img);
        };
        reader.readAsDataURL(files[i]);
      }
    });
  }

  // Botão de novo cadastro
  const btnNovoCadastro = document.getElementById("novo-cadastro");
  if (btnNovoCadastro) {
    btnNovoCadastro.addEventListener("click", function () {
      window.location.href = "/";
    });
  }
});
