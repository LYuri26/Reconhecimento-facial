document.addEventListener("DOMContentLoaded", function () {
  const form = document.getElementById("formCadastro");
  const fileInput = document.getElementById("imagens");
  const previewContainer = document.getElementById("preview-container");
  let selectedFiles = [];

  // Adiciona marcação de campos obrigatórios
  document.querySelectorAll("[required]").forEach((el) => {
    const label = document.querySelector(`label[for="${el.id}"]`);
    if (label) {
      label.classList.add("required-field");
    }
  });

  // Preview de imagens selecionadas
  fileInput.addEventListener("change", function (e) {
    previewContainer.innerHTML = "";
    selectedFiles = Array.from(e.target.files);

    if (selectedFiles.length > 20) {
      mostrarErro(fileInput, "Você pode selecionar no máximo 20 imagens.");
      fileInput.value = "";
      selectedFiles = [];
      return;
    }

    selectedFiles.forEach((file, index) => {
      if (!file.type.match("image.*")) {
        mostrarErro(fileInput, "Apenas imagens são permitidas.");
        return;
      }

      const reader = new FileReader();
      reader.onload = function (event) {
        const previewDiv = document.createElement("div");
        previewDiv.className = "col-6 col-md-3 col-lg-2";

        const imgWrapper = document.createElement("div");
        imgWrapper.className = "image-preview";

        const img = document.createElement("img");
        img.src = event.target.result;

        const removeBtn = document.createElement("button");
        removeBtn.className = "remove-btn";
        removeBtn.innerHTML = "×";
        removeBtn.addEventListener("click", function (e) {
          e.preventDefault();
          selectedFiles.splice(index, 1);
          updateFileInput();
          previewContainer.removeChild(previewDiv);
        });

        imgWrapper.appendChild(img);
        imgWrapper.appendChild(removeBtn);
        previewDiv.appendChild(imgWrapper);
        previewContainer.appendChild(previewDiv);
      };
      reader.readAsDataURL(file);
    });
  });

  function updateFileInput() {
    const dataTransfer = new DataTransfer();
    selectedFiles.forEach((file) => dataTransfer.items.add(file));
    fileInput.files = dataTransfer.files;
  }

  form.addEventListener("submit", function (e) {
    e.preventDefault();

    if (validarFormulario()) {
      this.submit();
    }
  });

  function validarFormulario() {
    let isValid = true;

    // Limpa mensagens de erro anteriores
    document
      .querySelectorAll(".is-invalid")
      .forEach((el) => el.classList.remove("is-invalid"));
    document.querySelectorAll(".invalid-feedback").forEach((el) => el.remove());

    // Validação de nome
    const nome = document.getElementById("nome");
    if (nome.value.trim() === "") {
      mostrarErro(nome, "Por favor, insira seu nome.");
      isValid = false;
    }

    // Validação de sobrenome
    const sobrenome = document.getElementById("sobrenome");
    if (sobrenome.value.trim() === "") {
      mostrarErro(sobrenome, "Por favor, insira seu sobrenome.");
      isValid = false;
    }

    // Validação de imagens
    if (selectedFiles.length < 1) {
      mostrarErro(fileInput, "Selecione pelo menos 1 imagem.");
      isValid = false;
    } else if (selectedFiles.length > 20) {
      mostrarErro(fileInput, "Máximo de 20 imagens excedido.");
      isValid = false;
    }

    return isValid;
  }

  function mostrarErro(input, mensagem) {
    input.classList.add("is-invalid");
    const divErro = document.createElement("div");
    divErro.className = "invalid-feedback";
    divErro.textContent = mensagem;
    input.parentNode.appendChild(divErro);
  }
});
