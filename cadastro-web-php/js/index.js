document.addEventListener("DOMContentLoaded", function () {
  // Elementos do DOM
  const formCadastro = document.getElementById("formCadastro");
  const fotoInput = document.getElementById("foto");
  const preview = document.getElementById("preview");

  // Preview da imagem selecionada
  fotoInput.addEventListener("change", function (e) {
    const file = e.target.files[0];

    if (file) {
      // Verificar tamanho do arquivo (máx 5MB)
      if (file.size > 5 * 1024 * 1024) {
        alert("A imagem deve ter no máximo 5MB!");
        e.target.value = ""; // Limpa o input
        preview.style.display = "none";
        return;
      }

      // Verificar tipo do arquivo
      const validTypes = ["image/jpeg", "image/png", "image/jpg"];
      if (!validTypes.includes(file.type)) {
        alert("Formato de imagem inválido! Use JPG ou PNG.");
        e.target.value = "";
        preview.style.display = "none";
        return;
      }

      const reader = new FileReader();

      reader.onload = function (e) {
        preview.src = e.target.result;
        preview.style.display = "block";
      };

      reader.readAsDataURL(file);
    } else {
      preview.style.display = "none";
    }
  });

  // Validação do formulário
  formCadastro.addEventListener("submit", function (e) {
    e.preventDefault();

    // Validação dos campos
    const nome = document.getElementById("nome").value.trim();
    const sobrenome = document.getElementById("sobrenome").value.trim();
    const foto = fotoInput.files[0];

    if (!nome || !sobrenome) {
      alert("Nome e sobrenome são obrigatórios!");
      return false;
    }

    if (!foto) {
      alert("Por favor, selecione uma foto!");
      return false;
    }

    // Enviar via AJAX
    const formData = new FormData(formCadastro);

    fetch(formCadastro.action, {
      method: "POST",
      body: formData,
    })
      .then((response) => response.json())
      .then((data) => {
        if (data.success) {
          window.location.href = data.redirect;
        } else {
          alert(data.message || "Erro no cadastro");
        }
      })
      .catch((error) => {
        console.error("Error:", error);
        alert("Erro ao processar o cadastro");
      });
  });
});
