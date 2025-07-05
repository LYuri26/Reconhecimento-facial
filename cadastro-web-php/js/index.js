document.addEventListener("DOMContentLoaded", function () {
  // Máscaras para CPF e Telefone
  $("#cpf").mask("000.000.000-00");
  $("#telefone").mask("(00) 00000-0000");

  // Validação do formulário
  $("#formDados").submit(function (e) {
    const cpf = $("#cpf").cleanVal();

    if (!validarCPF(cpf)) {
      alert("Por favor, insira um CPF válido");
      return false;
    }

    return true;
  });
});
