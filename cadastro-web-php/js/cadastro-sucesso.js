document.addEventListener("DOMContentLoaded", function () {
  // Pode adicionar animações ou interações específicas para esta página
  console.log("Página de sucesso carregada");

  // Exemplo: Animação de confirmação
  const checkIcon = document.querySelector(".fa-check-circle");
  if (checkIcon) {
    checkIcon.style.transform = "scale(0)";
    setTimeout(() => {
      checkIcon.style.transition = "transform 0.5s ease-out";
      checkIcon.style.transform = "scale(1)";
    }, 100);
  }
});
