function initializeWebcam(videoElementId) {
  const video = document.getElementById(videoElementId);

  if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
    return navigator.mediaDevices
      .getUserMedia({ video: true })
      .then(function (stream) {
        video.srcObject = stream;
        return true;
      })
      .catch(function (error) {
        console.error("Erro ao acessar webcam:", error);
        return false;
      });
  } else {
    console.error("getUserMedia não suportado neste navegador");
    return false;
  }
}

function captureFrame(videoElementId, quality = 0.9) {
  const video = document.getElementById(videoElementId);
  const canvas = document.createElement("canvas");
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
  canvas.getContext("2d").drawImage(video, 0, 0, canvas.width, canvas.height);

  return new Promise((resolve) => {
    canvas.toBlob(resolve, "image/jpeg", quality);
  });
}
