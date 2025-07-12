import os
import cv2
import numpy as np
import logging
from .utils import load_classifier

# Configura o logger
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)


class FaceRecognition:
    def __init__(self):
        """
        Inicializa os classificadores de rosto e olhos (Haar Cascade),
        e os reconhecedores faciais (LBPH e Eigen).
        """
        base_path = os.path.dirname(__file__)

        # Haar Cascades
        cascade_path = os.path.join(base_path, "cascade")
        self.face_cascade = cv2.CascadeClassifier(
            os.path.join(cascade_path, "haarcascade_frontalface_default.xml")
        )
        self.eye_cascade = cv2.CascadeClassifier(
            os.path.join(cascade_path, "haarcascade-eye.xml")
        )

        # Classificadores treinados
        classifier_path = os.path.join(base_path, "classifier")
        self.classifier_lbph = load_classifier(
            os.path.join(classifier_path, "classificadorLBPH.yml")
        )
        self.classifier_eigen = load_classifier(
            os.path.join(classifier_path, "classificadorEigen.yml")
        )

    def detect_faces(self, image_path: str) -> np.ndarray:
        """
        Detecta rostos em uma imagem.

        :param image_path: Caminho da imagem
        :return: Coordenadas dos rostos detectados
        """
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Não foi possível carregar a imagem: {image_path}")

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
        )
        return faces

    def recognize_face(
        self, image_path: str, db=None
    ) -> tuple[int, float] | tuple[None, float]:
        """
        Reconhece um rosto usando dois classificadores (LBPH e Eigen).

        :param image_path: Caminho da imagem
        :param db: Conexão com o banco de dados (não usado neste método)
        :return: (label, confianca) ou (None, 0)
        """
        try:
            img = cv2.imread(image_path)
            if img is None:
                logger.warning(f"Imagem não encontrada: {image_path}")
                return None, 0

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100)
            )

            if len(faces) == 0:
                return None, 0

            (x, y, w, h) = faces[0]
            face = gray[y : y + h, x : x + w]
            face = cv2.resize(face, (220, 220))
            face = cv2.equalizeHist(face)

            # Reconhecimento LBPH
            label_lbph, confidence_lbph = self.classifier_lbph.predict(face)

            # Reconhecimento Eigen
            label_eigen, confidence_eigen = self.classifier_eigen.predict(face)

            if confidence_lbph < 100 and confidence_eigen < 100:
                if label_lbph == label_eigen:
                    media = (confidence_lbph + confidence_eigen) / 2
                    return label_lbph, media
                else:
                    return (
                        (label_lbph, confidence_lbph)
                        if confidence_lbph < confidence_eigen
                        else (label_eigen, confidence_eigen)
                    )
            else:
                return None, 0

        except Exception as e:
            logger.error(f"Erro no reconhecimento facial: {str(e)}")
            return None, 0

    def extract_face_encodings(self, image_path: str) -> list[np.ndarray] | None:
        """
        Extrai os dados do rosto para treinamento do classificador.

        :param image_path: Caminho da imagem
        :return: Lista com vetor da imagem processada ou None
        """
        try:
            img = cv2.imread(image_path)
            if img is None:
                logger.warning(f"Imagem inválida: {image_path}")
                return None

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100)
            )

            if len(faces) == 0:
                return None

            (x, y, w, h) = faces[0]
            face = gray[y : y + h, x : x + w]
            face = cv2.resize(face, (220, 220))
            face = cv2.equalizeHist(face)

            return [face.flatten()]

        except Exception as e:
            logger.error(f"Erro ao extrair características faciais: {str(e)}")
            return None
