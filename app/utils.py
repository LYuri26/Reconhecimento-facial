import cv2
import numpy as np
import pickle
import os
from config import Config


def train_classifier(images, labels, output_path, classifier_type="LBPH"):
    """Treina um classificador com as imagens e labels fornecidos"""
    if classifier_type == "LBPH":
        classifier = cv2.face.LBPHFaceRecognizer_create()
    else:  # Eigen
        classifier = cv2.face.EigenFaceRecognizer_create()

    classifier.train(images, np.array(labels))
    classifier.save(output_path)
    return classifier


def load_classifier(classifier_path):
    """Carrega um classificador treinado"""
    if not os.path.exists(classifier_path):
        raise FileNotFoundError(f"Classifier file not found: {classifier_path}")

    if classifier_path.endswith(".pkl"):
        with open(classifier_path, "rb") as f:
            return pickle.load(f)
    else:
        # Para classificadores OpenCV
        classifier = cv2.face.LBPHFaceRecognizer_create()
        classifier.read(classifier_path)
        return classifier


def save_classifier(classifier, output_path):
    """Salva um classificador treinado"""
    if output_path.endswith(".pkl"):
        with open(output_path, "wb") as f:
            pickle.dump(classifier, f)
    else:
        classifier.save(output_path)
