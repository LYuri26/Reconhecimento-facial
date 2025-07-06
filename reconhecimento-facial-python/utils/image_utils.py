import cv2


def resize_image(image, size):
    """Redimensiona uma imagem para o tamanho especificado"""
    return cv2.resize(image, size)


def convert_to_grayscale(image):
    """Converte uma imagem para escala de cinza"""
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
