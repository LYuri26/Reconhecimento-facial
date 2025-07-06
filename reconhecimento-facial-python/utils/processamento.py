import cv2
import numpy as np
import face_recognition


class ProcessamentoImagem:
    def __init__(self, known_encodings=None, known_names=None):
        """
        Inicializa o processador de imagens com encodings conhecidos (opcional)

        Args:
            known_encodings (list): Lista de encodings faciais conhecidos
            known_names (list): Lista de nomes correspondentes aos encodings
        """
        self.known_encodings = known_encodings or []
        self.known_names = known_names or []
        self.scale_factor = 0.25  # Fator de redução para processamento

    def configurar_reconhecimento(self, known_encodings, known_names):
        """
        Configura os encodings e nomes conhecidos para reconhecimento

        Args:
            known_encodings (list): Lista de encodings faciais
            known_names (list): Lista de nomes correspondentes
        """
        self.known_encodings = known_encodings
        self.known_names = known_names

    def processar_frame(self, frame):
        """
        Processa um frame completo, incluindo detecção e reconhecimento facial

        Args:
            frame (numpy.ndarray): Frame de imagem BGR capturado da câmera

        Returns:
            tuple: (frame processado, lista de nomes reconhecidos)
        """
        # Reduz tamanho para processamento mais rápido
        small_frame = self._redimensionar_frame(frame)

        # Converte de BGR (OpenCV) para RGB (face_recognition)
        rgb_small_frame = self._converter_para_rgb(small_frame)

        # Detecta faces e reconhece
        face_locations, face_names = self._detectar_e_reconhecer(rgb_small_frame)

        # Desenha os resultados no frame original
        frame_processado = self._desenhar_resultados(frame, face_locations, face_names)

        return frame_processado, face_names

    def _redimensionar_frame(self, frame):
        """Redimensiona o frame para processamento mais rápido"""
        return cv2.resize(frame, (0, 0), fx=self.scale_factor, fy=self.scale_factor)

    def _converter_para_rgb(self, frame):
        """Converte de BGR (OpenCV) para RGB (face_recognition)"""
        return frame[:, :, ::-1]

    def _detectar_e_reconhecer(self, rgb_frame):
        """
        Detecta faces no frame e tenta reconhecer

        Args:
            rgb_frame (numpy.ndarray): Frame em formato RGB

        Returns:
            tuple: (lista de localizações, lista de nomes)
        """
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            # Verifica correspondência com encodings conhecidos
            matches = face_recognition.compare_faces(
                self.known_encodings, face_encoding
            )
            name = "Desconhecido"

            # Se encontrou correspondência, usa o primeiro
            if True in matches:
                first_match_index = matches.index(True)
                name = self.known_names[first_match_index]

            face_names.append(name)

        return face_locations, face_names

    def _desenhar_resultados(self, frame, face_locations, face_names):
        """
        Desenha retângulos e nomes no frame original

        Args:
            frame (numpy.ndarray): Frame original
            face_locations (list): Lista de localizações das faces
            face_names (list): Lista de nomes correspondentes

        Returns:
            numpy.ndarray: Frame com anotações
        """
        for (top, right, bottom, left), name in zip(face_locations, face_names):
            # Ajusta coordenadas para o tamanho original
            top = int(top / self.scale_factor)
            right = int(right / self.scale_factor)
            bottom = int(bottom / self.scale_factor)
            left = int(left / self.scale_factor)

            # Define cores diferentes para conhecidos/desconhecidos
            color = (0, 255, 0) if name != "Desconhecido" else (0, 0, 255)

            # Desenha retângulo ao redor do rosto
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)

            # Desenha label com nome
            cv2.rectangle(
                frame, (left, bottom - 35), (right, bottom), color, cv2.FILLED
            )
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(
                frame, name, (left + 6, bottom - 6), font, 0.8, (255, 255, 255), 1
            )

        return frame

    def extrair_encodings(self, image_path):
        """
        Extrai encodings faciais de uma imagem

        Args:
            image_path (str): Caminho para a imagem

        Returns:
            list: Lista de encodings faciais encontrados
        """
        image = face_recognition.load_image_file(image_path)
        face_locations = face_recognition.face_locations(image)
        return face_recognition.face_encodings(image, face_locations)
