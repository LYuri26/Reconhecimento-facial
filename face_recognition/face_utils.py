import os
import cv2
import pygame
import time
import subprocess
import logging
from typing import Optional
from config import SECURITY_SETTINGS

# Configuração de logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("face_utils.log"), logging.StreamHandler()],
)


class FaceUtils:
    @staticmethod
    def play_sound(sound_file: str, duration: Optional[float] = None) -> bool:
        """Toca um arquivo de áudio com opção de duração limitada"""
        try:
            if not os.path.exists(sound_file):
                logging.warning(f"Arquivo de áudio não encontrado: {sound_file}")
                return False

            pygame.mixer.init()
            pygame.mixer.music.load(sound_file)
            pygame.mixer.music.play()

            if duration:
                time.sleep(duration)
                pygame.mixer.music.stop()

            logging.info(f"Áudio tocado: {sound_file}")
            return True

        except Exception as e:
            logging.error(f"Erro ao tocar áudio: {e}")
            return False

    @staticmethod
    def play_alarm() -> None:
        """Toca o alarme de segurança configurado"""
        if SECURITY_SETTINGS["play_alarm"]:
            sound_file = SECURITY_SETTINGS["alarm_sound"]
            duration = SECURITY_SETTINGS["alarm_duration"]

            if not FaceUtils.play_sound(sound_file, duration):
                logging.warning("Falha ao tocar alarme de segurança")

    @staticmethod
    def lock_screen() -> bool:
        """Bloqueia a tela do computador"""
        if not SECURITY_SETTINGS["lock_screen"]:
            return False

        try:
            if os.name == "nt":  # Windows
                subprocess.run(
                    ["rundll32.exe", "user32.dll,LockWorkStation"], check=True
                )
            else:  # Linux/Mac
                # Tenta vários métodos comuns de bloqueio
                lock_commands = [
                    ["gnome-screensaver-command", "--lock"],
                    ["xdg-screensaver", "lock"],
                    ["i3lock"],
                    ["loginctl", "lock-session"],
                ]

                for cmd in lock_commands:
                    try:
                        subprocess.run(cmd, check=True)
                        break
                    except (subprocess.CalledProcessError, FileNotFoundError):
                        continue
                else:
                    raise RuntimeError("Nenhum método de bloqueio disponível")

            logging.info("Tela bloqueada com sucesso")
            return True

        except subprocess.CalledProcessError as e:
            logging.error(f"Erro ao bloquear tela: {e}")
            return False
        except Exception as e:
            logging.error(f"Erro geral ao bloquear tela: {e}")
            return False

    @staticmethod
    def take_photo(camera_index: int = 0, output_path: str = None) -> Optional[str]:
        """Tira uma foto usando a webcam e salva no caminho especificado"""
        try:
            cap = cv2.VideoCapture(camera_index)
            if not cap.isOpened():
                logging.error("Não foi possível acessar a câmera")
                return None

            ret, frame = cap.read()
            cap.release()

            if not ret:
                logging.error("Falha ao capturar imagem da câmera")
                return None

            if output_path is None:
                output_path = os.path.join(
                    os.path.dirname(__file__),
                    "../temp",
                    f"capture_{int(time.time())}.jpg",
                )
                os.makedirs(os.path.dirname(output_path), exist_ok=True)

            cv2.imwrite(output_path, frame)
            logging.info(f"Foto salva em: {output_path}")
            return output_path

        except Exception as e:
            logging.error(f"Erro ao capturar foto: {e}")
            return None

    @staticmethod
    def draw_face_landmarks(image, landmarks, color=(0, 255, 0), thickness=1):
        """Desenha landmarks faciais na imagem"""
        try:
            for landmark in landmarks:
                for point in landmark:
                    cv2.circle(image, point, thickness, color, -1)
            return image
        except Exception as e:
            logging.error(f"Erro ao desenhar landmarks: {e}")
            return image

    @staticmethod
    def validate_image(image_path: str, min_face_size: int = 100) -> bool:
        """Valida se uma imagem contém pelo menos uma face do tamanho mínimo"""
        try:
            image = face_recognition.load_image_file(image_path)
            face_locations = face_recognition.face_locations(image)

            if not face_locations:
                return False

            # Verifica se alguma face atende ao tamanho mínimo
            for top, right, bottom, left in face_locations:
                if (right - left) >= min_face_size and (bottom - top) >= min_face_size:
                    return True

            return False

        except Exception as e:
            logging.error(f"Erro ao validar imagem {image_path}: {e}")
            return False

    @staticmethod
    def calculate_image_quality(image_path: str) -> float:
        """Calcula uma métrica simples de qualidade de imagem (0-1)"""
        try:
            image = cv2.imread(image_path)
            if image is None:
                return 0.0

            # Métrica simples baseada em variação de Laplacian (nitidez)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            fm = cv2.Laplacian(gray, cv2.CV_64F).var()

            # Normaliza para escala 0-1 (valores empíricos)
            quality = min(max(fm / 100.0, 0.0), 1.0)
            return quality

        except Exception as e:
            logging.error(f"Erro ao calcular qualidade da imagem: {e}")
            return 0.0


if __name__ == "__main__":
    # Testes das funções utilitárias
    print("Testando utilitários faciais...")

    # Teste de captura de foto
    photo_path = FaceUtils.take_photo()
    print(f"Foto capturada: {photo_path}")

    if photo_path:
        # Teste de validação de imagem
        is_valid = FaceUtils.validate_image(photo_path)
        print(f"Imagem válida: {is_valid}")

        # Teste de qualidade de imagem
        quality = FaceUtils.calculate_image_quality(photo_path)
        print(f"Qualidade da imagem: {quality:.2f}")

    # Teste de alarme
    print("Testando alarme...")
    FaceUtils.play_alarm()

    # Teste de bloqueio de tela (comentado por segurança)
    # print("Testando bloqueio de tela...")
    # FaceUtils.lock_screen()
