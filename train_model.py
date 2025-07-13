import os
import cv2
import numpy as np
from PIL import Image, ImageEnhance
import mysql.connector
from mysql.connector import Error
import warnings
from datetime import datetime
import random
import sys
from collections import defaultdict
import glob
import unicodedata
import re


# Ignorar avisos específicos
warnings.filterwarnings("ignore", category=UserWarning)


def sanitizeFilename(name):
    """Sanitiza nomes de arquivos de forma consistente com cadastro.php"""
    try:
        # Converter para string se não for
        name = str(name)

        # Remover acentos e caracteres especiais
        name = (
            unicodedata.normalize("NFKD", name)
            .encode("ASCII", "ignore")
            .decode("ASCII")
        )

        # Substituir caracteres não alfanuméricos por underscore
        name = re.sub(r"[^a-zA-Z0-9-_]", "_", name)

        # Converter para minúsculas
        name = name.lower()

        # Remover underscores múltiplos
        name = re.sub(r"_+", "_", name)

        # Remover underscores no início e fim
        name = name.strip("_")

        # Limitar a 50 caracteres
        return name[:50]
    except Exception as e:
        print(f"Erro ao sanitizar nome: {str(e)}")
        return "unnamed"


class FaceTrainer:
    def __init__(self):
        # Configurações iniciais
        self.BASE_UPLOAD_PATH = "uploads/"
        self.TRAINING_DATA_PATH = "training_data/"
        self.MIN_IMAGES_PER_USER = 5  # Mínimo de imagens por usuário para treinamento
        self.VARIATIONS_PER_IMAGE = 15  # Variações geradas por imagem original

        # Carregar classificador de faces
        self.face_cascade = self._load_face_cascade()

        # Inicializar o reconhecedor facial
        self.recognizer = self._init_face_recognizer()
        if self.recognizer is None:
            print("\nERRO: Não foi possível inicializar o reconhecedor facial.")
            print("Por favor, instale o pacote correto com:")
            print("pip uninstall opencv-python")
            print("pip install opencv-contrib-python")
            sys.exit(1)

        # Configurações do MySQL
        self.db_config = {
            "host": "localhost",
            "user": "root",
            "password": "",
            "database": "reconhecimento_facial",
        }

    def _load_face_cascade(self):
        """Carrega o classificador de faces com tratamento de erro"""
        try:
            cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            if not os.path.exists(cascade_path):
                raise FileNotFoundError(
                    f"Arquivo do classificador não encontrado: {cascade_path}"
                )
            return cv2.CascadeClassifier(cascade_path)
        except Exception as e:
            print(f"ERRO ao carregar classificador de faces: {str(e)}")
            sys.exit(1)

    def _init_face_recognizer(self):
        """Inicializa o reconhecedor facial com suporte a múltiplas versões do OpenCV"""
        try:
            # Tentar a versão mais recente (OpenCV 4+)
            return cv2.face.LBPHFaceRecognizer_create(
                radius=2, neighbors=16, grid_x=8, grid_y=8, threshold=100
            )
        except AttributeError:
            try:
                # Tentar versão alternativa (OpenCV 3)
                return cv2.face.createLBPHFaceRecognizer(
                    radius=2, neighbors=16, grid_x=8, grid_y=8, threshold=100
                )
            except AttributeError:
                try:
                    # Tentar versão mais antiga
                    return cv2.createLBPHFaceRecognizer()
                except AttributeError:
                    return None

    def create_connection(self):
        """Cria conexão com o MySQL"""
        try:
            conn = mysql.connector.connect(**self.db_config)
            return conn
        except Error as e:
            print(f"ERRO ao conectar ao MySQL: {str(e)}")
            return None

    def verify_database(self):
        """Verifica se as tabelas necessárias existem no MySQL"""
        conn = self.create_connection()
        if conn is None:
            return False

        try:
            cursor = conn.cursor()

            # Verificar se todas as tabelas necessárias existem
            required_tables = ["cadastros", "imagens_cadastro", "treinamentos"]
            for table in required_tables:
                cursor.execute(
                    """
                    SELECT TABLE_NAME 
                    FROM INFORMATION_SCHEMA.TABLES 
                    WHERE TABLE_SCHEMA = %s AND TABLE_NAME = %s
                    """,
                    (self.db_config["database"], table),
                )
                if not cursor.fetchone():
                    print(f"ERRO: Tabela '{table}' não encontrada")
                    return False

            return True

        except Error as e:
            print(f"ERRO ao verificar tabelas: {str(e)}")
            return False
        finally:
            if conn.is_connected():
                conn.close()

    def get_image_path(self, db_path):
        """Converte caminho do banco para caminho completo no sistema de arquivos"""
        full_path = os.path.join(self.BASE_UPLOAD_PATH, db_path)
        if not os.path.exists(full_path):
            raise FileNotFoundError(f"Imagem não encontrada: {full_path}")
        return full_path

    def apply_random_transformation(self, image):
        """Aplica transformações aleatórias na imagem"""
        # Converter para PIL Image para algumas transformações
        pil_img = Image.fromarray(image)

        # Lista de transformações possíveis
        transformations = [
            lambda img: img,  # Nenhuma transformação
            lambda img: img.transpose(Image.FLIP_LEFT_RIGHT),
            lambda img: img.rotate(random.randint(-15, 15)),
            lambda img: ImageEnhance.Brightness(img).enhance(random.uniform(0.7, 1.3)),
            lambda img: ImageEnhance.Contrast(img).enhance(random.uniform(0.7, 1.3)),
            lambda img: ImageEnhance.Sharpness(img).enhance(random.uniform(0.7, 1.3)),
        ]

        # Aplicar 2-3 transformações aleatórias
        num_transformations = random.randint(2, 3)
        for _ in range(num_transformations):
            transform = random.choice(transformations)
            pil_img = transform(pil_img)

        # Converter de volta para numpy array
        return np.array(pil_img)

    def generate_variations(self, face_img, variations=20):
        """Gera variações da imagem facial para treinamento"""
        augmented_faces = []

        for i in range(variations):
            try:
                # Aplicar transformações aleatórias
                augmented = self.apply_random_transformation(face_img)

                # Adicionar ruído gaussiano (50% de chance)
                if random.random() > 0.5:
                    noise = np.random.normal(
                        0, random.uniform(0.5, 2), augmented.shape
                    ).astype(np.uint8)
                    augmented = cv2.add(augmented, noise)

                augmented_faces.append(augmented)
            except Exception as e:
                print(f"Erro ao gerar variação {i+1}: {str(e)}")

        return augmented_faces

    def save_training_image(
        self, cadastro_id, nome, sobrenome, face_img, variation_idx=None
    ):
        """Salva a imagem processada na pasta de treinamento"""
        try:
            # Criar nome da pasta usando nome e sobrenome
            user_folder_name = f"{nome}_{sobrenome}".replace(" ", "_").lower()
            user_folder = os.path.join(self.TRAINING_DATA_PATH, user_folder_name)
            os.makedirs(user_folder, exist_ok=True)

            # Gerar nome único para a imagem
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            if variation_idx is not None:
                img_filename = f"{user_folder_name}_{timestamp}_v{variation_idx}.jpg"
            else:
                img_filename = f"{user_folder_name}_{timestamp}.jpg"

            img_path = os.path.join(user_folder, img_filename)

            # Salvar a imagem
            cv2.imwrite(img_path, face_img)
            return img_path

        except Exception as e:
            print(f"Erro ao salvar imagem de treinamento: {str(e)}")
            return None

    def register_training(self, cadastro_id, nome, sobrenome, img_path):
        """Registra o treinamento na tabela treinamentos"""
        conn = self.create_connection()
        if conn is None:
            return False

        try:
            cursor = conn.cursor()

            # Converter caminho absoluto para relativo
            relative_path = os.path.relpath(img_path, start=self.TRAINING_DATA_PATH)

            cursor.execute(
                """
                INSERT INTO treinamentos 
                (cadastro_id, nome, sobrenome, caminho_imagem, data_processamento)
                VALUES (%s, %s, %s, %s, %s)
                """,
                (cadastro_id, nome, sobrenome, relative_path, datetime.now()),
            )
            conn.commit()
            return True

        except Error as e:
            print(f"Erro ao registrar treinamento: {str(e)}")
            return False
        finally:
            if conn.is_connected():
                conn.close()

    def get_user_images_from_db_and_fs(self):
        """Obtém todas as imagens dos usuários do banco de dados e do sistema de arquivos"""
        conn = self.create_connection()
        if conn is None:
            return None

        try:
            cursor = conn.cursor(dictionary=True)

            # Buscar todos os usuários cadastrados
            cursor.execute(
                """
                SELECT id, nome, sobrenome 
                FROM cadastros
                ORDER BY id
                """
            )
            users = cursor.fetchall()

            if not users:
                print("Nenhum usuário encontrado no banco de dados.")
                return None

            # Agrupar imagens por usuário
            user_images = defaultdict(list)

            for user in users:
                # Buscar imagens do banco de dados
                cursor.execute(
                    """
                    SELECT id, caminho_imagem 
                    FROM imagens_cadastro 
                    WHERE cadastro_id = %s
                    """,
                    (user["id"],),
                )
                db_images = cursor.fetchall()

                # Adicionar imagens do banco de dados
                for img in db_images:
                    user_images[user["id"]].append(
                        {
                            "id": img["id"],
                            "path": img["caminho_imagem"],
                            "nome": user["nome"],
                            "sobrenome": user["sobrenome"],
                            "source": "db",
                        }
                    )

                # Buscar imagens adicionais do sistema de arquivos
                user_folder_name = f"{sanitizeFilename(user['nome'])}_{sanitizeFilename(user['sobrenome'])}"
                user_folder = os.path.join(self.BASE_UPLOAD_PATH, user_folder_name)

                # Verificar se existe pasta do usuário
                if os.path.exists(user_folder):
                    # Procurar por subpastas (caso existam múltiplas pastas para o mesmo usuário)
                    user_subfolders = glob.glob(os.path.join(user_folder, "*/")) + [
                        user_folder + "/"
                    ]

                    for subfolder in user_subfolders:
                        # Buscar todas as imagens JPEG/PNG na pasta
                        fs_images = (
                            glob.glob(os.path.join(subfolder, "*.jpg"))
                            + glob.glob(os.path.join(subfolder, "*.jpeg"))
                            + glob.glob(os.path.join(subfolder, "*.png"))
                        )

                        # Adicionar imagens do sistema de arquivos que não estão no banco
                        for img_path in fs_images:
                            relative_path = os.path.relpath(
                                img_path, start=self.BASE_UPLOAD_PATH
                            )

                            # Verificar se a imagem já está no banco
                            if not any(
                                img["path"] == relative_path
                                for img in user_images[user["id"]]
                            ):
                                user_images[user["id"]].append(
                                    {
                                        "id": None,  # Não tem ID pois não está no banco
                                        "path": relative_path,
                                        "nome": user["nome"],
                                        "sobrenome": user["sobrenome"],
                                        "source": "fs",
                                    }
                                )

            return user_images

        except Error as e:
            print(f"Erro ao buscar imagens do banco de dados: {str(e)}")
            return None
        finally:
            if conn.is_connected():
                conn.close()

    def process_user_images(self, user_id, user_data):
        """Processa todas as imagens de um usuário"""
        face_samples = []
        registered_paths = []

        # Contador de imagens processadas com sucesso
        processed_count = 0

        for img_data in user_data:
            try:
                full_path = self.get_image_path(img_data["path"])
                pil_image = Image.open(full_path).convert("L")
                image_np = np.array(pil_image, "uint8")

                # Detectar faces com parâmetros otimizados
                faces = self.face_cascade.detectMultiScale(
                    image_np,
                    scaleFactor=1.05,
                    minNeighbors=6,
                    minSize=(40, 40),
                    flags=cv2.CASCADE_SCALE_IMAGE,
                )

                if len(faces) == 0:
                    print(f"Nenhuma face detectada em {img_data['path']}")
                    continue

                # Processar cada face encontrada na imagem
                for x, y, w, h in faces:
                    face_img = image_np[y : y + h, x : x + w]

                    # Padronizar tamanho e equalizar histograma
                    face_img = self._standardize_face_image(face_img)

                    # Salvar imagem original
                    saved_path = self.save_training_image(
                        user_id, img_data["nome"], img_data["sobrenome"], face_img
                    )

                    if saved_path:
                        face_samples.append(face_img)
                        registered_paths.append(saved_path)
                        processed_count += 1

                        # Gerar variações apenas se tivermos poucas imagens originais
                        if len(user_data) < 10:
                            variations = self.generate_variations(
                                face_img, variations=self.VARIATIONS_PER_IMAGE
                            )

                            for idx, variation in enumerate(variations):
                                var_path = self.save_training_image(
                                    user_id,
                                    img_data["nome"],
                                    img_data["sobrenome"],
                                    variation,
                                    variation_idx=idx,
                                )

                                if var_path:
                                    face_samples.append(variation)
                                    registered_paths.append(var_path)

            except Exception as e:
                print(f"Erro ao processar imagem {img_data['path']}: {str(e)}")

        # Registrar todas as imagens no banco de dados
        if registered_paths and processed_count > 0:
            for path in registered_paths:
                self.register_training(
                    user_id, img_data["nome"], img_data["sobrenome"], path
                )

        return face_samples if processed_count > 0 else None

    def _standardize_face_image(self, face_img):
        """Padroniza a imagem do rosto para tamanho e qualidade consistentes"""
        # Redimensionar para tamanho fixo
        face_img = cv2.resize(face_img, (200, 200))

        # Equalizar histograma para melhorar contraste
        face_img = cv2.equalizeHist(face_img)

        # Aplicar suavização para reduzir ruído
        face_img = cv2.medianBlur(face_img, 3)

        return face_img

    def get_images_and_labels(self):
        """Obtém todas as imagens e labels para treinamento"""
        user_images = self.get_user_images_from_db_and_fs()
        if user_images is None:
            return None, None

        face_samples = []
        ids = []
        skipped_users = 0

        for user_id, images in user_images.items():
            print(f"\nProcessando usuário {user_id} ({len(images)} imagens)...")

            # Processar todas as imagens do usuário
            user_faces = self.process_user_images(user_id, images)

            if user_faces:
                face_samples.extend(user_faces)
                ids.extend([user_id] * len(user_faces))
            else:
                print(f"AVISO: Nenhuma face válida encontrada para usuário {user_id}")
                skipped_users += 1

        if not face_samples:
            print("\nNenhuma face válida encontrada para treinamento.")
            return None, None

        print(f"\nResumo do treinamento:")
        print(
            f"- Usuários processados: {len(user_images) - skipped_users}/{len(user_images)}"
        )
        print(f"- Total de faces para treinamento: {len(face_samples)}")
        print(f"- Média de faces por usuário: {len(face_samples)/len(user_images):.1f}")

        return face_samples, np.array(ids)

    def train_model(self):
        """Executa o treinamento do modelo"""
        print("\n=== Sistema de Treinamento para Reconhecimento Facial ===")

        # Verificar/Criar pastas necessárias
        os.makedirs(self.TRAINING_DATA_PATH, exist_ok=True)
        os.makedirs("model", exist_ok=True)

        # Verificar banco de dados
        if not self.verify_database():
            return False

        # Obter dados para treinamento
        faces, ids = self.get_images_and_labels()

        if faces is None or len(faces) == 0:
            print("\nNenhuma face válida encontrada para treinamento.")
            return False

        print(
            f"\nIniciando treinamento com {len(faces)} faces de {len(np.unique(ids))} usuários..."
        )

        try:
            self.recognizer.train(faces, ids)
            self.recognizer.save("model/trained_model.yml")

            print("\nTreinamento concluído com sucesso!")
            print(f"Modelo salvo em: model/trained_model.yml")
            print(f"Imagens de treinamento salvas em: {self.TRAINING_DATA_PATH}")
            return True

        except Exception as e:
            print(f"\nErro durante o treinamento: {str(e)}")
            return False


if __name__ == "__main__":
    # Verificar instalação do OpenCV antes de começar
    try:
        print("Versão do OpenCV instalada:", cv2.__version__)
        if not hasattr(cv2, "face"):
            print("\nERRO: Módulo 'face' não encontrado no OpenCV.")
            print("Você instalou o pacote errado. Execute:")
            print("pip uninstall opencv-python")
            print("pip install opencv-contrib-python")
            sys.exit(1)
    except Exception as e:
        print("ERRO ao verificar OpenCV:", str(e))
        sys.exit(1)

    trainer = FaceTrainer()

    if not os.path.exists(trainer.BASE_UPLOAD_PATH):
        print(f"ERRO: Pasta 'uploads' não encontrada em {trainer.BASE_UPLOAD_PATH}")
        exit(1)

    if not trainer.train_model():
        exit(1)
