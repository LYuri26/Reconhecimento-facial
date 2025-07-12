import os
import sys
import cv2
import face_recognition
import pickle
import mysql.connector
import numpy as np
from datetime import datetime
import logging
from typing import Dict, List, Optional, Tuple
from pathlib import Path

# Configuração de caminhos
BASE_DIR = Path(__file__).parent.parent
sys.path.append(str(BASE_DIR))

try:
    from config import DB_CONFIG, MODEL_CONFIG, RECOGNITION_SETTINGS
except ImportError:
    logging.error("Erro ao importar configurações. Verifique o arquivo config.py")
    sys.exit(1)

# Configuração avançada de logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(BASE_DIR / "logs" / "face_trainer.log"),
        logging.StreamHandler(),
    ],
)


class FaceTrainer:
    def __init__(self):
        """Inicializa o sistema de treinamento facial"""
        self.model_path = Path(MODEL_CONFIG["model_path"])
        self.min_images = MODEL_CONFIG["min_images_per_person"]
        self.detection_method = MODEL_CONFIG["face_detection_method"]
        self.num_jitters = MODEL_CONFIG["num_jitters"]

        # Estruturas de dados
        self.encodings = []
        self.names = []
        self.ids = []
        self.image_paths = []

        # Configuração inicial
        self._setup_directories()

        logging.info("FaceTrainer inicializado com sucesso")

    def _setup_directories(self) -> None:
        """Cria os diretórios necessários"""
        try:
            # Diretório para modelos
            self.model_path.parent.mkdir(parents=True, exist_ok=True)

            # Diretório para logs
            (BASE_DIR / "logs").mkdir(exist_ok=True)

            logging.debug("Diretórios configurados com sucesso")
        except Exception as e:
            logging.error(f"Erro ao configurar diretórios: {e}")
            raise

    def get_db_connection(self) -> mysql.connector.MySQLConnection:
        """Estabelece e retorna uma conexão com o banco de dados"""
        try:
            # Cria conexão direta sem pool
            conn = mysql.connector.connect(
                host=DB_CONFIG["host"],
                user=DB_CONFIG["user"],
                password=DB_CONFIG["password"],
                database=DB_CONFIG["database"],
                raise_on_warnings=DB_CONFIG["raise_on_warnings"],
            )
            logging.debug("Conexão com o banco estabelecida")
            return conn
        except mysql.connector.Error as err:
            logging.error(f"Erro ao conectar ao banco: {err}")
            raise

    def load_persons_data(self) -> Dict[int, Dict]:
        """
        Carrega dados de pessoas e imagens do banco de dados
        Retorna um dicionário com estrutura:
        {
            person_id: {
                'name': 'Nome Completo',
                'images': [
                    {'path': '/caminho/da/imagem.jpg', 'id': 123},
                    ...
                ]
            },
            ...
        }
        """
        conn = None
        try:
            conn = self.get_db_connection()
            cursor = conn.cursor(dictionary=True)

            query = """
                SELECT 
                    c.id AS person_id,
                    CONCAT(c.nome, ' ', c.sobrenome) AS full_name,
                    i.id AS image_id,
                    i.caminho_imagem AS image_path
                FROM cadastros c
                JOIN imagens_cadastro i ON c.id = i.cadastro_id
                ORDER BY c.id, i.id
            """
            cursor.execute(query)

            persons = {}
            uploads_dir = BASE_DIR / "uploads"

            for row in cursor.fetchall():
                person_id = row["person_id"]
                if person_id not in persons:
                    persons[person_id] = {"name": row["full_name"], "images": []}

                img_path = uploads_dir / row["image_path"]
                if img_path.exists():
                    persons[person_id]["images"].append(
                        {"path": str(img_path), "id": row["image_id"]}
                    )
                else:
                    logging.warning(f"Imagem não encontrada: {img_path}")

            logging.info(f"Carregados dados de {len(persons)} pessoas do banco")
            return persons

        except mysql.connector.Error as err:
            logging.error(f"Erro no banco de dados: {err}")
            raise
        except Exception as e:
            logging.error(f"Erro inesperado ao carregar dados: {e}")
            raise
        finally:
            if conn and conn.is_connected():
                cursor.close()
                conn.close()

    def validate_persons(self, persons_data: Dict[int, Dict]) -> Dict[int, Dict]:
        """Filtra pessoas com número suficiente de imagens válidas"""
        valid_persons = {}
        skipped_persons = 0

        for person_id, data in persons_data.items():
            if len(data["images"]) >= self.min_images:
                valid_persons[person_id] = data
            else:
                skipped_persons += 1
                logging.warning(
                    f"Pessoa {data['name']} (ID: {person_id}) tem apenas "
                    f"{len(data['images'])} imagens (mínimo requerido: {self.min_images})"
                )

        if skipped_persons:
            logging.warning(f"Total de pessoas ignoradas: {skipped_persons}")

        logging.info(f"{len(valid_persons)} pessoas válidas para treinamento")
        return valid_persons

    def process_image(self, image_path: str) -> Optional[List[np.ndarray]]:
        """
        Processa uma única imagem e retorna os encodings faciais
        Retorna None em caso de falha
        """
        try:
            # Carrega a imagem
            image = face_recognition.load_image_file(image_path)

            # Detecta faces com o método especificado
            face_locations = face_recognition.face_locations(
                image, model=self.detection_method
            )

            if not face_locations:
                logging.debug(f"Nenhuma face detectada em {image_path}")
                return None

            # Gera encodings para cada face detectada
            encodings = face_recognition.face_encodings(
                image,
                known_face_locations=face_locations,
                num_jitters=self.num_jitters,
                model="small",
            )

            logging.debug(
                f"Processada {image_path} - {len(encodings)} face(s) encontrada(s)"
            )
            return encodings

        except Exception as e:
            logging.error(f"Erro ao processar {image_path}: {e}")
            return None

    def train_for_person(self, person_id: int, person_data: Dict) -> Tuple[int, int]:
        """
        Processa todas as imagens de uma pessoa específica
        Retorna: (num_images_processed, num_faces_found)
        """
        faces_count = 0
        processed_images = 0

        for img_data in person_data["images"]:
            encodings = self.process_image(img_data["path"])

            if encodings:
                # Adiciona cada encoding encontrado
                for encoding in encodings:
                    self.encodings.append(encoding)
                    self.names.append(person_data["name"])
                    self.ids.append(person_id)
                    self.image_paths.append(img_data["path"])
                    faces_count += 1

                processed_images += 1

        return processed_images, faces_count

    def train_model(self) -> bool:
        """Executa o pipeline completo de treinamento"""
        try:
            logging.info("Iniciando processo de treinamento")

            # 1. Carrega dados do banco
            persons_data = self.load_persons_data()
            if not persons_data:
                raise ValueError("Nenhum dado de pessoa encontrado no banco")

            # 2. Filtra pessoas com imagens suficientes
            valid_persons = self.validate_persons(persons_data)
            if not valid_persons:
                raise ValueError(
                    "Nenhuma pessoa com imagens suficientes para treinamento"
                )

            # 3. Processa imagens para cada pessoa
            total_images = 0
            total_faces = 0

            for person_id, person_data in valid_persons.items():
                logging.info(f"Processando {person_data['name']} (ID: {person_id})")

                processed, faces = self.train_for_person(person_id, person_data)
                total_images += processed
                total_faces += faces

                logging.info(
                    f"Resultado: {processed}/{len(person_data['images'])} imagens processadas, "
                    f"{faces} faces encontradas"
                )

            if total_faces == 0:
                raise ValueError(
                    "Nenhum rosto válido encontrado em todas as imagens processadas"
                )

            # 4. Salva o modelo treinado
            return self.save_model(total_images, total_faces, len(valid_persons))

        except Exception as e:
            logging.error(f"Falha no processo de treinamento: {e}")
            return False

    def save_model(
        self, total_images: int, total_faces: int, total_persons: int
    ) -> bool:
        """Salva o modelo treinado em disco"""
        try:
            model_data = {
                "encodings": self.encodings,
                "names": self.names,
                "ids": self.ids,
                "image_paths": self.image_paths,
                "metadata": {
                    "created_at": datetime.now().isoformat(),
                    "total_persons": total_persons,
                    "total_images": total_images,
                    "total_faces": total_faces,
                    "detection_method": self.detection_method,
                    "num_jitters": self.num_jitters,
                    "min_images": self.min_images,
                    "recognition_threshold": RECOGNITION_SETTINGS[
                        "recognition_threshold"
                    ],
                    "version": "1.0",
                },
            }

            with open(self.model_path, "wb") as f:
                pickle.dump(model_data, f, protocol=pickle.HIGHEST_PROTOCOL)

            logging.info(
                "Modelo treinado salvo com sucesso!\n"
                f"Local: {self.model_path}\n"
                f"Pessoas: {total_persons}\n"
                f"Imagens processadas: {total_images}\n"
                f"Faces detectadas: {total_faces}\n"
                f"Tamanho do modelo: {os.path.getsize(self.model_path) / 1024:.2f} KB"
            )
            return True

        except Exception as e:
            logging.error(f"Falha ao salvar modelo: {e}")
            return False


def main():
    try:
        trainer = FaceTrainer()

        if trainer.train_model():
            logging.info("Treinamento concluído com sucesso!")
            sys.exit(0)
        else:
            logging.error("O treinamento falhou")
            sys.exit(1)

    except KeyboardInterrupt:
        logging.info("Treinamento interrompido pelo usuário")
        sys.exit(1)
    except Exception as e:
        logging.error(f"Erro fatal: {str(e)}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
