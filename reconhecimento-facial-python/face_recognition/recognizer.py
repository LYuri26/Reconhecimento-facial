import cv2
import numpy as np
import os
import pickle
import time
from config import Config
from typing import List, Dict, Optional, Tuple

# Verifica disponibilidade de bibliotecas
try:
    import face_recognition

    FACE_RECOG_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ face_recognition não disponível: {str(e)}")
    FACE_RECOG_AVAILABLE = False

try:
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.preprocessing import LabelEncoder

    SKLEARN_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ scikit-learn não disponível: {str(e)}")
    SKLEARN_AVAILABLE = False


class FaceRecognizer:
    def __init__(self):
        self.config = Config()
        self.known_encodings: List[np.ndarray] = []
        self.known_names: List[str] = []
        self.known_ids: List[int] = []
        self.known_danger_levels: List[str] = []
        self.label_encoder = LabelEncoder() if SKLEARN_AVAILABLE else None
        self.knn_classifier: Optional[KNeighborsClassifier] = None
        self.recognition_threshold = self.config.FACE_RECOGNITION_THRESHOLD

    def load_known_faces(self, pessoas: List[Dict]) -> bool:
        """Carrega rostos conhecidos do banco de dados"""
        print("⏳ Carregando rostos conhecidos...")
        start_time = time.time()

        self.known_encodings = []
        self.known_names = []
        self.known_ids = []
        self.known_danger_levels = []

        if not FACE_RECOG_AVAILABLE:
            print("⚠️ face_recognition não disponível - usando reconhecimento básico")
            return False

        try:
            for pessoa in pessoas:
                full_path = self.config.get_full_path(pessoa.folder)

                if not os.path.exists(full_path):
                    print(f"⚠️ Pasta não encontrada: {full_path}")
                    continue

                for file in os.listdir(full_path):
                    if file.lower().endswith((".png", ".jpg", ".jpeg")):
                        img_path = os.path.join(full_path, file)
                        self._process_face_image(img_path, pessoa)

            if self.known_encodings:
                if SKLEARN_AVAILABLE:
                    self._train_classifier()
                elapsed = time.time() - start_time
                print(
                    f"✅ {len(self.known_encodings)} rostos carregados em {elapsed:.2f}s"
                )
                return True

            print("⚠️ Nenhum rosto válido encontrado")
            return False

        except Exception as e:
            print(f"❌ Erro ao carregar rostos: {str(e)}")
            return False

    def _process_face_image(self, img_path: str, pessoa: Dict):
        """Processa uma imagem de rosto e extrai características"""
        try:
            img = cv2.imread(img_path)
            if img is None:
                return

            # Convertendo de BGR (OpenCV) para RGB (face_recognition)
            rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Encontrar todas as faces na imagem
            face_locations = face_recognition.face_locations(rgb_img)
            if face_locations:
                # Pegar apenas a primeira face encontrada
                face_encoding = face_recognition.face_encodings(
                    rgb_img, face_locations
                )[0]

                self.known_encodings.append(face_encoding)
                self.known_names.append(pessoa.name)
                self.known_ids.append(pessoa.id)
                self.known_danger_levels.append(pessoa.danger_level)

        except Exception as e:
            print(f"⚠️ Erro ao processar {img_path}: {str(e)}")

    def _preprocess_image(self, img: np.ndarray) -> np.ndarray:
        """Aplica pré-processamento na imagem"""
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Equalize histogram
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        equalized = clahe.apply(gray)

        # Convert back to 3 channels
        return cv2.cvtColor(equalized, cv2.COLOR_GRAY2BGR)

    def _train_classifier(self):
        """Treina o classificador KNN"""
        try:
            if SKLEARN_AVAILABLE and self.known_encodings:
                self.knn_classifier = KNeighborsClassifier(
                    n_neighbors=3, metric="cosine", weights="distance"
                )
                encoded_labels = self.label_encoder.fit_transform(self.known_names)
                self.knn_classifier.fit(self.known_encodings, encoded_labels)
        except Exception as e:
            print(f"❌ Erro ao treinar classificador: {str(e)}")
            self.knn_classifier = None

    def recognize_faces(self, frame: np.ndarray, faces: List[Tuple]) -> List[Dict]:
        """Reconhece faces na imagem"""
        results = []

        if not DEEPFACE_AVAILABLE or not self.knn_classifier:
            return self._basic_recognition(faces)

        for x, y, w, h in faces:
            try:
                face_img = frame[y : y + h, x : x + w]
                face_img = self._preprocess_image(face_img)

                if DEEPFACE_AVAILABLE:
                    embedding = DeepFace.represent(
                        img_path=face_img,
                        model_name=self.config.RECOGNITION_MODEL,
                        enforce_detection=False,
                        detector_backend="skip",
                    )

                    if not embedding:
                        raise ValueError("Embedding não gerado")

                    result = self._process_embedding(
                        embedding[0]["embedding"], (x, y, w, h)
                    )
                    results.append(result)

            except Exception as e:
                print(f"⚠️ Erro no reconhecimento: {str(e)}")
                results.append(self._unknown_face((x, y, w, h)))

        # Otimiza periodicamente o classificador
        if time.time() - self.last_optimization_time > 3600:  # A cada hora
            self._optimize_classifier()
            self.last_optimization_time = time.time()

        return results

    def _process_embedding(self, embedding: np.ndarray, location: Tuple) -> Dict:
        """Processa um embedding para reconhecimento"""
        if not SKLEARN_AVAILABLE or not self.knn_classifier:
            return self._unknown_face(location)

        distances, indices = self.knn_classifier.kneighbors(
            [embedding], n_neighbors=min(3, len(self.known_encodings))
        )

        avg_distance = np.mean(distances)
        confidence = 1 - min(avg_distance / self.recognition_threshold, 1.0)

        if confidence > self.recognition_threshold:
            neighbor_labels = self.label_encoder.inverse_transform(indices[0])
            unique, counts = np.unique(neighbor_labels, return_counts=True)
            best_match = unique[np.argmax(counts)]
            match_idx = self.known_names.index(best_match)

            return {
                "location": (
                    location[0],
                    location[1],
                    location[0] + location[2],
                    location[1] + location[3],
                ),
                "name": self.known_names[match_idx],
                "confidence": float(confidence),
                "user_id": self.known_ids[match_idx],
                "danger_level": self.known_danger_levels[match_idx],
            }
        else:
            return self._unknown_face(location)

    def _unknown_face(self, location: Tuple) -> Dict:
        """Retorna um resultado para face desconhecida"""
        return {
            "location": (
                location[0],
                location[1],
                location[0] + location[2],
                location[1] + location[3],
            ),
            "name": "Desconhecido",
            "confidence": 0,
            "user_id": None,
            "danger_level": "BAIXO",
        }

    def _basic_recognition(self, faces: List[Tuple]) -> List[Dict]:
        """Método básico de reconhecimento quando DeepFace não está disponível"""
        return [self._unknown_face(face) for face in faces]

    def _optimize_classifier(self):
        """Otimiza o classificador periodicamente"""
        try:
            if SKLEARN_AVAILABLE and len(self.known_encodings) > 100:
                print("⏳ Otimizando classificador...")
                self.knn_classifier = KNeighborsClassifier(
                    n_neighbors=min(5, len(self.known_encodings) // 20),
                    metric="cosine",
                    weights="distance",
                )
                encoded_labels = self.label_encoder.transform(self.known_names)
                self.knn_classifier.fit(self.known_encodings, encoded_labels)
        except Exception as e:
            print(f"⚠️ Erro ao otimizar classificador: {str(e)}")

    def save_model(self, path: str = "models/face_recognition_model.pkl") -> bool:
        """Salva o modelo treinado"""
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "wb") as f:
                pickle.dump(
                    {
                        "encodings": self.known_encodings,
                        "names": self.known_names,
                        "ids": self.known_ids,
                        "danger_levels": self.known_danger_levels,
                        "label_encoder": self.label_encoder,
                        "classifier": self.knn_classifier,
                        "threshold": self.recognition_threshold,
                    },
                    f,
                )
            print(f"💾 Modelo salvo em {path}")
            return True
        except Exception as e:
            print(f"❌ Erro ao salvar modelo: {str(e)}")
            return False

    def load_model(self, path: str = "models/face_recognition_model.pkl") -> bool:
        """Carrega um modelo previamente treinado"""
        try:
            with open(path, "rb") as f:
                data = pickle.load(f)
                self.known_encodings = data["encodings"]
                self.known_names = data["names"]
                self.known_ids = data["ids"]
                self.known_danger_levels = data["danger_levels"]
                self.label_encoder = data["label_encoder"]
                self.knn_classifier = data["classifier"]
                self.recognition_threshold = data.get("threshold", 0.6)
            print(f"♻️ Modelo carregado de {path}")
            return True
        except Exception as e:
            print(f"❌ Erro ao carregar modelo: {str(e)}")
            return False
