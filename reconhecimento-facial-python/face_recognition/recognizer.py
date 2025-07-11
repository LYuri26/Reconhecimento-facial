# recognizer.py (ajustado para usar imagem temporária no reconhecimento e debug de vetores)
import cv2
import numpy as np
import os
import pickle
import time
import tempfile
from PIL import Image
from config import Config
from typing import List, Dict, Optional, Tuple

try:
    from deepface import DeepFace

    DEEPFACE_AVAILABLE = True
    print("✅ DeepFace carregado com sucesso")
except ImportError as e:
    print(f"❌ DeepFace não disponível: {str(e)}")
    DEEPFACE_AVAILABLE = False

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

    def load_known_faces(self, pessoas: List) -> bool:
        print("⏳ Carregando rostos conhecidos...")
        start_time = time.time()

        self.known_encodings = []
        self.known_names = []
        self.known_ids = []
        self.known_danger_levels = []

        if not DEEPFACE_AVAILABLE:
            print("⚠️ DeepFace não disponível - usando reconhecimento básico")
            return False

        try:
            for pessoa in pessoas:
                full_path = self.config.get_full_path(pessoa.folder)

                if not os.path.exists(full_path):
                    print(f"⚠️ Pasta não encontrada: {full_path}")
                    continue

                print(f"📁 Processando pasta de {pessoa.name}: {full_path}")

                for file in os.listdir(full_path):
                    if file.lower().endswith((".png", ".jpg", ".jpeg")):
                        img_path = os.path.join(full_path, file)
                        print(f"📷 Tentando processar imagem: {img_path}")
                        self._process_face_image(img_path, pessoa)

                print(
                    f"🧠 Rostos carregados para {pessoa.name}: {len(self.known_encodings)}"
                )

            if len(self.known_encodings) > 0:
                print(f"✅ Total de embeddings: {len(self.known_encodings)}")
                if SKLEARN_AVAILABLE:
                    self._train_classifier()
                elapsed = time.time() - start_time
                print(f"⏱ Tempo total: {elapsed:.2f}s")
                return True

            print("⚠️ Nenhum rosto válido encontrado")
            return False

        except Exception as e:
            print(f"❌ Erro ao carregar rostos: {str(e)}")
            return False

    def _process_face_image(self, img_path: str, pessoa) -> None:
        try:
            embedding = DeepFace.represent(
                img_path=img_path,
                model_name=self.config.RECOGNITION_MODEL,
                enforce_detection=False,
                detector_backend="opencv",
            )

            if embedding:
                self.known_encodings.append(embedding[0]["embedding"])
                self.known_names.append(pessoa.name)
                self.known_ids.append(pessoa.id)
                self.known_danger_levels.append(pessoa.danger_level)
                print(f"✅ Embedding gerado para {pessoa.name}")
                print(
                    f"[DEBUG] Embedding treino {pessoa.name}: {embedding[0]['embedding'][:5]}"
                )
            else:
                print(f"⚠️ Nenhum embedding retornado para {img_path}")
        except Exception as e:
            print(f"❌ Erro ao processar imagem {img_path}: {str(e)}")

    def recognize_faces(self, frame: np.ndarray, faces: List[Tuple]) -> List[Dict]:
        results = []

        if not DEEPFACE_AVAILABLE:
            return self._basic_recognition(faces)

        for x, y, w, h in faces:
            try:
                face_img = frame[y : y + h, x : x + w]
                face_img_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)

                with tempfile.NamedTemporaryFile(
                    suffix=".jpg", delete=False
                ) as tmp_file:
                    Image.fromarray(face_img_rgb).save(tmp_file.name)
                    embedding = DeepFace.represent(
                        img_path=tmp_file.name,
                        model_name=self.config.RECOGNITION_MODEL,
                        enforce_detection=False,
                        detector_backend="opencv",
                    )
                    os.remove(tmp_file.name)

                if embedding:
                    print(f"[DEBUG] Embedding ao vivo: {embedding[0]['embedding'][:5]}")
                    result = self._process_embedding(
                        embedding[0]["embedding"], (x, y, w, h)
                    )
                    print(f"🔍 Resultado: {result}")
                    results.append(result)
                else:
                    results.append(self._unknown_face((x, y, w, h)))

            except Exception as e:
                print(f"⚠️ Erro no reconhecimento: {str(e)}")
                results.append(self._unknown_face((x, y, w, h)))

        return results

    def _process_embedding(self, embedding: np.ndarray, location: Tuple) -> Dict:
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
        return [self._unknown_face(face) for face in faces]

    def _train_classifier(self):
        try:
            if SKLEARN_AVAILABLE and len(self.known_encodings) > 0:
                print("⏳ Treinando classificador...")
                self.knn_classifier = KNeighborsClassifier(
                    n_neighbors=3, metric="cosine", weights="distance"
                )
                encoded_labels = self.label_encoder.fit_transform(self.known_names)
                self.knn_classifier.fit(self.known_encodings, encoded_labels)
                print("✅ Classificador treinado com sucesso")
        except Exception as e:
            print(f"❌ Erro ao treinar classificador: {str(e)}")
