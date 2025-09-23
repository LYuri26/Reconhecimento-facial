# arquivo: check_model.py
import pickle
import os
import numpy as np


def check_model():
    model_path = "model/deepface_model.pkl"

    if not os.path.exists(model_path):
        print("❌ Modelo não encontrado!")
        return

    with open(model_path, "rb") as f:
        model_data = pickle.load(f)

    print("=== INFORMAÇÕES DO MODELO ===")
    print(f"Versão: {model_data.get('model_info', {}).get('version', 'N/A')}")
    print(
        f"Data do treinamento: {model_data.get('model_info', {}).get('training_date', 'N/A')}"
    )
    print(f"Total de usuários: {len(model_data['embeddings_db'])}")

    print("\n=== USUÁRIOS NO MODELO ===")
    for user_id, user_data in model_data["embeddings_db"].items():
        print(f"ID: {user_id}")
        print(f"Nome: {user_data['nome']} {user_data['sobrenome']}")
        print(f"Embeddings: {len(user_data['embeddings'])}")
        print(f"Embedding médio shape: {np.array(user_data['embedding']).shape}")
        print("-" * 40)


if __name__ == "__main__":
    check_model()
