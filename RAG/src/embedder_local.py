# RAG/src/embedder_local.py

import os
from dotenv import load_dotenv
import cohere

# Wczytanie zmiennych środowiskowych z pliku .env
load_dotenv()

COHERE_API_KEY = os.getenv("COHERE_API_KEY")
if not COHERE_API_KEY:
    raise ValueError("❌ Brak klucza COHERE_API_KEY w pliku .env!")

class Embedder:
    """
    Klasa do generowania embeddingów za pomocą Cohere API
    """
    def __init__(self):
        self.client = cohere.Client(api_key=COHERE_API_KEY)

    def embed(self, texts: list[str]) -> list[list[float]]:
        """
        texts: lista stringów do embedowania
        return: lista embeddingów (list floatów)
        """
        if not texts:
            return []

        response = self.client.embed(
            model="embed-english-v2.0",  # możesz zmienić model jeśli chcesz
            texts=texts
        )
        return response.embeddings
