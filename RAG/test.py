# Sprawdź czy:
# - Dane są poprawnie załadowane do ChromaDB
# - Embeddingi są generowane poprawnie
# - Zapytania zwracają wyniki

import chromadb
from chromadb.config import Settings

# Sprawdź liczbę dokumentów w kolekcji
client = chromadb.Client(Settings())
collection = client.get_or_create_collection("bmw_docs")
print(f"Dokumenty w kolekcji: {collection.count()}")

# Test zapytania
results = collection.query(
    query_texts=["wybór modelu BMW"],
    n_results=3
)
print("Wyniki zapytania:", results)