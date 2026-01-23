# RAG/rag_service.py

import os
import sys
import chromadb

# -----------------------------
# Ścieżki
# -----------------------------
BASE_DIR = os.path.dirname(__file__)
SRC_DIR = os.path.join(BASE_DIR, "src")
DB_PATH = os.path.join(BASE_DIR, "db")

if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

from embedder_local import Embedder

COLLECTION_NAME = "bmw_pl"
TOP_K = 5

# -----------------------------
# 1️⃣ Query do ChromaDB
# -----------------------------
def rag_query(query: str, top_k: int = TOP_K):
    """
    Zwraca raw results z ChromaDB albo None
    """
    try:
        client = chromadb.PersistentClient(path=DB_PATH)

        collections = [c.name for c in client.list_collections()]
        if COLLECTION_NAME not in collections:
            print(f"❌ Kolekcja '{COLLECTION_NAME}' nie istnieje.")
            return None

        collection = client.get_collection(COLLECTION_NAME)

        embedder = Embedder()
        query_embedding = embedder.embed([query])[0]

        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )

        if not results or not results.get("documents"):
            return None

        return results

    except Exception as e:
        print(f"❌ Błąd ChromaDB: {e}")
        return None


# -----------------------------
# 2️⃣ Budowanie kontekstu
# -----------------------------
def build_context(results) -> str:
    """
    Buduje czysty kontekst tekstowy z wyników Chroma
    """
    if not results or not results.get("documents"):
        return ""

    docs = results["documents"][0]
    metas = results["metadatas"][0]

    seen = set()
    context_chunks = []

    for doc, meta in zip(docs, metas):
        source = meta.get("source", "unknown")
        chunk_index = meta.get("chunk_index", "NA")
        chunk_id = f"{source}_{chunk_index}"

        if chunk_id in seen:
            continue

        seen.add(chunk_id)
        context_chunks.append(
            f"[Źródło: {source}, chunk {chunk_index}]\n{doc.strip()}"
        )

    return "\n\n".join(context_chunks)


# -----------------------------
# 3️⃣ Główna funkcja RAG (SELEKTYWNA)
# -----------------------------
def run_rag(query: str, strict: bool = False) -> str | None:
    """
    strict = True:
        - brak kontekstu → None (twarda odmowa)
    strict = False:
        - brak kontekstu → "" (LLM może odpowiedzieć sam)
    """

    results = rag_query(query)

    if not results:
        return None if strict else ""

    context = build_context(results)

    if not context.strip():
        return None if strict else ""

    return context
