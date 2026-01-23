from pathlib import Path
import json
import os
from dotenv import load_dotenv

import chromadb
import cohere

# ======================================================
# 🔧 .env
# ======================================================
BASE_DIR = Path(__file__).resolve().parent.parent
ENV_PATH = BASE_DIR / ".env"
load_dotenv(ENV_PATH)

COHERE_API_KEY = os.getenv("COHERE_API_KEY")
if not COHERE_API_KEY:
    raise RuntimeError("❌ COHERE_API_KEY nie został załadowany z .env")

print("✅ COHERE_API_KEY załadowany")

# ======================================================
# 📂 Ścieżki
# ======================================================
RAG_CHUNKS_DIR = BASE_DIR / "RAG" / "rag_chunks"
PROCESSED_FOLDER = max([d for d in RAG_CHUNKS_DIR.iterdir() if d.is_dir()], key=lambda d: d.stat().st_mtime)
ALL_MODELS_FILE = PROCESSED_FOLDER / "all_models.json"
CHROMA_DIR = BASE_DIR / "RAG" / "chroma_db"
CHROMA_DIR.mkdir(parents=True, exist_ok=True)

# ======================================================
# 🤖 Cohere
# ======================================================
co = cohere.Client(COHERE_API_KEY)

def embed_texts(texts: list[str]) -> list[list[float]]:
    response = co.embed(
        texts=texts,
        model="embed-multilingual-v3.0",
        input_type="search_document"
    )
    return response.embeddings

# ======================================================
# 🧠 Chroma Client z persystencją
# ======================================================
# W nowym API Chroma, persist_directory przekazujemy przy tworzeniu klienta
chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))

# ======================================================
# 🧹 Sanityzacja metadanych
# ======================================================
def sanitize_metadata(metadata: dict) -> dict:
    clean = {}
    for k, v in metadata.items():
        if v is None:
            continue
        if isinstance(v, (str, int, float, bool)):
            clean[k] = v
        else:
            clean[k] = str(v)
    return clean

# ======================================================
# 📥 Ładowanie chunków
# ======================================================
def load_chunks():
    collection = chroma_client.get_or_create_collection(
        name="rag_chunks"
        # persist_directory NIE jest tutaj potrzebny - klient już wie gdzie zapisywać
    )

    txt_files = list(PROCESSED_FOLDER.rglob("*.txt"))
    if not txt_files:
        print("⚠️ Brak chunków .txt")
        return

    BATCH_SIZE = 500
    total = 0

    for i in range(0, len(txt_files), BATCH_SIZE):
        batch = txt_files[i:i+BATCH_SIZE]
        texts, metadatas, ids = [], [], []

        for f in batch:
            texts.append(f.read_text(encoding="utf-8"))
            meta_file = f.with_name(f.stem + "_meta.json")
            metadata = json.loads(meta_file.read_text(encoding="utf-8")) if meta_file.exists() else {}
            metadatas.append(sanitize_metadata(metadata))
            ids.append(str(f.relative_to(PROCESSED_FOLDER)))

        embeddings = embed_texts(texts)

        collection.add(documents=texts, embeddings=embeddings, metadatas=metadatas, ids=ids)
        total += len(texts)

    print(f"✅ Załadowano {total} chunków")

# ======================================================
# 🚗 Ładowanie modeli BMW
# ======================================================
def load_models():
    if not ALL_MODELS_FILE.exists():
        print("⚠️ Brak all_models.json — pomijam modele")
        return

    collection = chroma_client.get_or_create_collection(
        name="bmw_models"
        # persist_directory NIE jest tutaj potrzebny
    )

    models = json.loads(ALL_MODELS_FILE.read_text(encoding="utf-8"))

    BATCH_SIZE = 200
    total = 0

    for i in range(0, len(models), BATCH_SIZE):
        batch = models[i:i+BATCH_SIZE]
        texts, metadatas, ids = [], [], []

        for idx, model in enumerate(batch, start=i):
            text = (
                f"Model: {model.get('model_name','')}\n"
                f"Typ nadwozia: {model.get('typ_nadwozia','')}\n"
                f"Moc: {model.get('moc_silnika','')}\n"
                f"Zasięg: {model.get('zasieg','')}\n"
                f"Napęd: {model.get('naped','')}"
            )
            texts.append(text)
            metadatas.append(sanitize_metadata(model))
            ids.append(f"bmw_model_{idx}")

        embeddings = embed_texts(texts)
        collection.add(documents=texts, embeddings=embeddings, metadatas=metadatas, ids=ids)
        total += len(texts)

    print(f"✅ Załadowano {total} modeli BMW")

# ======================================================
# ▶️ MAIN
# ======================================================
if __name__ == "__main__":
    print(f"🧠 [LOAD] CHROMA_DIR = {CHROMA_DIR}")
    load_chunks()
    load_models()
    print(f"\n🎉 Chroma DB gotowa w: {CHROMA_DIR}")