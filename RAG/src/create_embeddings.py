# RAG/src/create_embeddings.py
from pathlib import Path
import json
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma

BASE_DIR = Path(__file__).resolve().parent.parent
DB_PATH = BASE_DIR / "db"
INPUT_FILE = BASE_DIR / "output/firecrawl_normalized.json"

def build_embeddings():
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    texts = [f"Model: {d['model']}, Engine: {d['engine']}, Power: {d['power']}, Year: {d['year']}" for d in data]

    vectorstore = Chroma(persist_directory=str(DB_PATH), embedding_function=OpenAIEmbeddings())
    vectorstore.add_texts(texts)
    vectorstore.persist()
    print(f"Embeddings saved to {DB_PATH}")

if __name__ == "__main__":
    build_embeddings()
