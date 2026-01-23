from pathlib import Path
import chromadb
from chromadb.config import Settings
import os
from dotenv import load_dotenv
import cohere

# ==== ENV ====
BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

co = cohere.Client(os.getenv("COHERE_API_KEY"))

# ==== CHROMA ====
CHROMA_DIR = BASE_DIR / "RAG" / "chroma_db"
client = chromadb.Client(
    Settings(
        persist_directory=str(CHROMA_DIR),
        anonymized_telemetry=False
    )
)

models = client.get_collection("bmw_models")

# ==== QUERY ====
query = "BMW X7 diesel zasięg"
emb = co.embed(
    texts=[query],
    model="embed-multilingual-v3.0",
    input_type="search_query"
).embeddings[0]

results = models.query(
    query_embeddings=[emb],
    n_results=5
)

for i, (doc, meta) in enumerate(zip(results["documents"][0], results["metadatas"][0])):
    print(f"\n#{i+1}")
    print(doc)
    print(meta)
