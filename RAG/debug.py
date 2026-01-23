from pathlib import Path
import os
from dotenv import load_dotenv
import chromadb
import cohere

# ======================================================
# 🔧 ŚCIEŻKI I ENV
# ======================================================
BASE_DIR = Path(__file__).resolve().parent.parent
ENV_PATH = BASE_DIR / ".env"
load_dotenv(ENV_PATH)

COHERE_API_KEY = os.getenv("COHERE_API_KEY")
if not COHERE_API_KEY:
    raise RuntimeError("❌ COHERE_API_KEY nie został załadowany z .env")

CHROMA_DIR = BASE_DIR / "RAG" / "chroma_db"
print(f"🧠 [DEBUG] Używam Chroma DB z: {CHROMA_DIR}")

# Sprawdzenie czy folder istnieje
if not CHROMA_DIR.exists():
    print(f"❌ Folder Chroma DB nie istnieje: {CHROMA_DIR}")
    exit(1)

# ======================================================
# 🤖 Cohere client (embed test query)
# ======================================================
co = cohere.Client(COHERE_API_KEY)

def embed_texts(texts: list[str]) -> list[list[float]]:
    response = co.embed(
        texts=texts,
        model="embed-multilingual-v3.0",
        input_type="search_query"  # Zmienione na search_query dla zapytań!
    )
    return response.embeddings

# ======================================================
# 🧩 Chroma client - MUSI być PersistentClient!
# ======================================================
try:
    # UŻYJ PersistentClient z tą samą ścieżką co w load_to_chroma.py
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    print(f"✅ Połączono z Chroma DB: {CHROMA_DIR}")
    
    collections = client.list_collections()
    
    if not collections:
        print("❌ Brak kolekcji w Chroma")
    else:
        print(f"📦 Znaleziono {len(collections)} kolekcji:")
        for i, c in enumerate(collections):
            count = c.count()
            print(f" {i+1}. '{c.name}' - {count} dokumentów")
        
        # ======================================================
        # 🔹 Test zapytań do obu kolekcji
        # ======================================================
        query_text = "X7"
        print(f"\n🔎 Test query: '{query_text}'")
        
        query_embedding = embed_texts([query_text])
        
        for collection in collections:
            print(f"\n--- Kolekcja: '{collection.name}' ---")
            
            results = collection.query(
                query_embeddings=query_embedding,
                n_results=3
            )
            
            if results['documents'][0]:
                for i, doc in enumerate(results['documents'][0]):
                    metadata = results['metadatas'][0][i]
                    distance = results['distances'][0][i] if results['distances'] else "N/A"
                    print(f"\n{i+1}. [distance: {distance:.4f}]")
                    print(f"   Dokument: {doc[:200]}...")  # Pierwsze 200 znaków
                    print(f"   Metadata: {metadata}")
            else:
                print("   Brak wyników")
                
        # ======================================================
        # 🔹 Dodatkowo: pokaż kilka przykładowych dokumentów
        # ======================================================
        print(f"\n{'='*50}")
        print("📋 PRZYKŁADOWE DOKUMENTY Z KAŻDEJ KOLEKCJI:")
        
        for collection in collections:
            print(f"\n--- Przykłady z kolekcji: '{collection.name}' ---")
            
            # Pobierz kilka pierwszych dokumentów
            sample_results = collection.get(limit=2)
            
            if sample_results['documents']:
                for i, doc in enumerate(sample_results['documents']):
                    metadata = sample_results['metadatas'][i]
                    print(f"\n{i+1}. Dokument (ID: {sample_results['ids'][i]}):")
                    print(f"   {doc[:300]}...")  # Pierwsze 300 znaków
                    print(f"   Metadata: {metadata}")
            else:
                print("   Kolekcja pusta")

except Exception as e:
    print(f"❌ Błąd podczas łączenia z Chroma DB: {e}")
    print("\n💡 Wskazówki:")
    print("1. Upewnij się, że najpierw uruchomiłeś load_to_chroma.py")
    print("2. Sprawdź czy folder chroma_db zawiera dane")
    print("3. Jeśli nie, usuń folder chroma_db i uruchom load_to_chroma.py ponownie")