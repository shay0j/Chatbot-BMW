import os
import sys
import chromadb
from pathlib import Path

print("="*60)
print("DIAGNOZA RAG - SZYBKI TEST")
print("="*60)

# 1. Sprawdź katalogi
print("\n1. KATALOGI:")
data_dir = Path("./data")
chroma_dir = data_dir / "chroma_db"
kb_dir = data_dir / "knowledge_base"

print(f"data/ exists: {data_dir.exists()}")
print(f"data/chroma_db/ exists: {chroma_dir.exists()}")
print(f"data/knowledge_base/ exists: {kb_dir.exists()}")

if kb_dir.exists():
    files = list(kb_dir.glob("*"))
    print(f"Files in knowledge_base: {len(files)}")
    for f in files[:10]:
        print(f"  - {f.name}")

# 2. Sprawdź ChromaDB bezpośrednio
print("\n2. CHROMADB DIRECT TEST:")
try:
    client = chromadb.PersistentClient(path=str(chroma_dir))
    collections = client.list_collections()
    print(f"Collections: {len(collections)}")
    
    for col in collections:
        print(f"\nCollection: '{col.name}'")
        print(f"  Count: {col.count()}")
        
        # Pobierz próbkę
        if col.count() > 0:
            try:
                sample = col.peek(limit=2)
                if 'documents' in sample and sample['documents']:
                    print(f"  Sample documents: {len(sample['documents'][0])}")
                    for i, doc in enumerate(sample['documents'][0][:2]):
                        print(f"    Doc {i+1}: {doc[:100]}...")
                else:
                    print("  No documents in sample")
            except Exception as e:
                print(f"  Error peeking: {e}")
                
except Exception as e:
    print(f"Error accessing ChromaDB: {e}")

# 3. Sprawdź import rag_service
print("\n3. TEST RAG_SERVICE IMPORT:")
try:
    sys.path.insert(0, str(Path(".").absolute()))
    from app.services.rag_service import RAGService
    print("✓ RAGService import successful")
    
    # Spróbuj zainicjalizować
    try:
        rag = RAGService()
        print("✓ RAGService initialized")
        
        # Sprawdź atrybuty
        if hasattr(rag, 'collection'):
            print(f"✓ Has collection attribute")
            print(f"  Collection name: {rag.collection.name}")
            print(f"  Collection count: {rag.collection.count()}")
            
    except Exception as e:
        print(f"✗ Error initializing RAGService: {e}")
        
except ImportError as e:
    print(f"✗ Import error: {e}")

# 4. Test wyszukiwania (jeśli wszystko OK)
print("\n4. TEST WYSZUKIWANIA:")
try:
    # Spróbuj wykonać testowe wyszukiwanie
    rag = RAGService()
    query = "model BMW"
    
    import asyncio
    
    async def test_search():
        results = await rag.search(query, k=3)
        print(f"Query: '{query}'")
        if results and 'documents' in results:
            docs = results['documents'][0]
            print(f"Found {len(docs)} documents")
            for i, doc in enumerate(docs):
                print(f"  Doc {i+1}: {doc[:150]}...")
        else:
            print("No results or invalid format")
    
    asyncio.run(test_search())
    
except Exception as e:
    print(f"Search test failed: {e}")

print("\n" + "="*60)
print("KONIEC DIAGNOZY")
print("="*60)