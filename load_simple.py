import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(".").absolute()))

print("="*60)
print("ŁADOWANIE DOKUMENTÓW DO RAG")
print("="*60)

try:
    # 1. Import RAGService
    from app.services.rag_service import RAGService
    print("✓ Zaimportowano RAGService")
    
    # 2. Inicjalizuj
    rag = RAGService()
    print("✓ Zainicjalizowano RAGService")
    
    # 3. Sprawdź bieżący stan
    count_before = rag.collection.count()
    print(f"✓ Obecna liczba dokumentów: {count_before}")
    
    # 4. Załaduj dokumenty z knowledge_base
    kb_path = Path("./data/knowledge_base")
    if not kb_path.exists():
        print("✗ Brak katalogu knowledge_base!")
        exit()
    
    files = list(kb_path.glob("*.txt"))
    print(f"✓ Znaleziono {len(files)} plików .txt")
    
    texts = []
    metadatas = []
    
    for file_path in files:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        texts.append(content)
        metadatas.append({
            'source': file_path.name,
            'path': str(file_path)
        })
        print(f"  → {file_path.name} ({len(content)} znaków)")
    
    # 5. Dodaj dokumenty do ChromaDB
    print(f"\nDodawanie {len(texts)} dokumentów...")
    
    # Generuj ID
    ids = [f"doc_{i}_{Path(meta['source']).stem}" for i, meta in enumerate(metadatas)]
    
    # Dodaj do kolekcji
    rag.collection.add(
        documents=texts,
        metadatas=metadatas,
        ids=ids
    )
    print("✓ Dokumenty dodane do kolekcji")
    
    # 6. Sprawdź końcowy stan
    count_after = rag.collection.count()
    print(f"✓ Nowa liczba dokumentów: {count_after}")
    print(f"✓ Dodano: {count_after - count_before} dokumentów")
    
    # 7. Test wyszukiwania
    print("\n" + "-"*40)
    print("TEST WYSZUKIWANIA")
    print("-"*40)
    
    test_queries = [
        "jakie modele BMW są dostępne",
        "silnik diesel BMW",
        "cena Seria 3",
        "wyposażenie M Sport"
    ]
    
    for query in test_queries:
        print(f"\nZapytanie: '{query}'")
        
        try:
            # Wyszukaj
            results = rag.collection.query(
                query_texts=[query],
                n_results=2
            )
            
            if results and 'documents' in results and results['documents'][0]:
                docs = results['documents'][0]
                print(f"  ✓ Znaleziono {len(docs)} dokumentów")
                
                for i, doc in enumerate(docs):
                    # Znajdź źródło
                    source = "nieznane"
                    if results.get('metadatas') and results['metadatas'][0]:
                        source = results['metadatas'][0][i].get('source', 'nieznane')
                    
                    print(f"  Dokument {i+1} ({source}):")
                    print(f"    {doc[:150]}...")
            else:
                print(f"  ✗ Brak wyników")
                
        except Exception as e:
            print(f"  ✗ Błąd wyszukiwania: {e}")
    
    print("\n" + "="*60)
    print("SUKCES! Dokumenty załadowane.")
    print("Teraz uruchom: python run.py")
    print("="*60)
    
except Exception as e:
    print(f"\n✗ BŁĄD: {e}")
    import traceback
    traceback.print_exc()