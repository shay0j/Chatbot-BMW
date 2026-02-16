import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
import json
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
import sys
import hashlib
import shutil
import time

class BMREmbeddingGenerator:
    """Generator embeddingów zoptymalizowany dla danych BMW z integracją ChromaDB - ROBUST VERSION"""
    
    def __init__(self, use_cohere: bool = False, cohere_api_key: str = None):
        """
        Inicjalizuje generator embeddingów
        
        Args:
            use_cohere: Czy używać Cohere API (lepsze, ale płatne)
            cohere_api_key: Klucz API Cohere (jeśli use_cohere=True)
        """
        self.use_cohere = use_cohere
        
        if use_cohere and cohere_api_key:
            print("🔑 Używam Cohere Embedding API")
            try:
                self.embedding_fn = embedding_functions.CohereEmbeddingFunction(
                    api_key=cohere_api_key,
                    model_name="embed-multilingual-v3.0"
                )
            except Exception as e:
                print(f"⚠️  Błąd Cohere: {e}, używam domyślnych embeddingów")
                self.use_cohere = False
                self.embedding_fn = embedding_functions.DefaultEmbeddingFunction()
        else:
            print("🤖 Używam domyślnych embeddingów ChromaDB")
            self.embedding_fn = embedding_functions.DefaultEmbeddingFunction()
    
    def load_chunks(self, data_folder: Path) -> list:
        """Wczytuje chunk-i z folderu RAG"""
        # Szukaj najpierw all_chunks.jsonl
        files_to_try = [
            data_folder / "all_chunks.jsonl",
            data_folder / "model_chunks.jsonl",
            data_folder / "other_chunks.jsonl"
        ]
        
        for file_path in files_to_try:
            if file_path.exists():
                print(f"📖 Wczytuję: {file_path.name}")
                return self._load_jsonl(file_path)
        
        # Jeśli nie znaleziono, szukaj dowolnego .jsonl
        jsonl_files = list(data_folder.glob("*.jsonl"))
        if jsonl_files:
            print(f"📖 Wczytuję: {jsonl_files[0].name}")
            return self._load_jsonl(jsonl_files[0])
        
        return []
    
    def _load_jsonl(self, file_path: Path) -> list:
        """Wczytuje plik JSONL"""
        chunks = []
        error_count = 0
        
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for line_num, line in tqdm(enumerate(lines, 1), desc="Wczytywanie chunków", total=len(lines)):
                try:
                    chunk = json.loads(line)
                    # Walidacja chunka
                    if self._validate_chunk(chunk):
                        chunks.append(chunk)
                except json.JSONDecodeError as e:
                    error_count += 1
                    if error_count <= 5:  # Pokaż tylko pierwsze 5 błędów
                        print(f"⚠️  Błąd w linii {line_num}: {e}")
        
        if error_count > 0:
            print(f"⚠️  Łącznie błędów: {error_count}")
        
        print(f"✅ Wczytano {len(chunks)} poprawnych chunków")
        return chunks
    
    def _validate_chunk(self, chunk: dict) -> bool:
        """Waliduje pojedynczy chunek"""
        required_fields = ['id', 'text', 'metadata']
        
        # Sprawdź wymagane pola
        for field in required_fields:
            if field not in chunk:
                return False
        
        # Sprawdź czy tekst nie jest pusty
        if not chunk['text'] or len(chunk['text'].strip()) < 30:
            return False
        
        # Sprawdź czy nie ma placeholderów
        text_lower = chunk['text'].lower()
        bad_phrases = ['lorem ipsum', 'skip to main content', 'dummy text', 'placeholder']
        if any(phrase in text_lower for phrase in bad_phrases):
            return False
        
        return True
    
    def _prepare_metadata_for_chromadb(self, metadata: dict) -> dict:
        """
        Przygotowuje metadata dla ChromaDB - konwertuje listy na stringi
        
        ChromaDB akceptuje tylko: str, int, float, bool
        NIE akceptuje: list, dict, None
        """
        cleaned_metadata = {}
        
        for key, value in metadata.items():
            if value is None:
                # Pomijaj None
                continue
            elif isinstance(value, list):
                # Listy konwertuj na string (join przecinkami)
                if value:
                    # Usuń duplikaty i posortuj dla spójności
                    unique_values = []
                    for item in value:
                        if item not in unique_values and item is not None:
                            unique_values.append(str(item))
                    cleaned_metadata[key] = ', '.join(unique_values)
                else:
                    # Puste listy -> pusty string
                    cleaned_metadata[key] = ""
            elif isinstance(value, dict):
                # Słowniki konwertuj na JSON string
                try:
                    cleaned_metadata[key] = json.dumps(value, ensure_ascii=False)
                except:
                    cleaned_metadata[key] = str(value)
            elif isinstance(value, (str, int, float, bool)):
                # Typy akceptowane przez ChromaDB
                cleaned_metadata[key] = value
            else:
                # Wszystko inne konwertuj na string
                cleaned_metadata[key] = str(value)
        
        return cleaned_metadata
    
    def create_chromadb_collection(self, collection_name: str = "bmw_docs", 
                                   persist_directory: Path = None) -> chromadb.Collection:
        """
        Tworzy lub łączy się z kolekcją w ChromaDB
        
        Args:
            collection_name: Nazwa kolekcji
            persist_directory: Ścieżka do zapisu bazy (None = pamięć)
        """
        if persist_directory:
            print(f"💾 Używam trwałej bazy w: {persist_directory}")
            
            # Jeśli folder istnieje, zapytaj czy usunąć
            if persist_directory.exists():
                print("⚠️  Znaleziono istniejącą bazę...")
                response = input("🧹 Czy chcesz usunąć starą bazę? (t/n): ")
                if response.lower() == 't':
                    try:
                        shutil.rmtree(persist_directory)
                        print("🗑️  Usunięto starą bazę")
                        time.sleep(1)  # Daj czas na usunięcie
                    except Exception as e:
                        print(f"⚠️  Nie udało się usunąć: {e}")
            
            # Utwórz folder
            persist_directory.mkdir(exist_ok=True)
            client = chromadb.PersistentClient(path=str(persist_directory))
        else:
            print("⚡ Używam bazy w pamięci")
            client = chromadb.Client(Settings())
        
        # Sprawdź czy kolekcja istnieje
        try:
            existing_collections = client.list_collections()
        except Exception as e:
            print(f"⚠️  Błąd przy pobieraniu kolekcji: {e}")
            print("🆕 Tworzę nową kolekcję...")
            return client.create_collection(
                name=collection_name,
                embedding_function=self.embedding_fn,
                metadata={"hnsw:space": "cosine"}
            )
        
        if collection_name in [c.name for c in existing_collections]:
            print(f"📂 Łączę się z istniejącą kolekcją: {collection_name}")
            collection = client.get_collection(name=collection_name)
            
            # Zapytaj użytkownika co zrobić
            if collection.count() > 0:
                print(f"⚠️  Kolekcja ma już {collection.count()} dokumentów")
                response = input("🧹 Czy chcesz usunąć istniejące dane? (t/n): ")
                if response.lower() == 't':
                    client.delete_collection(name=collection_name)
                    print("🗑️  Usunięto istniejącą kolekcję")
                    # Stwórz nową
                    collection = client.create_collection(
                        name=collection_name,
                        embedding_function=self.embedding_fn,
                        metadata={"hnsw:space": "cosine"}
                    )
                else:
                    print("📝 Dodam nowe dokumenty do istniejącej kolekcji")
        else:
            # Stwórz nową kolekcję
            print(f"🆕 Tworzę nową kolekcję: {collection_name}")
            collection = client.create_collection(
                name=collection_name,
                embedding_function=self.embedding_fn,
                metadata={"hnsw:space": "cosine"}
            )
        
        return collection
    
    def add_to_chromadb(self, chunks: list, collection: chromadb.Collection, 
                        batch_size: int = 50) -> None:
        """Dodaje chunk-i do ChromaDB z poprawionymi metadanymi"""
        if not chunks:
            print("❌ Brak chunków do dodania")
            return
        
        print(f"📤 Dodaję {len(chunks)} chunków do ChromaDB...")
        
        # Przygotuj dane
        ids = []
        documents = []
        metadatas = []
        
        for chunk in tqdm(chunks, desc="Przygotowanie danych"):
            # Unikalne ID (hash tekstu + oryginalne ID)
            text_hash = hashlib.md5(chunk['text'].encode()).hexdigest()[:8]
            unique_id = f"{chunk['id']}_{text_hash}"
            
            # Przygotuj metadata DLA CHROMADB
            metadata = chunk['metadata'].copy()
            metadata['source_file'] = chunk.get('source_file', 'unknown')
            metadata['added_at'] = datetime.now().isoformat()
            
            # OCZYŚĆ METADATA DLA CHROMADB
            cleaned_metadata = self._prepare_metadata_for_chromadb(metadata)
            
            ids.append(unique_id)
            documents.append(chunk['text'])
            metadatas.append(cleaned_metadata)
        
        print(f"✅ Przygotowano {len(ids)} dokumentów do dodania")
        
        # Dodaj partiami (mniejszy batch_size dla bezpieczeństwa)
        successful_docs = 0
        failed_docs = 0
        
        for i in tqdm(range(0, len(chunks), batch_size), desc="Ładowanie do ChromaDB"):
            batch_ids = ids[i:i+batch_size]
            batch_docs = documents[i:i+batch_size]
            batch_metas = metadatas[i:i+batch_size]
            
            try:
                collection.add(
                    ids=batch_ids,
                    documents=batch_docs,
                    metadatas=batch_metas
                )
                successful_docs += len(batch_ids)
            except Exception as e:
                print(f"⚠️  Błąd przy ładowaniu batch-a {i//batch_size}: {e}")
                print("   Próbuję dodać pojedynczo...")
                
                # Spróbuj dodać pojedynczo
                for j in range(len(batch_ids)):
                    try:
                        collection.add(
                            ids=[batch_ids[j]],
                            documents=[batch_docs[j]],
                            metadatas=[batch_metas[j]]
                        )
                        successful_docs += 1
                    except Exception as e2:
                        failed_docs += 1
                        if failed_docs <= 5:  # Pokaż tylko 5 pierwszych błędów
                            print(f"❌ Nie udało się dodać dokumentu {batch_ids[j]}: {e2}")
        
        print(f"✅ Dodano {successful_docs} dokumentów do kolekcji")
        if failed_docs > 0:
            print(f"⚠️  Nie udało się dodać {failed_docs} dokumentów")
        
        # Pokaż statystyki zamiast próbki metadanych
        if successful_docs > 0:
            self._show_collection_stats(collection)
    
    def _show_collection_stats(self, collection):
        """Pokazuje statystyki kolekcji"""
        print(f"\n📊 STATYSTYKI KOLEKCJI:")
        print(f"   Nazwa: {collection.name}")
        print(f"   Dokumenty: {collection.count()}")
        
        # Spróbuj pobrać przykładowe dane
        try:
            # Pobierz pierwsze 5 dokumentów
            results = collection.get(limit=min(5, collection.count()))
            
            if results['metadatas'] and len(results['metadatas']) > 0:
                print(f"   Przykładowe pola metadata:")
                # Weź pierwszy dokument
                first_doc_meta = results['metadatas'][0]
                for key in list(first_doc_meta.keys())[:5]:  # Pokaż pierwsze 5 kluczy
                    print(f"     - {key}")
            else:
                print("   Nie udało się pobrać metadanych")
                
        except Exception as e:
            print(f"   Błąd przy pobieraniu statystyk: {e}")
    
    def test_retrieval(self, collection: chromadb.Collection, test_queries: list = None):
        """Testuje retrieval z przykładowymi pytaniami o BMW"""
        if not test_queries:
            test_queries = [
                "Ile kosztuje BMW X3?",
                "Jakie są opcje finansowania BMW?",
                "Gdzie mogę zrobić test drive BMW?",
                "Jakie modele BMW są elektryczne?",
                "Jaka jest moc silnika BMW X5?",
            ]
        
        print(f"\n🔍 TEST RETRIEVAL - {len(test_queries)} pytań")
        print("=" * 80)
        
        results_summary = []
        
        for query in test_queries:
            print(f"\n❓ PYTANIE: {query}")
            
            try:
                results = collection.query(
                    query_texts=[query],
                    n_results=3,
                    include=["documents", "metadatas", "distances"]
                )
                
                if results['documents'] and results['documents'][0]:
                    # Znaleziono wyniki
                    for i, (doc, meta, distance) in enumerate(zip(
                        results['documents'][0], 
                        results['metadatas'][0], 
                        results['distances'][0]
                    )):
                        print(f"   {i+1}. (dystans: {distance:.3f})")
                        # Modele są teraz stringiem, nie listą
                        models_str = meta.get('models', '')
                        print(f"      Modele: {models_str}")
                        print(f"      Priorytet: {meta.get('retrieval_priority', 1)}")
                        print(f"      Fragment: {doc[:100]}...")
                    
                    results_summary.append((query, "✅ TRAFNE"))
                else:
                    print("   ❌ BRAK WYNIKÓW")
                    results_summary.append((query, "❌ BRAK"))
                    
            except Exception as e:
                print(f"   ❌ BŁĄD: {e}")
                results_summary.append((query, "❌ BŁĄD"))
        
        # Podsumowanie testów
        print(f"\n📊 PODSUMOWANIE TESTOW:")
        print("=" * 80)
        
        for query, result in results_summary:
            print(f"{result} - {query}")
        
        trafne_count = sum(1 for _, result in results_summary if "✅" in result)
        print(f"\n🎯 Skuteczność: {trafne_count}/{len(test_queries)} ({trafne_count/len(test_queries)*100:.1f}%)")
    
    def save_config(self, collection: chromadb.Collection, output_path: Path):
        """Zapisuje konfigurację bazy danych"""
        try:
            config = {
                'created_at': datetime.now().isoformat(),
                'collection_name': collection.name,
                'collection_count': collection.count(),
                'embedding_function': 'Cohere' if self.use_cohere else 'Default',
                'metadata': collection.metadata,
                'settings': {
                    'hnsw:space': 'cosine',
                    'allow_reset': True
                }
            }
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(config, f, ensure_ascii=False, indent=2)
            
            print(f"💾 Zapisano konfigurację: {output_path}")
        except Exception as e:
            print(f"⚠️  Nie udało się zapisać konfiguracji: {e}")


def find_latest_rag_data():
    """Znajduje najnowsze dane RAG"""
    output_base = Path(r"C:\Users\hellb\Documents\Chatbot_BMW\RAG\output")
    
    # Szukaj folderów z "rag_ready_final" (najnowsze) lub "rag_ready"
    final_folders = [f for f in output_base.iterdir() 
                    if f.is_dir() and "rag_ready_final" in f.name]
    
    if final_folders:
        latest = sorted(final_folders)[-1]
        print(f"📁 Znaleziono FINALNE dane RAG: {latest.name}")
        return latest
    
    # Jeśli nie ma final, szukaj innych
    rag_folders = [f for f in output_base.iterdir() 
                  if f.is_dir() and "rag_ready" in f.name]
    
    if not rag_folders:
        print("❌ Nie znaleziono danych RAG!")
        print("   Najpierw uruchom 3_chunker_final.py")
        return None
    
    latest = sorted(rag_folders)[-1]
    print(f"📁 Znaleziono dane RAG: {latest.name}")
    return latest


def main():
    """Główna funkcja"""
    print("=" * 70)
    print("🧠 EMBEDDINGI & CHROMADB - BMW CHATBOT (ROBUST VERSION)")
    print("=" * 70)
    
    try:
        # 1. Znajdź dane
        data_folder = find_latest_rag_data()
        if not data_folder:
            return
        
        print(f"\n📊 Analizuję folder: {data_folder}")
        
        # 2. Wybierz tryb embeddingów
        print("\n🤖 WYBIERZ TRYB EMBEDDINGÓW:")
        print("   1. Domyślne embeddingi ChromaDB (bezpłatne, szybkie)")
        print("   2. Cohere API (płatne, najlepsza jakość)")
        
        choice = input("   Wybierz (1-2) [domyślnie 1]: ").strip() or "1"
        
        if choice == "2":
            cohere_key = input("   Podaj klucz API Cohere: ").strip()
            if not cohere_key:
                print("   ⚠️  Brak klucza, używam domyślnych embeddingów")
                generator = BMREmbeddingGenerator(use_cohere=False)
            else:
                generator = BMREmbeddingGenerator(use_cohere=True, cohere_api_key=cohere_key)
        else:
            generator = BMREmbeddingGenerator(use_cohere=False)
        
        # 3. Wczytaj chunki
        print(f"\n📖 Wczytuję chunk-i z {data_folder.name}...")
        chunks = generator.load_chunks(data_folder)
        
        if not chunks:
            print("❌ Nie wczytano żadnych chunków!")
            return
        
        print(f"✅ Wczytano {len(chunks)} wysokiej jakości chunków")
        
        # 4. Stwórz/połącz z ChromaDB
        print(f"\n💾 KONFIGURACJA CHROMADB:")
        print("   1. Baza trwała (zapis na dysk)")
        print("   2. Baza w pamięci (tylko do testów)")
        
        db_choice = input("   Wybierz (1-2) [domyślnie 1]: ").strip() or "1"
        
        if db_choice == "1":
            # Użyj nowej nazwy folderu, żeby uniknąć problemów
            persist_dir = Path(r"C:\Users\hellb\Documents\Chatbot_BMW\RAG\chroma_db_working")
            collection = generator.create_chromadb_collection(
                collection_name="bmw_docs",
                persist_directory=persist_dir
            )
        else:
            collection = generator.create_chromadb_collection(
                collection_name="bmw_docs_test",
                persist_directory=None
            )
        
        # 5. Dodaj chunki do bazy
        print(f"\n📤 Dodaję {len(chunks)} chunków do ChromaDB...")
        generator.add_to_chromadb(chunks, collection)
        
        # 6. Test retrieval
        print(f"\n🔍 Rozpoczynam testy retrieval...")
        generator.test_retrieval(collection)
        
        # 7. Zapisz konfigurację
        config_path = data_folder / "chromadb_config.json"
        generator.save_config(collection, config_path)
        
        # 8. Instrukcje dalsze
        print(f"\n🎉 SUKCES! Baza danych gotowa.")
        print(f"📁 Dane RAG: {data_folder}")
        print(f"💾 Baza ChromaDB: {persist_dir if db_choice == '1' else 'pamięć'}")
        print(f"📄 Konfiguracja: {config_path}")
        
        print(f"\n🚀 Następne kroki:")
        print("   1. Zaktualizuj ścieżkę w rag_test_chromadb.py:")
        print(f"      chroma_path = Path(r\"{persist_dir if db_choice == '1' else 'ChromaDB in memory'}\")")
        print(f"      collection_name='bmw_docs'")
        print("   2. Uruchom testy:")
        print("      python rag_test_chromadb.py")
        
        return collection
        
    except Exception as e:
        print(f"\n❌ Wystąpił krytyczny błąd: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⏹️  Przerwano przez użytkownika.")
    
    print("\n" + "="*70)
    print("🧠 Embedding generator zakończył pracę")
    print("="*70)