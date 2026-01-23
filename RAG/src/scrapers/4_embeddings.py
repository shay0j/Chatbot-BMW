from sentence_transformers import SentenceTransformer
import numpy as np
import json
import pickle
from pathlib import Path
from datetime import datetime
import logging
from tqdm import tqdm

class EmbeddingGenerator:
    """Generuje embeddingi dla dokumentów"""
    
    def __init__(self, model_name='paraphrase-multilingual-mpnet-base-v2'):
        """
        Inicjalizuje model do embeddingów.
        Dobre modele dla polskiego:
        - 'paraphrase-multilingual-mpnet-base-v2' (najlepszy dla PL)
        - 'all-MiniLM-L6-v2' (szybszy, mniejszy)
        - 'intfloat/multilingual-e5-large' (bardzo dobry ale duży)
        """
        print(f"🔄 Ładuję model: {model_name}")
        self.model = SentenceTransformer(model_name)
        print(f"✅ Model załadowany!")
    
    def generate_embeddings(self, chunks, batch_size=32):
        """Generuje embeddingi dla listy chunk-ów"""
        if not chunks:
            print("❌ Brak danych do embeddingów")
            return []
        
        print(f"🔢 Generuję embeddingi dla {len(chunks)} chunk-ów...")
        
        # Ekstraktuj teksty
        texts = [chunk['text'] for chunk in chunks]
        
        # Generuj embeddingi batchami
        embeddings = []
        for i in tqdm(range(0, len(texts), batch_size), desc="Generowanie embeddingów"):
            batch = texts[i:i + batch_size]
            batch_embeddings = self.model.encode(batch, show_progress_bar=False)
            embeddings.extend(batch_embeddings)
        
        # Konwertuj do numpy array
        embeddings = np.array(embeddings)
        
        print(f"✅ Wygenerowano {len(embeddings)} embeddingów")
        print(f"   Wymiary: {embeddings.shape}")
        
        return embeddings
    
    def save_embeddings(self, chunks, embeddings, output_path):
        """Zapisuje embeddingi i metadane"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 1. Zapisuj w formacie FAISS-friendly
        data_to_save = {
            'chunks': chunks,
            'embeddings': embeddings,
            'metadata': {
                'total_chunks': len(chunks),
                'embedding_dim': embeddings.shape[1] if len(embeddings) > 0 else 0,
                'model_name': str(self.model),
                'created_at': datetime.now().isoformat()
            }
        }
        
        # 2. Zapis jako pickle
        pickle_file = output_path.with_suffix('.pkl')
        with open(pickle_file, 'wb') as f:
            pickle.dump(data_to_save, f)
        
        # 3. Zapis metadanych jako JSON
        metadata_file = output_path.with_suffix('.metadata.json')
        metadata = {
            'total_chunks': len(chunks),
            'embedding_dim': embeddings.shape[1] if len(embeddings) > 0 else 0,
            'model': str(self.model),
            'created_at': datetime.now().isoformat(),
            'chunks_info': []
        }
        
        for i, chunk in enumerate(chunks[:50]):  # Zapisz info o pierwszych 50 chunkach
            metadata['chunks_info'].append({
                'id': chunk.get('id', f'chunk_{i}'),
                'title': chunk['metadata'].get('title', '')[:100],
                'models': chunk['metadata'].get('models', []),
                'has_prices': bool(chunk['metadata'].get('prices')),
                'has_specs': bool(chunk['metadata'].get('engine_specs')),
                'text_preview': chunk['text'][:200] + '...' if len(chunk['text']) > 200 else chunk['text']
            })
        
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        
        # 4. Zapis do prostego formatu dla analizy
        simple_file = output_path.with_suffix('.simple.jsonl')
        with open(simple_file, 'w', encoding='utf-8') as f:
            for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                simple_data = {
                    'id': chunk.get('id', f'chunk_{i}'),
                    'text': chunk['text'],
                    'metadata': chunk['metadata'],
                    'embedding': embedding.tolist()  # Konwertuj numpy do listy
                }
                json.dump(simple_data, f, ensure_ascii=False)
                f.write('\n')
        
        print(f"💾 Zapisano embeddingi:")
        print(f"   📦 Pickle: {pickle_file} ({pickle_file.stat().st_size / 1024 / 1024:.2f} MB)")
        print(f"   📄 Metadata: {metadata_file}")
        print(f"   📊 Simple format: {simple_file}")
        
        return data_to_save

def find_rag_data():
    """Znajduje najnowsze dane RAG"""
    output_base = Path(r"C:\Users\hellb\Documents\Chatbot_BMW\RAG\output")
    
    # Szukaj folderów z "rag_ready"
    rag_folders = [f for f in output_base.iterdir() 
                  if f.is_dir() and "rag_ready" in f.name]
    
    if not rag_folders:
        print("❌ Nie znaleziono danych RAG!")
        print("   Najpierw uruchom 3_chunker.py")
        return None
    
    latest_rag = sorted(rag_folders)[-1]
    print(f"📁 Znaleziono dane RAG: {latest_rag.name}")
    
    # Sprawdź jakie pliki są dostępne
    jsonl_files = list(latest_rag.glob("*.jsonl"))
    if not jsonl_files:
        print(f"❌ Brak plików .jsonl w {latest_rag}")
        return None
    
    return latest_rag, jsonl_files

def load_chunks(data_folder):
    """Wczytuje chunk-i z folderu"""
    # Spróbuj wczytać wszystkie chunk-i
    all_chunks_file = data_folder / "all_chunks.jsonl"
    if all_chunks_file.exists():
        print(f"📖 Wczytuję: {all_chunks_file.name}")
        return load_chunks_from_file(all_chunks_file)
    
    # Jeśli nie ma all_chunks, spróbuj model_chunks
    model_chunks_file = data_folder / "model_chunks.jsonl"
    if model_chunks_file.exists():
        print(f"📖 Wczytuję: {model_chunks_file.name}")
        return load_chunks_from_file(model_chunks_file)
    
    # W przeciwnym razie weź pierwszy plik .jsonl
    jsonl_files = list(data_folder.glob("*.jsonl"))
    if jsonl_files:
        print(f"📖 Wczytuję: {jsonl_files[0].name}")
        return load_chunks_from_file(jsonl_files[0])
    
    return []

def load_chunks_from_file(file_path):
    """Wczytuje chunk-i z pliku JSONL"""
    chunks = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                chunk = json.loads(line)
                chunks.append(chunk)
            except json.JSONDecodeError as e:
                print(f"⚠️ Błąd parsowania linii: {e}")
                continue
    
    print(f"   Wczytano {len(chunks)} chunk-ów")
    return chunks

def create_embeddings():
    """Główna funkcja tworzenia embeddingów"""
    # Znajdź dane
    result = find_rag_data()
    if not result:
        return
    
    data_folder, jsonl_files = result
    
    # Wybierz plik do przetworzenia
    print(f"\n📚 Dostępne pliki w {data_folder.name}:")
    for i, file in enumerate(jsonl_files, 1):
        size_kb = file.stat().st_size / 1024
        print(f"   {i}. {file.name} ({size_kb:.1f} KB)")
    
    choice = input(f"\n🎯 Wybierz plik (1-{len(jsonl_files)}) lub Enter dla 'all_chunks.jsonl': ")
    
    if choice.strip() == '':
        # Domyślnie all_chunks.jsonl
        chosen_file = data_folder / "all_chunks.jsonl"
        if not chosen_file.exists():
            chosen_file = jsonl_files[0]
    else:
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(jsonl_files):
                chosen_file = jsonl_files[idx]
            else:
                print("❌ Nieprawidłowy wybór")
                return
        except ValueError:
            print("❌ Nieprawidłowy wybór")
            return
    
    # Wczytaj chunk-i
    print(f"\n📖 Wczytuję: {chosen_file.name}")
    chunks = load_chunks_from_file(chosen_file)
    
    if not chunks:
        print("❌ Brak danych do przetworzenia")
        return
    
    # Wybierz model
    print(f"\n🤖 Dostępne modele embeddingów:")
    print("   1. paraphrase-multilingual-mpnet-base-v2 (najlepszy dla PL, 768d)")
    print("   2. all-MiniLM-L6-v2 (szybszy, 384d)")
    print("   3. intfloat/multilingual-e5-base (dobry kompromis, 768d)")
    
    model_choice = input("🎯 Wybierz model (1-3) lub Enter dla domyślnego (1): ")
    
    model_map = {
        '1': 'paraphrase-multilingual-mpnet-base-v2',
        '2': 'all-MiniLM-L6-v2',
        '3': 'intfloat/multilingual-e5-base'
    }
    
    model_name = model_map.get(model_choice.strip() or '1', 'paraphrase-multilingual-mpnet-base-v2')
    
    # Generuj embeddingi
    print(f"\n🚀 Rozpoczynam generowanie embeddingów z modelem: {model_name}")
    
    # Sprawdź czy zainstalowany sentence-transformers
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print("❌ Brak sentence-transformers!")
        print("   Zainstaluj: pip install sentence-transformers")
        return
    
    generator = EmbeddingGenerator(model_name=model_name)
    
    # Generuj embeddingi
    embeddings = generator.generate_embeddings(chunks)
    
    if len(embeddings) == 0:
        print("❌ Nie udało się wygenerować embeddingów")
        return
    
    # Zapisz embeddingi
    print(f"\n💾 Zapisuję embeddingi...")
    output_name = f"embeddings_{model_name.replace('/', '_').replace('-', '_')}"
    output_path = data_folder / output_name
    
    saved_data = generator.save_embeddings(chunks, embeddings, output_path)
    
    # Podsumowanie
    print(f"\n✅ PODSUMOWANIE:")
    print(f"   📊 Chunk-i: {len(chunks)}")
    print(f"   🔢 Embeddingi: {len(embeddings)}")
    print(f"   📐 Wymiar: {embeddings.shape[1]}")
    print(f"   🗂️  Zapisano w: {data_folder}")
    
    # Testuj podobieństwo
    if len(chunks) >= 2:
        print(f"\n🧪 Test podobieństwa między pierwszymi dwoma chunk-ami:")
        similarity = np.dot(embeddings[0], embeddings[1]) / (
            np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1])
        )
        print(f"   Podobieństwo: {similarity:.3f}")
        
        # Jeśli similarity jest bardzo niskie, chunk-i są różne tematycznie
        if similarity < 0.3:
            print("   ℹ️  Niskie podobieństwo - chunk-i dotyczą różnych tematów")
        elif similarity > 0.7:
            print("   ℹ️  Wysokie podobieństwo - chunk-i dotyczą podobnych tematów")
    
    return saved_data

if __name__ == "__main__":
    print("=" * 60)
    print("🧠 GENEROWANIE EMBEDDINGÓW - BMW CHATBOT")
    print("=" * 60)
    
    # Sprawdź czy są dane RAG
    result = find_rag_data()
    if not result:
        exit()
    
    # Zapytaj czy generować embeddingi
    print("\n" + "="*60)
    create_emb = input("🧠 Czy chcesz wygenerować embeddingi? (t/n): ")
    
    if create_emb.lower() == 't':
        print("\n🚀 Rozpoczynam generowanie embeddingów...")
        saved_data = create_embeddings()
        
        if saved_data:
            print(f"\n🎉 Embeddingi gotowe!")
            print(f"   Następny krok: stworzenie wektorowej bazy danych (FAISS/Chroma)")
            print(f"   Uruchom 5_vector_db.py aby kontynuować.")
    else:
        print("❌ Anulowano generowanie embeddingów.")