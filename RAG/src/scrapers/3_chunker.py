from langchain.text_splitter import RecursiveCharacterTextSplitter
import json
import pandas as pd
from pathlib import Path
from datetime import datetime
import re

class RAGPreprocessor:
    """Przygotowuje dane dla RAG"""
    
    def __init__(self, chunk_size=1000, chunk_overlap=200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,  # Używamy len zamiast tiktoken
            separators=["\n\n", "\n", " ", ""]
        )
    
    def _estimate_tokens(self, text):
        """Szacuje liczbę tokenów (prosta metoda)"""
        # Przyjmij że 1 token ≈ 4 znaki dla angielskiego/polskiego
        return len(text) // 4
    
    def _create_chunk_metadata(self, item, chunk_index, total_chunks, chunk_text):
        """Tworzy metadata dla chunka"""
        metadata = {
            'unique_id': item.get('unique_id', f"doc_{abs(hash(item.get('url', '')))}"),
            'source_url': item.get('url', ''),
            'title': item.get('title', '')[:200],
            'categories': item.get('categories', []),
            'detected_models': item.get('detected_models', []),
            'is_model_page': item.get('is_model_page', False),
            'original_word_count': item.get('word_count', 0),
            'scraped_at': item.get('scraped_at', ''),
            'normalized_at': item.get('normalized_at', ''),
            'chunk_index': chunk_index,
            'total_chunks': total_chunks,
            'chunk_char_count': len(chunk_text),
            'chunk_token_estimate': self._estimate_tokens(chunk_text)
        }
        
        # Dodaj specyfikacje
        specs = item.get('specifications', {})
        if specs:
            # Dodaj najważniejsze specyfikacje
            if 'model_info' in specs:
                metadata['models'] = specs['model_info'].get('models', [])
                if 'model_from_url' in specs['model_info']:
                    metadata['model_from_url'] = specs['model_info']['model_from_url']
            
            # Dodaj kluczowe specyfikacje
            if 'engine' in specs:
                engine_specs = {}
                for key, value in specs['engine'].items():
                    if value and value != 0:
                        engine_specs[key] = value
                if engine_specs:
                    metadata['engine_specs'] = engine_specs
            
            if 'prices' in specs and specs['prices']:
                price_info = []
                for price in specs['prices'][:3]:
                    if isinstance(price, dict):
                        price_info.append({
                            'formatted': price.get('formatted', ''),
                            'amount': price.get('amount', 0)
                        })
                if price_info:
                    metadata['prices'] = price_info
        
        return metadata
    
    def create_chunks(self, data_items):
        """Tworzy chunk-i z danych"""
        chunks = []
        
        for item in data_items:
            # Treść do podziału
            content = item.get('content', '')
            if not content or len(content.strip()) < 50:
                continue
            
            # Podziel na chunk-i
            content_chunks = self.text_splitter.split_text(content)
            
            # Stwórz dokumenty z metadata
            for i, chunk_text in enumerate(content_chunks):
                chunk_metadata = self._create_chunk_metadata(
                    item, i, len(content_chunks), chunk_text
                )
                
                chunks.append({
                    'id': f"{chunk_metadata['unique_id']}_chunk_{i:03d}",
                    'text': chunk_text,
                    'metadata': chunk_metadata
                })
        
        return chunks
    
    def save_for_rag(self, chunks, output_path):
        """Zapisuje chunk-i w formacie dla RAG"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 1. JSONL format
        jsonl_file = output_path.with_suffix('.jsonl')
        with open(jsonl_file, 'w', encoding='utf-8') as f:
            for chunk in chunks:
                json.dump(chunk, f, ensure_ascii=False)
                f.write('\n')
        
        # 2. CSV format
        csv_file = output_path.with_suffix('.csv')
        try:
            csv_data = []
            for chunk in chunks:
                row = {
                    'id': chunk.get('id', ''),
                    'text_preview': (chunk['text'][:300] + '...') if len(chunk['text']) > 300 else chunk['text'],
                    'source_url': chunk['metadata'].get('source_url', ''),
                    'title': chunk['metadata'].get('title', ''),
                    'models': ', '.join(chunk['metadata'].get('models', [])),
                    'is_model_page': chunk['metadata'].get('is_model_page', False),
                    'chunk_size': chunk['metadata'].get('chunk_char_count', 0),
                    'token_estimate': chunk['metadata'].get('chunk_token_estimate', 0)
                }
                csv_data.append(row)
            
            df = pd.DataFrame(csv_data)
            df.to_csv(csv_file, index=False, encoding='utf-8')
            print(f"   📊 CSV: {csv_file}")
        except Exception as e:
            print(f"   ⚠️ Nie udało się zapisać CSV: {e}")
        
        # 3. Statystyki
        stats = {
            'total_chunks': len(chunks),
            'avg_chunk_size': sum(c['metadata'].get('chunk_char_count', 0) for c in chunks) / len(chunks) if chunks else 0,
            'avg_token_estimate': sum(c['metadata'].get('chunk_token_estimate', 0) for c in chunks) / len(chunks) if chunks else 0,
            'min_chars': min((c['metadata'].get('chunk_char_count', 0) for c in chunks), default=0),
            'max_chars': max((c['metadata'].get('chunk_char_count', 0) for c in chunks), default=0),
            'model_chunks': sum(1 for c in chunks if c['metadata'].get('is_model_page', False)),
            'other_chunks': sum(1 for c in chunks if not c['metadata'].get('is_model_page', False)),
            'chunks_with_prices': sum(1 for c in chunks if c['metadata'].get('prices')),
            'chunks_with_engine_specs': sum(1 for c in chunks if c['metadata'].get('engine_specs')),
            'saved_at': datetime.now().isoformat()
        }
        
        stats_file = output_path.with_suffix('.stats.json')
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
        
        print(f"💾 Zapisałem {len(chunks)} chunk-i:")
        print(f"   📄 JSONL: {jsonl_file} ({jsonl_file.stat().st_size / 1024:.1f} KB)")
        print(f"   📈 Statystyki: {stats_file}")
        print(f"   🚗 Chunk-i z modelami: {stats['model_chunks']}")
        print(f"   📰 Inne chunk-i: {stats['other_chunks']}")
        print(f"   💰 Chunk-i z cenami: {stats['chunks_with_prices']}")
        print(f"   🔧 Chunk-i ze specyfikacjami: {stats['chunks_with_engine_specs']}")
        print(f"   📊 Rozmiar: {stats['min_chars']}-{stats['max_chars']} znaków")
        
        return chunks

def find_latest_data():
    """Znajduje najnowsze dane"""
    output_base = Path(r"C:\Users\hellb\Documents\Chatbot_BMW\RAG\output")
    
    # 1. Najpierw szukaj znormalizowanych danych
    normalized_folders = [f for f in output_base.iterdir() 
                         if f.is_dir() and "_normalized" in f.name]
    
    if normalized_folders:
        latest_normalized = sorted(normalized_folders)[-1]
        print(f"📁 Znaleziono znormalizowane dane: {latest_normalized.name}")
        return latest_normalized
    
    # 2. Jeśli nie ma znormalizowanych, szukaj surowych danych
    raw_folders = [f for f in output_base.iterdir() 
                  if f.is_dir() and f.name.startswith("bmw_complete")]
    
    if not raw_folders:
        print("❌ Nie znaleziono żadnych danych!")
        return None
    
    latest_raw = sorted(raw_folders)[-1]
    print(f"📁 Znaleziono surowe dane: {latest_raw.name}")
    print("⚠️  Najpierw uruchom normalizację danych!")
    return latest_raw

def analyze_output():
    """Analizuje dostępne dane"""
    latest_folder = find_latest_data()
    
    if latest_folder:
        # Sprawdź jakie pliki są dostępne
        files = list(latest_folder.glob("*.json*"))
        files.extend(latest_folder.glob("*.txt"))
        
        print(f"📊 Znaleziono {len(files)} plików:")
        for file in files[:10]:  # Pokaż tylko pierwsze 10
            size_kb = file.stat().st_size / 1024
            print(f"   - {file.name} ({size_kb:.1f} KB)")
        
        if len(files) > 10:
            print(f"   ... i {len(files) - 10} więcej")
        
        return True
    return False

def prepare_for_rag():
    """Przygotowuje dane dla RAG"""
    # Znajdź znormalizowane dane
    output_base = Path(r"C:\Users\hellb\Documents\Chatbot_BMW\RAG\output")
    
    # Szukaj folderów z "_normalized"
    normalized_folders = [f for f in output_base.iterdir() 
                         if f.is_dir() and "_normalized" in f.name]
    
    if not normalized_folders:
        print("❌ Nie znaleziono znormalizowanych danych!")
        print("   Najpierw uruchom normalizację (2_normalizer.py).")
        return
    
    latest_normalized = sorted(normalized_folders)[-1]
    
    # Wczytaj znormalizowane dane
    all_file = latest_normalized / "all_normalized.jsonl"
    if not all_file.exists():
        print(f"❌ Brak pliku: {all_file}")
        print(f"   Sprawdzam inne pliki w {latest_normalized}:")
        for file in latest_normalized.iterdir():
            print(f"   - {file.name}")
        return
    
    # Wczytaj dane
    data_items = []
    with open(all_file, 'r', encoding='utf-8') as f:
        for line in f:
            data_items.append(json.loads(line))
    
    print(f"\n📋 Wczytano {len(data_items)} znormalizowanych stron z: {latest_normalized.name}")
    
    # Stwórz chunk-i
    preprocessor = RAGPreprocessor(chunk_size=800, chunk_overlap=150)
    
    # Podziel na model i inne
    model_items = [item for item in data_items if item.get('is_model_page', False)]
    other_items = [item for item in data_items if not item.get('is_model_page', False)]
    
    print(f"   🚗 Strony z modelami: {len(model_items)}")
    print(f"   📰 Inne strony: {len(other_items)}")
    
    # Stwórz chunk-i dla modeli
    print("\n✂️ Tworzę chunk-i dla modeli...")
    model_chunks = preprocessor.create_chunks(model_items)
    
    # Stwórz chunk-i dla innych treści
    print("✂️ Tworzę chunk-i dla innych treści...")
    other_chunks = preprocessor.create_chunks(other_items)
    
    # Połącz
    all_chunks = model_chunks + other_chunks
    
    if not all_chunks:
        print("❌ Nie udało się utworzyć żadnych chunk-ów!")
        return
    
    # Zapisz
    rag_output = latest_normalized.parent / f"rag_ready_{latest_normalized.name.replace('_normalized', '')}"
    rag_output.mkdir(exist_ok=True)
    
    # Zapisz wszystkie chunk-i
    chunks_file = rag_output / "all_chunks.jsonl"
    print(f"\n💾 Zapisuję chunk-i do: {rag_output}")
    preprocessor.save_for_rag(all_chunks, chunks_file)
    
    # Zapisz też osobno modele
    if model_chunks:
        models_file = rag_output / "model_chunks.jsonl"
        preprocessor.save_for_rag(model_chunks, models_file)
    
    # Zapisz też osobno inne
    if other_chunks:
        others_file = rag_output / "other_chunks.jsonl"
        preprocessor.save_for_rag(other_chunks, others_file)
    
    # Pokaż przykładowy chunk
    if all_chunks:
        print(f"\n📋 PRZYKŁADOWY CHUNK:")
        sample = all_chunks[0]
        print(f"   ID: {sample['id']}")
        print(f"   Tytuł: {sample['metadata'].get('title', '')[:60]}...")
        print(f"   Modele: {sample['metadata'].get('models', [])}")
        print(f"   Rozmiar: {sample['metadata'].get('chunk_char_count', 0)} znaków")
        print(f"   Text preview: {sample['text'][:100]}...")
    
    return all_chunks

if __name__ == "__main__":
    print("=" * 60)
    print("🤖 PRZYGOTOWANIE DANYCH DLA RAG - BMW CHATBOT")
    print("=" * 60)
    
    # Sprawdź czy są dane
    print("\n🔍 Szukam danych...")
    if not analyze_output():
        print("❌ Brak danych. Najpierw uruchom crawlera (scraper_zle_ceny.py).")
        exit()
    
    # Krok 2: Przygotowanie dla RAG
    print("\n" + "="*60)
    prepare_rag = input("🤖 Czy przygotować dane dla RAG? (t/n): ")
    
    if prepare_rag.lower() == 't':
        print("\n🚀 Rozpoczynam przygotowanie danych dla RAG...")
        chunks = prepare_for_rag()
        
        if chunks:
            print(f"\n✅ Gotowe! Masz {len(chunks)} chunk-i do embeddingów.")
            print(f"   Następny krok: stworzenie embeddingów i wektorowej bazy danych.")
            
            # Sprawdź gdzie zapisano dane
            output_base = Path(r"C:\Users\hellb\Documents\Chatbot_BMW\RAG\output")
            rag_folders = [f for f in output_base.iterdir() 
                          if f.is_dir() and "rag_ready" in f.name]
            
            if rag_folders:
                latest_rag = sorted(rag_folders)[-1]
                print(f"\n📁 Twoje dane RAG są w: {latest_rag}")
                print(f"   Uruchom teraz 4_embeddings.py aby stworzyć embeddingi.")
        else:
            print("❌ Nie udało się przygotować danych dla RAG.")
    else:
        print("❌ Anulowano przygotowanie danych.")