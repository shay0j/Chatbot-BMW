import faiss
import numpy as np
import json
import pickle
from pathlib import Path
from datetime import datetime
import logging
from typing import List, Dict, Any
import sqlite3

class VectorDatabase:
    """Tworzy i zarządza wektorową bazą danych"""
    
    def __init__(self, dimension: int = 768):
        self.dimension = dimension
        self.index = None
        self.chunks = []
        self.metadata = []
        
    def create_faiss_index(self, embeddings: np.ndarray, index_type: str = "FlatIP"):
        """
        Tworzy indeks FAISS
        index_type: "FlatIP" (cosine similarity), "FlatL2" (Euclidean), "IVFFlat" (dla dużych zbiorów)
        """
        print(f"🔧 Tworzę indeks FAISS ({index_type}) dla {len(embeddings)} wektorów...")
        
        # Normalizuj wektory dla cosine similarity
        if index_type == "FlatIP":
            faiss.normalize_L2(embeddings)
            self.index = faiss.IndexFlatIP(self.dimension)
        elif index_type == "FlatL2":
            self.index = faiss.IndexFlatL2(self.dimension)
        elif index_type == "IVFFlat":
            # IVFFlat jest szybszy dla dużych zbiorów
            quantizer = faiss.IndexFlatL2(self.dimension)
            nlist = min(100, len(embeddings) // 39)  # Liczba centroidów
            self.index = faiss.IndexIVFFlat(quantizer, self.dimension, nlist, faiss.METRIC_L2)
            self.index.train(embeddings)
        else:
            raise ValueError(f"Nieznany typ indeksu: {index_type}")
        
        # Dodaj wektory do indeksu
        self.index.add(embeddings)
        
        print(f"✅ Indeks FAISS utworzony: {self.index.ntotal} wektorów")
        return self.index
    
    def add_metadata(self, chunks: List[Dict[str, Any]]):
        """Dodaje metadane do bazy"""
        self.chunks = chunks
        self.metadata = []
        
        for i, chunk in enumerate(chunks):
            meta = {
                'id': chunk.get('id', f'chunk_{i}'),
                'chunk_index': i,
                'text': chunk['text'],
                'metadata': chunk['metadata'],
                'embedding_timestamp': datetime.now().isoformat()
            }
            self.metadata.append(meta)
        
        print(f"📋 Dodano metadane dla {len(self.metadata)} chunk-ów")
    
    def search_similar(self, query_embedding: np.ndarray, k: int = 5):
        """Szuka podobnych wektorów"""
        if self.index is None:
            raise ValueError("Indeks nie został zainicjalizowany")
        
        # Normalizuj query dla cosine similarity
        if isinstance(self.index, faiss.IndexFlatIP):
            query_embedding = query_embedding.astype('float32')
            faiss.normalize_L2(query_embedding.reshape(1, -1))
        
        # Wyszukaj
        distances, indices = self.index.search(query_embedding.reshape(1, -1), k)
        
        # Przygotuj wyniki
        results = []
        for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
            if idx < len(self.metadata):
                chunk_meta = self.metadata[idx]
                results.append({
                    'rank': i + 1,
                    'similarity_score': float(distance),
                    'chunk_id': chunk_meta['id'],
                    'text': chunk_meta['text'],
                    'metadata': chunk_meta['metadata']
                })
        
        return results
    
    def save_database(self, output_path: Path):
        """Zapisuje całą bazę danych"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 1. Zapisz indeks FAISS
        faiss_file = output_path.with_suffix('.faiss')
        faiss.write_index(self.index, str(faiss_file))
        
        # 2. Zapisz metadane
        metadata_file = output_path.with_suffix('.metadata.pkl')
        with open(metadata_file, 'wb') as f:
            pickle.dump({
                'chunks': self.chunks,
                'metadata': self.metadata,
                'dimension': self.dimension,
                'total_vectors': self.index.ntotal,
                'created_at': datetime.now().isoformat()
            }, f)
        
        # 3. Zapisz do SQLite dla łatwego dostępu
        sqlite_file = output_path.with_suffix('.db')
        self._save_to_sqlite(sqlite_file)
        
        # 4. Zapisz statystyki
        stats = self._generate_stats()
        stats_file = output_path.with_suffix('.stats.json')
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
        
        print(f"💾 Zapisano bazę wektorową:")
        print(f"   🗂️  FAISS index: {faiss_file} ({faiss_file.stat().st_size / 1024 / 1024:.2f} MB)")
        print(f"   📄 Metadata: {metadata_file}")
        print(f"   🗃️  SQLite: {sqlite_file}")
        print(f"   📊 Statystyki: {stats_file}")
        
        return {
            'faiss': faiss_file,
            'metadata': metadata_file,
            'sqlite': sqlite_file,
            'stats': stats_file
        }
    
    def _save_to_sqlite(self, db_path: Path):
        """Zapisuje do SQLite"""
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Tabela chunks
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS chunks (
            id TEXT PRIMARY KEY,
            chunk_index INTEGER,
            text TEXT,
            title TEXT,
            source_url TEXT,
            models TEXT,
            categories TEXT,
            is_model_page INTEGER,
            has_prices INTEGER,
            has_specs INTEGER,
            chunk_size INTEGER,
            token_estimate INTEGER,
            similarity_info TEXT
        )
        ''')
        
        # Tabela search_log
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS search_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            query TEXT,
            results_count INTEGER,
            search_timestamp TEXT
        )
        ''')
        
        # Wstaw dane
        for meta in self.metadata:
            cursor.execute('''
            INSERT OR REPLACE INTO chunks VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                meta['id'],
                meta['chunk_index'],
                meta['text'],
                meta['metadata'].get('title', ''),
                meta['metadata'].get('source_url', ''),
                ', '.join(meta['metadata'].get('models', [])),
                ', '.join(meta['metadata'].get('categories', [])),
                1 if meta['metadata'].get('is_model_page', False) else 0,
                1 if meta['metadata'].get('prices') else 0,
                1 if meta['metadata'].get('engine_specs') else 0,
                meta['metadata'].get('chunk_char_count', 0),
                meta['metadata'].get('chunk_token_estimate', 0),
                json.dumps({'total_chunks': meta['metadata'].get('total_chunks', 0)})
            ))
        
        conn.commit()
        conn.close()
    
    def _generate_stats(self):
        """Generuje statystyki bazy"""
        stats = {
            'total_chunks': len(self.metadata),
            'total_vectors': self.index.ntotal if self.index else 0,
            'dimension': self.dimension,
            'model_chunks': sum(1 for m in self.metadata if m['metadata'].get('is_model_page', False)),
            'other_chunks': sum(1 for m in self.metadata if not m['metadata'].get('is_model_page', False)),
            'chunks_with_prices': sum(1 for m in self.metadata if m['metadata'].get('prices')),
            'chunks_with_specs': sum(1 for m in self.metadata if m['metadata'].get('engine_specs')),
            'avg_text_length': np.mean([len(m['text']) for m in self.metadata]) if self.metadata else 0,
            'created_at': datetime.now().isoformat()
        }
        
        # Statystyki modeli
        all_models = []
        for meta in self.metadata:
            all_models.extend(meta['metadata'].get('models', []))
        
        if all_models:
            from collections import Counter
            model_counts = Counter(all_models)
            stats['top_models'] = dict(model_counts.most_common(10))
        
        return stats

def find_embedding_data():
    """Znajduje najnowsze embeddingi"""
    output_base = Path(r"C:\Users\hellb\Documents\Chatbot_BMW\RAG\output")
    
    # Szukaj embeddingów
    embedding_files = []
    for folder in output_base.iterdir():
        if folder.is_dir() and "rag_ready" in folder.name:
            for file in folder.glob("embeddings_*.pkl"):
                embedding_files.append((folder, file))
    
    if not embedding_files:
        print("❌ Nie znaleziono embeddingów!")
        print("   Najpierw uruchom 4_embeddings.py")
        return None
    
    # Sortuj po czasie modyfikacji
    embedding_files.sort(key=lambda x: x[1].stat().st_mtime, reverse=True)
    
    latest_folder, latest_file = embedding_files[0]
    print(f"📁 Znaleziono embeddingi: {latest_file.name}")
    print(f"   W folderze: {latest_folder.name}")
    
    return latest_folder, latest_file

def load_embedding_data(embedding_file: Path):
    """Wczytuje dane embeddingów"""
    print(f"📖 Wczytuję: {embedding_file.name}")
    
    with open(embedding_file, 'rb') as f:
        data = pickle.load(f)
    
    chunks = data['chunks']
    embeddings = data['embeddings']
    metadata = data.get('metadata', {})
    
    print(f"   ✅ Chunk-i: {len(chunks)}")
    print(f"   ✅ Embeddingi: {len(embeddings)}")
    print(f"   ✅ Wymiar: {embeddings.shape[1]}")
    print(f"   ✅ Model: {metadata.get('model_name', 'Nieznany')}")
    
    return chunks, embeddings, metadata

def create_vector_database():
    """Tworzy wektorową bazę danych"""
    # Znajdź embeddingi
    result = find_embedding_data()
    if not result:
        return
    
    folder, embedding_file = result
    
    # Wczytaj dane
    chunks, embeddings, metadata = load_embedding_data(embedding_file)
    
    if len(chunks) == 0 or len(embeddings) == 0:
        print("❌ Brak danych do przetworzenia")
        return
    
    # Wybierz typ indeksu
    print(f"\n🔧 Dostępne typy indeksu FAISS:")
    print("   1. FlatIP - Cosine similarity (najlepsza jakość)")
    print("   2. FlatL2 - Euclidean distance (dokładny)")
    print("   3. IVFFlat - Szybki dla dużych zbiorów (przybliżony)")
    
    index_choice = input("🎯 Wybierz typ indeksu (1-3) lub Enter dla domyślnego (1): ")
    
    index_map = {
        '1': 'FlatIP',
        '2': 'FlatL2',
        '3': 'IVFFlat'
    }
    
    index_type = index_map.get(index_choice.strip() or '1', 'FlatIP')
    
    # Tworzenie bazy
    print(f"\n🏗️  Tworzę wektorową bazę danych...")
    print(f"   Typ indeksu: {index_type}")
    print(f"   Wymiar: {embeddings.shape[1]}")
    
    # Sprawdź czy zainstalowany faiss
    try:
        import faiss
    except ImportError:
        print("❌ Brak FAISS!")
        print("   Zainstaluj: pip install faiss-cpu")
        return
    
    # Inicjalizuj bazę
    db = VectorDatabase(dimension=embeddings.shape[1])
    
    # Stwórz indeks
    db.create_faiss_index(embeddings, index_type=index_type)
    
    # Dodaj metadane
    db.add_metadata(chunks)
    
    # Zapisz bazę
    print(f"\n💾 Zapisuję bazę danych...")
    db_name = f"vector_db_{index_type}"
    output_path = folder / db_name
    
    saved_files = db.save_database(output_path)
    
    # Testuj wyszukiwanie
    print(f"\n🧪 Testuję wyszukiwanie...")
    
    # Użyj pierwszego embeddingu jako query
    if len(embeddings) > 1:
        query_embedding = embeddings[0]
        results = db.search_similar(query_embedding, k=3)
        
        if results:
            print(f"   🔎 Wyniki wyszukiwania (najbardziej podobne):")
            for result in results[:2]:  # Pokaż 2 pierwsze
                print(f"      #{result['rank']}: similarity={result['similarity_score']:.3f}")
                print(f"         ID: {result['chunk_id']}")
                print(f"         Text: {result['text'][:100]}...")
                print()
    
    # Podsumowanie
    print(f"\n✅ PODSUMOWANIE:")
    print(f"   🏢 Baza danych gotowa!")
    print(f"   📊 Chunk-i: {len(chunks)}")
    print(f"   🔢 Wektory: {embeddings.shape[0]}")
    print(f"   🗂️  Zapisano w: {folder}")
    print(f"   🚗 Modele BMW: {len([c for c in chunks if c['metadata'].get('is_model_page', False)])}")
    
    # Info o następnych krokach
    print(f"\n🎉 Wszystko gotowe! Możesz teraz:")
    print(f"   1. Uruchomić 6_rag_test.py aby przetestować RAG")
    print(f"   2. Przetestować wyszukiwanie za pomocą test_search.py")
    
    return db

if __name__ == "__main__":
    print("=" * 60)
    print("🏢 TWORZENIE WEKTOROWEJ BAZY DANYCH - BMW RAG")
    print("=" * 60)
    
    # Sprawdź czy są embeddingi
    result = find_embedding_data()
    if not result:
        exit()
    
    # Zapytaj czy tworzyć bazę
    print("\n" + "="*60)
    create_db = input("🏢 Czy chcesz stworzyć wektorową bazę danych? (t/n): ")
    
    if create_db.lower() == 't':
        print("\n🚀 Rozpoczynam tworzenie bazy...")
        db = create_vector_database()
        
        if db:
            print(f"\n🎉 Baza danych gotowa do użycia w RAG systemie!")
            print(f"   Możesz teraz zbudować chatbota BMW.")
    else:
        print("❌ Anulowano tworzenie bazy.")