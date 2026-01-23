import faiss
import numpy as np
import pickle
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from sentence_transformers import SentenceTransformer
from datetime import datetime
import logging

# Konfiguracja logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RAGSystem:
    """Kompletny system RAG dla BMW"""
    
    def __init__(self, vector_db_path: Optional[Path] = None, model_name: str = None):
        """
        Inicjalizuje system RAG
        """
        self.vector_db_path = vector_db_path
        self.model_name = model_name or 'paraphrase-multilingual-mpnet-base-v2'
        self.index = None
        self.metadata = []
        self.chunks = []
        self.model = None
        self.embedding_dim = None
        
        # Ładowanie modelu embeddingów
        self._load_embedding_model()
        
        # Ładowanie bazy danych
        if vector_db_path:
            self.load_vector_database(vector_db_path)
    
    def _load_embedding_model(self):
        """Ładuje model do embeddingów"""
        logger.info(f"🔄 Ładuję model embeddingów: {self.model_name}")
        try:
            self.model = SentenceTransformer(self.model_name)
            # Test embedding dimension
            test_embedding = self.model.encode(["test"])
            self.embedding_dim = test_embedding.shape[1]
            logger.info(f"✅ Model załadowany. Wymiar: {self.embedding_dim}")
        except Exception as e:
            logger.error(f"❌ Błąd ładowania modelu: {e}")
            raise
    
    def load_vector_database(self, db_path: Path):
        """Ładuje wektorową bazę danych"""
        logger.info(f"📂 Ładuję bazę danych z: {db_path}")
        
        # Szukaj plików
        faiss_file = None
        metadata_file = None
        
        for suffix in ['.faiss', '.index']:
            potential_file = db_path.with_suffix(suffix)
            if potential_file.exists():
                faiss_file = potential_file
                break
        
        for suffix in ['.metadata.pkl', '.pkl']:
            potential_file = db_path.with_suffix(suffix)
            if potential_file.exists():
                metadata_file = potential_file
                break
        
        if not faiss_file or not metadata_file:
            # Spróbuj znaleźć w folderze
            folder = db_path if db_path.is_dir() else db_path.parent
            faiss_files = list(folder.glob("*.faiss"))
            pkl_files = list(folder.glob("*.pkl"))
            
            if faiss_files:
                faiss_file = faiss_files[0]
            if pkl_files:
                metadata_file = pkl_files[0]
        
        if not faiss_file:
            raise FileNotFoundError(f"Nie znaleziono pliku FAISS w: {db_path}")
        
        if not metadata_file:
            raise FileNotFoundError(f"Nie znaleziono pliku metadanych w: {db_path}")
        
        # Ładuj indeks FAISS
        logger.info(f"📊 Ładuję indeks FAISS: {faiss_file.name}")
        self.index = faiss.read_index(str(faiss_file))
        
        # Ładuj metadane
        logger.info(f"📋 Ładuję metadane: {metadata_file.name}")
        with open(metadata_file, 'rb') as f:
            metadata_data = pickle.load(f)
        
        self.chunks = metadata_data.get('chunks', [])
        self.metadata = metadata_data.get('metadata', [])
        
        logger.info(f"✅ Załadowano bazę: {self.index.ntotal} wektorów, {len(self.chunks)} chunk-ów")
        
        return True
    
    def query(self, question: str, k: int = 5, filters: Optional[Dict] = None) -> List[Dict]:
        """
        Wykonuje zapytanie do systemu RAG
        
        Args:
            question: Pytanie użytkownika
            k: Liczba wyników do zwrócenia
            filters: Filtry do zastosowania (np. {'is_model_page': True})
        
        Returns:
            Lista wyników z podobieństwami i metadanymi
        """
        logger.info(f"🔍 Przetwarzam zapytanie: '{question}'")
        
        # Generuj embedding dla pytania
        question_embedding = self.model.encode([question])
        
        # Normalizuj dla cosine similarity
        if isinstance(self.index, faiss.IndexFlatIP):
            faiss.normalize_L2(question_embedding)
        
        # Wyszukaj podobne wektory
        distances, indices = self.index.search(question_embedding, k)
        
        # Przygotuj wyniki
        results = []
        for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
            if idx < len(self.metadata):
                chunk_meta = self.metadata[idx]
                
                # Zastosuj filtry jeśli podane
                if filters:
                    skip = False
                    for key, value in filters.items():
                        if key in chunk_meta['metadata']:
                            if chunk_meta['metadata'][key] != value:
                                skip = True
                                break
                    if skip:
                        continue
                
                # Przygotuj odpowiedź
                result = {
                    'rank': i + 1,
                    'similarity_score': float(distance),
                    'chunk_id': chunk_meta['id'],
                    'text': chunk_meta['text'],
                    'metadata': chunk_meta['metadata'],
                    'source_info': {
                        'title': chunk_meta['metadata'].get('title', ''),
                        'url': chunk_meta['metadata'].get('source_url', ''),
                        'models': chunk_meta['metadata'].get('models', []),
                        'categories': chunk_meta['metadata'].get('categories', [])
                    }
                }
                results.append(result)
        
        logger.info(f"✅ Znaleziono {len(results)} wyników")
        return results
    
    def generate_answer(self, question: str, context_results: List[Dict], 
                       max_context_length: int = 3000) -> Dict:
        """
        Generuje odpowiedź na podstawie znalezionych kontekstów
        
        Args:
            question: Pytanie użytkownika
            context_results: Wyniki z query()
            max_context_length: Maksymalna długość kontekstu
            
        Returns:
            Słownik z odpowiedzią i źródłami
        """
        if not context_results:
            return {
                'answer': "Przepraszam, nie znalazłem odpowiednich informacji w bazie danych.",
                'sources': [],
                'confidence': 0.0
            }
        
        # Przygotuj kontekst
        context_parts = []
        total_length = 0
        sources = []
        
        for result in context_results:
            if total_length + len(result['text']) <= max_context_length:
                context_parts.append(result['text'])
                total_length += len(result['text'])
                
                # Dodaj źródło
                source_info = {
                    'title': result['metadata'].get('title', ''),
                    'url': result['metadata'].get('source_url', ''),
                    'similarity': result['similarity_score'],
                    'models': result['metadata'].get('models', [])
                }
                sources.append(source_info)
            else:
                break
        
        context = "\n\n".join(context_parts)
        
        # Prosta odpowiedź (w pełnym RAG użyłbyś LLM jak GPT)
        answer = self._generate_simple_answer(question, context, sources)
        
        return {
            'answer': answer,
            'sources': sources,
            'context_length': total_length,
            'results_used': len(context_parts),
            'confidence': np.mean([r['similarity_score'] for r in context_results[:len(context_parts)]])
        }
    
    def _generate_simple_answer(self, question: str, context: str, sources: List[Dict]) -> str:
        """
        Prosta generacja odpowiedzi bez LLM
        """
        question_lower = question.lower()
        
        # Dla różnych typów pytań
        if any(word in question_lower for word in ['cena', 'koszt', 'ile kosztuje', 'cena od']):
            # Szukaj informacji o cenach w kontekście
            price_lines = []
            for line in context.split('\n'):
                if any(price_word in line.lower() for price_word in ['zł', 'pln', 'cena', 'od', 'euro', '€']):
                    price_lines.append(line.strip())
            
            if price_lines:
                return f"Na podstawie znalezionych informacji o cenach BMW:\n\n" + "\n".join(price_lines[:3])
        
        elif any(word in question_lower for word in ['moc', 'silnik', 'km', 'koni']):
            # Szukaj informacji o mocy
            engine_lines = []
            for line in context.split('\n'):
                if any(engine_word in line.lower() for engine_word in ['km', 'koni', 'moc', 'silnik', 'hp', 'ps']):
                    engine_lines.append(line.strip())
            
            if engine_lines:
                return f"Informacje o parametrach silnika:\n\n" + "\n".join(engine_lines[:3])
        
        elif any(word in question_lower for word in ['model', 'modele', 'seria', 'x1', 'x3', 'x5']):
            # Dla pytań o modele
            return f"Znalezione informacje o modelach BMW:\n\n{context[:500]}..."
        
        # Ogólna odpowiedź
        return f"Oto informacje na temat '{question}':\n\n{context[:800]}..."
    
    def get_database_info(self) -> Dict:
        """Zwraca informacje o bazie danych"""
        return {
            'total_vectors': self.index.ntotal if self.index else 0,
            'total_chunks': len(self.chunks),
            'embedding_dim': self.embedding_dim,
            'index_type': str(type(self.index).__name__) if self.index else 'None',
            'model_name': self.model_name,
            'loaded_at': datetime.now().isoformat()
        }

def find_latest_vector_db():
    """Znajduje najnowszą bazę wektorową"""
    output_base = Path(r"C:\Users\hellb\Documents\Chatbot_BMW\RAG\output")
    
    # Szukaj plików .faiss
    faiss_files = []
    for folder in output_base.iterdir():
        if folder.is_dir():
            for file in folder.glob("*.faiss"):
                faiss_files.append((folder, file))
    
    if not faiss_files:
        logger.error("❌ Nie znaleziono baz danych FAISS")
        return None
    
    # Sortuj po czasie modyfikacji
    faiss_files.sort(key=lambda x: x[1].stat().st_mtime, reverse=True)
    
    latest_folder, latest_file = faiss_files[0]
    
    # Znajdź odpowiadający plik metadanych
    metadata_file = None
    for suffix in ['.metadata.pkl', '.pkl']:
        potential_file = latest_file.with_suffix(suffix)
        if potential_file.exists():
            metadata_file = potential_file
            break
    
    if not metadata_file:
        # Szukaj w folderze
        pkl_files = list(latest_folder.glob("*.pkl"))
        if pkl_files:
            metadata_file = pkl_files[0]
    
    logger.info(f"📁 Znaleziono bazę: {latest_file.name}")
    logger.info(f"   Folder: {latest_folder.name}")
    logger.info(f"   Metadata: {metadata_file.name if metadata_file else 'Nie znaleziono'}")
    
    return latest_file

def test_rag_system():
    """Testuje system RAG"""
    print("=" * 70)
    print("🧪 TEST SYSTEMU RAG - BMW CHATBOT")
    print("=" * 70)
    
    # Znajdź bazę danych
    db_file = find_latest_vector_db()
    if not db_file:
        return
    
    # Utwórz system RAG
    print(f"\n🚀 Inicjalizuję system RAG z bazą: {db_file.name}")
    
    try:
        rag = RAGSystem(vector_db_path=db_file)
    except Exception as e:
        print(f"❌ Błąd inicjalizacji: {e}")
        return
    
    # Pokaż informacje o bazie
    db_info = rag.get_database_info()
    print(f"\n📊 INFORMACJE O BAZIE:")
    print(f"   Wektory: {db_info['total_vectors']}")
    print(f"   Chunk-i: {db_info['total_chunks']}")
    print(f"   Wymiar: {db_info['embedding_dim']}")
    print(f"   Model: {db_info['model_name']}")
    
    # Testowe pytania
    test_questions = [
        "Ile kosztuje BMW X3?",
        "Jaka jest moc silnika w BMW X5?",
        "Jakie są modele elektryczne BMW?",
        "Ile wynosi przyspieszenie 0-100 km/h w BMW i4?",
        "Jakie są ceny BMW serii 3?",
        "Czy BMW X1 jest dostępne jako hybryda?",
        "Jaki jest zasięg BMW iX?",
        "Gdzie mogę znaleźć serwis BMW w Polsce?",
        "Jakie są opcje finansowania BMW?",
        "Czym się różni BMW M3 od zwykłej serii 3?"
    ]
    
    print(f"\n🎯 TESTUJĘ {len(test_questions)} PYTAŃ:")
    print("=" * 70)
    
    for i, question in enumerate(test_questions, 1):
        print(f"\n{i}. ❓ PYTANIE: {question}")
        
        try:
            # Wyszukaj podobne fragmenty
            results = rag.query(question, k=3)
            
            if results:
                # Wygeneruj odpowiedź
                answer_data = rag.generate_answer(question, results)
                
                print(f"   ✅ ODPOWIEDŹ: {answer_data['answer'][:200]}...")
                print(f"   📊 Ufność: {answer_data['confidence']:.3f}")
                print(f"   📚 Źródła: {len(answer_data['sources'])}")
                
                # Pokaż najlepsze źródło
                if answer_data['sources']:
                    best_source = answer_data['sources'][0]
                    print(f"   🏆 Najlepsze źródło: {best_source['title'][:50]}...")
                    if best_source['models']:
                        print(f"      Modele: {', '.join(best_source['models'][:3])}")
            else:
                print(f"   ⚠️  Brak wyników")
                
        except Exception as e:
            print(f"   ❌ Błąd: {e}")
    
    # Interaktywny tryb
    print(f"\n" + "=" * 70)
    print("💬 TRYB INTERAKTYWNY")
    print("=" * 70)
    
    while True:
        question = input("\n🎯 Twoje pytanie o BMW (lub 'exit' aby zakończyć): ")
        
        if question.lower() in ['exit', 'quit', 'q']:
            print("👋 Do widzenia!")
            break
        
        if not question.strip():
            continue
        
        print(f"\n🔍 Szukam odpowiedzi...")
        
        try:
            # Wyszukaj z filtrami dla modeli
            filters = None
            if any(model in question.lower() for model in ['x3', 'x5', 'x1', 'i4', 'ix', 'm3']):
                # Można dodać filtry specyficzne dla modelu
                pass
            
            results = rag.query(question, k=5, filters=filters)
            
            if results:
                answer_data = rag.generate_answer(question, results)
                
                print(f"\n✅ ODPOWIEDŹ:")
                print(f"   {answer_data['answer']}")
                print(f"\n📊 Statystyki:")
                print(f"   Ufność: {answer_data['confidence']:.3f}")
                print(f"   Źródła: {len(answer_data['sources'])}")
                print(f"   Długość kontekstu: {answer_data['context_length']} znaków")
                
                print(f"\n📚 ŹRÓDŁA:")
                for i, source in enumerate(answer_data['sources'][:3], 1):
                    print(f"   {i}. {source['title'][:60]}...")
                    if source['models']:
                        print(f"      Modele: {', '.join(source['models'][:3])}")
                    print(f"      Podobieństwo: {source['similarity']:.3f}")
            else:
                print(f"❌ Nie znalazłem odpowiednich informacji.")
                
        except Exception as e:
            print(f"❌ Błąd: {e}")

if __name__ == "__main__":
    test_rag_system()