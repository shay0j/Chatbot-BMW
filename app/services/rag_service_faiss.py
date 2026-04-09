"""
Serwis RAG używający FAISS + Cohere embeddings (bez ONNXRUNTIME)
"""
import asyncio
import os
import pickle
import numpy as np
import hashlib
import json
import re
import time
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
from pathlib import Path
import httpx
from dotenv import load_dotenv

load_dotenv()

# ============================================
# CONFIG
# ============================================

class Settings:
    COHERE_API_KEY = os.getenv("COHERE_API_KEY")
    COHERE_EMBED_MODEL = os.getenv("COHERE_EMBED_MODEL", "embed-multilingual-v3.0")
    SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.5"))
    TOP_K_DOCUMENTS = int(os.getenv("TOP_K_DOCUMENTS", "3"))
    FAISS_INDEX_PATH = os.getenv("FAISS_INDEX_PATH", "RAG/faiss_index")
    RAG_SOURCES_DIR = os.getenv("RAG_SOURCES_DIR", "RAG/RAG_sources")
    EMBEDDING_BATCH_SIZE = int(os.getenv("EMBEDDING_BATCH_SIZE", "10"))  # Mniejsze partie
    
    @property
    def FAISS_INDEX_PATH_OBJ(self) -> Path:
        return Path(self.FAISS_INDEX_PATH).absolute()
    
    @property
    def RAG_SOURCES_DIR_OBJ(self) -> Path:
        return Path(self.RAG_SOURCES_DIR).absolute()

settings = Settings()

# ============================================
# LOGGER
# ============================================

class Logger:
    def info(self, msg): print(f"[INFO] {msg}")
    def warning(self, msg): print(f"[WARNING] {msg}")
    def error(self, msg): print(f"[ERROR] {msg}")
    def debug(self, msg): print(f"[DEBUG] {msg}")

log = Logger()

# ============================================
# POMOCNICZA FUNKCJA DO CZYTANIA PLIKÓW
# ============================================

def read_file_with_fallback(file_path: Path) -> str:
    """Próbuje czytać plik z różnymi kodowaniami"""
    encodings = ['utf-8', 'windows-1250', 'iso-8859-2', 'latin1', 'cp1250', 'cp852']
    for encoding in encodings:
        try:
            with open(file_path, 'r', encoding=encoding) as f:
                return f.read()
        except (UnicodeDecodeError, UnicodeError):
            continue
    # Ostateczność - ignoruj błędy
    with open(file_path, 'r', encoding='latin1', errors='replace') as f:
        return f.read()

# ============================================
# COHERE EMBEDDINGS (przez API)
# ============================================

class CohereEmbedder:
    def __init__(self):
        self.api_key = settings.COHERE_API_KEY
        self.model = settings.COHERE_EMBED_MODEL
        self._client = None
    
    async def _get_client(self):
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=60.0)  # Zwiększony timeout
        return self._client
    
    async def embed_batch(self, texts: List[str], retry_count: int = 3) -> np.ndarray:
        """Embedding dla partii tekstów z retry"""
        if not texts:
            return np.array([])
        
        client = await self._get_client()
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "texts": texts,
            "model": self.model,
            "input_type": "search_document"
        }
        
        for attempt in range(retry_count):
            try:
                response = await client.post(
                    "https://api.cohere.ai/v1/embed",
                    headers=headers,
                    json=payload
                )
                
                if response.status_code == 200:
                    data = response.json()
                    embeddings = data.get("embeddings", [])
                    return np.array(embeddings, dtype=np.float32)
                else:
                    error_text = response.text if hasattr(response, 'text') else str(response.status_code)
                    log.warning(f"Cohere API error (attempt {attempt+1}/{retry_count}): {response.status_code} - {error_text[:100]}")
                    
                    if attempt < retry_count - 1:
                        await asyncio.sleep(2 ** attempt)  # Exponential backoff
                    else:
                        return np.array([])
                        
            except Exception as e:
                log.warning(f"Embedding error (attempt {attempt+1}/{retry_count}): {e}")
                if attempt < retry_count - 1:
                    await asyncio.sleep(2 ** attempt)
                else:
                    return np.array([])
        
        return np.array([])
    
    async def embed(self, texts: List[str]) -> np.ndarray:
        """Zwraca embeddingi dla listy tekstów (wspiera batch processing)"""
        if not texts:
            return np.array([])
        
        batch_size = settings.EMBEDDING_BATCH_SIZE
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            log.debug(f"  Embedding batch {i//batch_size + 1}/{(len(texts)-1)//batch_size + 1} ({len(batch)} docs)")
            
            embeddings = await self.embed_batch(batch)
            if len(embeddings) == 0:
                log.error(f"Failed to embed batch {i}")
                return np.array([])
            
            all_embeddings.append(embeddings)
            
            # Mała przerwa między batchami
            if i + batch_size < len(texts):
                await asyncio.sleep(0.3)
        
        return np.vstack(all_embeddings) if all_embeddings else np.array([])
    
    async def embed_query(self, query: str) -> np.ndarray:
        """Embedding dla pojedynczego zapytania"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "texts": [query],
            "model": self.model,
            "input_type": "search_query"
        }
        
        client = await self._get_client()
        try:
            response = await client.post(
                "https://api.cohere.ai/v1/embed",
                headers=headers,
                json=payload
            )
            
            if response.status_code == 200:
                data = response.json()
                embeddings = data.get("embeddings", [])
                return np.array(embeddings[0] if embeddings else [], dtype=np.float32)
            else:
                log.error(f"Query embedding error: {response.status_code}")
                return np.array([])
        except Exception as e:
            log.error(f"Query embedding error: {e}")
            return np.array([])

# ============================================
# FAISS VECTOR STORE
# ============================================

class FAISSVectorStore:
    def __init__(self):
        self.index = None
        self.documents = []
        self.embedder = CohereEmbedder()
        self.index_path = settings.FAISS_INDEX_PATH_OBJ / "faiss_index.bin"
        self.metadata_path = settings.FAISS_INDEX_PATH_OBJ / "metadata.pkl"
        self._load()
    
    def _load(self):
        """Ładuje istniejący indeks jeśli istnieje"""
        try:
            import faiss
            if self.index_path.exists() and self.metadata_path.exists():
                self.index = faiss.read_index(str(self.index_path))
                with open(self.metadata_path, 'rb') as f:
                    self.documents = pickle.load(f)
                log.info(f"✅ Załadowano FAISS indeks z {len(self.documents)} dokumentami")
                return True
        except ImportError:
            log.error("FAISS not installed: pip install faiss-cpu")
        except Exception as e:
            log.error(f"Error loading FAISS index: {e}")
        return False
    
    def _save(self):
        """Zapisuje indeks do pliku"""
        try:
            import faiss
            self.index_path.parent.mkdir(parents=True, exist_ok=True)
            faiss.write_index(self.index, str(self.index_path))
            with open(self.metadata_path, 'wb') as f:
                pickle.dump(self.documents, f)
            log.info(f"✅ Zapisano FAISS indeks z {len(self.documents)} dokumentami")
            return True
        except Exception as e:
            log.error(f"Error saving FAISS index: {e}")
            return False
    
    async def create_index(self, texts: List[str], metadatas: List[Dict] = None):
        """Tworzy nowy indeks z listy tekstów (z batch processing)"""
        try:
            import faiss
            
            if not texts:
                log.error("Brak tekstów do indeksowania")
                return False
            
            log.info(f"📊 Tworzenie embeddingów dla {len(texts)} dokumentów (batchami po {settings.EMBEDDING_BATCH_SIZE})...")
            
            # Tworzenie embeddingów z batch processing
            embeddings = await self.embedder.embed(texts)
            
            if len(embeddings) == 0 or embeddings.shape[0] != len(texts):
                log.error(f"Nie udało się stworzyć embeddingów. Oczekiwano {len(texts)}, otrzymano {len(embeddings) if len(embeddings) > 0 else 0}")
                return False
            
            dimension = embeddings.shape[1]
            self.index = faiss.IndexFlatIP(dimension)  # Inner Product (cosine similarity)
            
            # Normalizuj embeddingi dla cosine similarity
            faiss.normalize_L2(embeddings)
            self.index.add(embeddings)
            
            self.documents = []
            for i, text in enumerate(texts):
                # Skróć tekst jeśli za długi (dla przechowywania)
                content_preview = text[:5000] if len(text) > 5000 else text
                self.documents.append({
                    "content": content_preview,
                    "metadata": metadatas[i] if metadatas and i < len(metadatas) else {"id": i},
                    "id": f"doc_{i}"
                })
            
            self._save()
            log.info(f"✅ Utworzono indeks FAISS z {len(self.documents)} dokumentami")
            return True
            
        except ImportError:
            log.error("FAISS not installed. Run: pip install faiss-cpu")
            return False
        except Exception as e:
            log.error(f"Error creating index: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    async def search(self, query: str, top_k: int = 5) -> List[Tuple[Dict, float]]:
        """Wyszukuje podobne dokumenty"""
        if self.index is None:
            return []
        
        try:
            import faiss
            
            query_embedding = await self.embedder.embed_query(query)
            if len(query_embedding) == 0:
                return []
            
            # Normalizuj
            query_embedding = query_embedding.reshape(1, -1)
            faiss.normalize_L2(query_embedding)
            
            # Wyszukaj
            k = min(top_k, len(self.documents))
            distances, indices = self.index.search(query_embedding, k)
            
            results = []
            for i, idx in enumerate(indices[0]):
                if idx >= 0 and idx < len(self.documents):
                    # Cosine similarity: im bliżej 1 tym lepiej
                    similarity = float(distances[0][i])
                    results.append((self.documents[idx], similarity))
            
            return results
            
        except Exception as e:
            log.error(f"Search error: {e}")
            return []
    
    async def get_document_count(self) -> int:
        return len(self.documents)

# ============================================
# INTENT DETECTOR
# ============================================

class IntentDetector:
    def __init__(self):
        self.bmw_models = [
            "x1", "x2", "x3", "x4", "x5", "x6", "x7", "xm",
            "i3", "i4", "i5", "i7", "i8", "ix",
            "m2", "m3", "m4", "m5", "m8", "z4",
            "seria 1", "seria 2", "seria 3", "seria 4", "seria 5", "seria 7", "seria 8"
        ]
    
    def should_skip_rag(self, query: str) -> bool:
        greetings = ["cześć", "hej", "witam", "dzień dobry", "siema", "hello"]
        return query.lower().strip() in greetings
    
    def detect_intent(self, query: str) -> Dict[str, Any]:
        query_lower = query.lower()
        if self.should_skip_rag(query):
            return {"skip_rag": True, "primary_intent": "greeting", "detected_models": [], "is_technical": False}
        
        detected_models = []
        for model in self.bmw_models:
            if re.search(r'\b' + re.escape(model) + r'\b', query_lower):
                detected_models.append(model.upper())
        
        return {
            "skip_rag": False,
            "primary_intent": "general",
            "detected_models": detected_models,
            "is_technical": any(k in query_lower for k in ["moc", "silnik", "km", "nm", "0-100"])
        }

# ============================================
# RAG SERVICE
# ============================================

class RAGService:
    def __init__(self):
        self.vector_store = FAISSVectorStore()
        self.intent_detector = IntentDetector()
        self.top_k = settings.TOP_K_DOCUMENTS
        self._stats = {"queries": 0, "retrieved": 0}
        log.info("✅ RAGService gotowy (FAISS)")
    
    async def retrieve_with_intent_check(self, query: str, top_k: int = None) -> Dict[str, Any]:
        if top_k is None:
            top_k = self.top_k
        
        self._stats["queries"] += 1
        intent = self.intent_detector.detect_intent(query)
        
        if intent["skip_rag"]:
            return {"has_data": False, "skip_rag": True, "documents": [], "sources": []}
        
        results = await self.vector_store.search(query, top_k)
        self._stats["retrieved"] += len(results)
        
        documents = []
        sources = []
        for doc, score in results:
            documents.append({
                "content": doc["content"],
                "metadata": doc["metadata"],
                "score": score
            })
            sources.append({
                "title": doc["metadata"].get("model", doc["metadata"].get("title", doc["metadata"].get("filename", "Dokument"))),
                "content": doc["content"][:150],
                "score": score
            })
        
        return {
            "has_data": len(documents) > 0,
            "skip_rag": False,
            "confidence": documents[0]["score"] if documents else 0,
            "intent": intent["primary_intent"],
            "detected_models": intent["detected_models"],
            "tech": intent["is_technical"],
            "documents": documents,
            "sources": sources,
            "documents_retrieved": len(results)
        }
    
    async def health_check(self) -> Dict[str, Any]:
        return {
            "status": "healthy" if await self.vector_store.get_document_count() > 0 else "degraded",
            "documents_in_store": await self.vector_store.get_document_count(),
            "vector_store": "faiss"
        }
    
    async def get_stats(self) -> Dict[str, Any]:
        return {
            "documents_in_store": await self.vector_store.get_document_count(),
            "queries": self._stats["queries"],
            "retrieved": self._stats["retrieved"]
        }

# ============================================
# ŁADOWANIE Z KATALOGU (TXT + CSV)
# ============================================

async def initialize_vector_store_from_directory(
    directory_path: str = None,
    force_recreate: bool = True
) -> bool:
    """Tworzy bazę FAISS z wszystkich plików .txt i .csv w katalogu"""
    
    if not settings.COHERE_API_KEY:
        log.error("COHERE_API_KEY not set")
        return False
    
    if directory_path is None:
        directory = settings.RAG_SOURCES_DIR_OBJ
    else:
        directory = Path(directory_path)
    
    if not directory.exists():
        log.error(f"Katalog nie istnieje: {directory}")
        return False
    
    texts = []
    metadatas = []
    
    # Znajdź wszystkie pliki .txt i .csv
    txt_files = list(directory.glob("*.txt"))
    csv_files = list(directory.glob("*.csv"))
    
    log.info(f"📁 Katalog: {directory}")
    log.info(f"📄 Znaleziono: {len(txt_files)} plików TXT, {len(csv_files)} plików CSV")
    
    # === 1. ŁADOWANIE PLIKÓW TXT (z obsługą różnych kodowań) ===
    for txt_file in txt_files:
        try:
            # Użyj funkcji z fallback encoding
            content = read_file_with_fallback(txt_file)
            
            # Podziel na mniejsze kawałki jeśli za długie (maks 1500 znaków)
            chunks = []
            if len(content) > 1500:
                # Podziel na akapity
                paragraphs = content.split('\n\n')
                current_chunk = ""
                for para in paragraphs:
                    if len(current_chunk) + len(para) < 1500:
                        current_chunk += para + "\n\n"
                    else:
                        if current_chunk:
                            chunks.append(current_chunk.strip())
                        current_chunk = para + "\n\n"
                if current_chunk:
                    chunks.append(current_chunk.strip())
            else:
                chunks = [content]
            
            for i, chunk in enumerate(chunks):
                if chunk.strip():  # Tylko niepuste fragmenty
                    texts.append(chunk)
                    metadatas.append({
                        "source": str(txt_file),
                        "type": "txt",
                        "filename": txt_file.name,
                        "chunk": i,
                        "title": txt_file.stem.replace("_", " ").replace("-", " ").title()
                    })
            log.info(f"  ✅ {txt_file.name} - {len(chunks)} fragmentów")
        except Exception as e:
            log.error(f"  ❌ Błąd czytania {txt_file.name}: {e}")
    
    # === 2. ŁADOWANIE PLIKÓW CSV ===
    for csv_file in csv_files:
        try:
            import pandas as pd
            # Spróbuj różnych kodowań dla CSV
            df = None
            for encoding in ['utf-8', 'windows-1250', 'latin1']:
                try:
                    df = pd.read_csv(csv_file, encoding=encoding)
                    break
                except:
                    continue
            
            if df is None:
                log.error(f"  ❌ Nie udało się odczytać {csv_file.name}")
                continue
                
            log.info(f"  ✅ {csv_file.name} - {len(df)} wierszy (CSV)")
            
            for idx, row in df.iterrows():
                content_parts = []
                metadata = {
                    "source": str(csv_file),
                    "type": "csv",
                    "filename": csv_file.name,
                    "row_id": idx
                }
                
                for col in df.columns:
                    value = row[col]
                    if pd.notna(value) and str(value).strip():
                        content_parts.append(f"{col}: {value}")
                        if col.lower() in ["model", "series", "type", "category", "title"]:
                            metadata[col.lower()] = str(value)
                
                if content_parts:
                    texts.append("\n".join(content_parts))
                    metadatas.append(metadata)
        except Exception as e:
            log.error(f"  ❌ Błąd czytania {csv_file.name}: {e}")
    
    if not texts:
        log.error("Nie znaleziono żadnych dokumentów do indeksowania")
        return False
    
    log.info(f"📊 Utworzono {len(texts)} dokumentów z {len(txt_files) + len(csv_files)} plików")
    
    # Sprawdź czy chcemy wymusić recreate
    vector_store = FAISSVectorStore()
    current_count = await vector_store.get_document_count()
    
    if force_recreate or current_count == 0:
        # Utwórz nowy indeks
        success = await vector_store.create_index(texts, metadatas)
        if success:
            log.info(f"✅ BAZA WEKTOROWA GOTOWA! Dokumenty: {len(texts)}")
        else:
            log.error("❌ Błąd tworzenia bazy")
        return success
    else:
        log.info(f"ℹ️ Indeks już istnieje ({current_count} dokumentów). Użyj force_recreate=True aby przebudować.")
        return True

# ============================================
# INICJALIZACJA BAZY Z CSV (dla kompatybilności)
# ============================================

async def initialize_vector_store_from_csv(
    csv_path: str,
    force_recreate: bool = True
) -> bool:
    """Tworzy bazę FAISS z pliku CSV (dla kompatybilności wstecznej)"""
    
    if not settings.COHERE_API_KEY:
        log.error("COHERE_API_KEY not set")
        return False
    
    csv_path_obj = Path(csv_path)
    if not csv_path_obj.exists():
        log.error(f"Plik CSV nie istnieje: {csv_path_obj}")
        return False
    
    log.info(f"📁 Plik źródłowy: {csv_path_obj}")
    
    try:
        import pandas as pd
        df = pd.read_csv(csv_path_obj, encoding='utf-8')
        log.info(f"✅ Wczytano {len(df)} wierszy")
    except ImportError:
        log.error("Pandas not installed: pip install pandas")
        return False
    
    # Utwórz dokumenty
    texts = []
    metadatas = []
    for idx, row in df.iterrows():
        content_parts = []
        metadata = {"row_id": idx, "type": "csv", "source": str(csv_path_obj)}
        for col in df.columns:
            value = row[col]
            if pd.notna(value) and str(value).strip():
                content_parts.append(f"{col}: {value}")
                if col.lower() in ["model", "series", "type", "category", "title"]:
                    metadata[col.lower()] = str(value)
        if content_parts:
            texts.append("\n".join(content_parts))
            metadatas.append(metadata)
    
    log.info(f"📄 Utworzono {len(texts)} dokumentów")
    
    # Utwórz indeks FAISS
    vector_store = FAISSVectorStore()
    success = await vector_store.create_index(texts, metadatas)
    
    if success:
        log.info(f"✅ BAZA WEKTOROWA GOTOWA! Dokumenty: {len(texts)}")
    else:
        log.error("❌ Błąd tworzenia bazy")
    
    return success

# ============================================
# REBUILD INDEX
# ============================================

async def rebuild_index() -> bool:
    """Przebudowuje indeks z katalogu źródłowego"""
    log.info("🔄 Rozpoczynam przebudowę indeksu...")
    
    # Usuń stare indeksy
    index_dir = settings.FAISS_INDEX_PATH_OBJ
    if index_dir.exists():
        import shutil
        shutil.rmtree(index_dir)
        log.info(f"🗑️ Usunięto stary indeks: {index_dir}")
    
    # Załaduj od nowa
    success = await initialize_vector_store_from_directory(force_recreate=True)
    
    if success:
        log.info("✅ Indeks przebudowany pomyślnie!")
    else:
        log.error("❌ Błąd podczas przebudowy indeksu")
    
    return success

# ============================================
# DEPENDENCY INJECTION
# ============================================

_rag_service_instance = None

async def get_rag_service(rebuild: bool = False) -> RAGService:
    global _rag_service_instance
    
    if rebuild:
        await rebuild_index()
        _rag_service_instance = None
    
    if _rag_service_instance is None:
        # Sprawdź czy indeks istnieje
        index_dir = settings.FAISS_INDEX_PATH_OBJ
        if not index_dir.exists() or not (index_dir / "faiss_index.bin").exists():
            log.info("🔧 Indeks nie istnieje - tworzenie nowego...")
            await initialize_vector_store_from_directory(force_recreate=True)
        
        _rag_service_instance = RAGService()
    
    return _rag_service_instance