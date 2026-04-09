"""
Serwis RAG (Retrieval-Augmented Generation) dla BMW Assistant.
Łączy ChromaDB z Cohere embeddings - WERSJA BEZ ONNXRUNTIME.
"""
import asyncio
import os
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
import hashlib
import json
import re
from pathlib import Path

# ============================================
# WYŁĄCZ TOKENIZERS PARALLELISM I UKRYJ WARNINGI
# ============================================
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# ============================================
# IMPORTY - POMIJAMY DOMYŚLNY EMBEDDER
# ============================================
import numpy as np
from dotenv import load_dotenv

# Specjalny import chromadb - unikamy domyślnego embeddera
try:
    import chromadb
    from chromadb.config import Settings
    # Nie importujemy embedding_functions na górze - zaimportujemy później
except ImportError as e:
    print(f"⚠️ ChromaDB import error: {e}")
    print("Uruchom: pip install chromadb")
    chromadb = None

load_dotenv()

# ============================================
# CONFIG
# ============================================

class Settings:
    CHROMA_COLLECTION_NAME = os.getenv("CHROMA_COLLECTION_NAME", "bmw_models")
    CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", "RAG/chroma_db_working")
    COHERE_API_KEY = os.getenv("COHERE_API_KEY")
    COHERE_EMBED_MODEL = os.getenv("COHERE_EMBED_MODEL", "embed-multilingual-v3.0")
    SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.5"))
    TOP_K_DOCUMENTS = int(os.getenv("TOP_K_DOCUMENTS", "3"))
    
    @property
    def CHROMA_DB_PATH_OBJ(self) -> Path:
        return Path(self.CHROMA_DB_PATH).absolute()

settings = Settings()

# ============================================
# LOGGER
# ============================================

class SimpleLogger:
    def __init__(self, name: str = "rag_service"):
        self.name = name
    
    def _log(self, level: str, message: str):
        timestamp = datetime.now().strftime("%H:%M:%S")
        print(f"{timestamp} - {self.name} - {level} - {message}")
    
    def info(self, message: str): self._log("INFO", message)
    def warning(self, message: str): self._log("WARNING", message)
    def error(self, message: str): self._log("ERROR", message)
    def debug(self, message: str): self._log("DEBUG", message)

log = SimpleLogger()

# ============================================
# CACHE
# ============================================

class SimpleCacheService:
    def __init__(self, namespace: str = "default"):
        self.namespace = namespace
        self.cache = {}
    
    async def get(self, key: str):
        full_key = f"{self.namespace}_{key}"
        if full_key in self.cache:
            entry = self.cache[full_key]
            if datetime.now() < entry['expires']:
                return entry['value']
            del self.cache[full_key]
        return None
    
    async def set(self, key: str, value, ttl: int = 300):
        full_key = f"{self.namespace}_{key}"
        self.cache[full_key] = {'value': value, 'expires': datetime.now() + timedelta(seconds=ttl)}
    
    async def delete(self, key: str):
        full_key = f"{self.namespace}_{key}"
        self.cache.pop(full_key, None)

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
        
        self.intent_keywords = {
            "technical": ["moc", "silnik", "prędkość", "przyspieszenie", "spalanie", "kw", "km", "nm", "0-100"],
            "price": ["cena", "koszt", "leasing", "kredyt", "rata"],
            "test_drive": ["test drive", "jazda próbna"],
            "dealer": ["dealer", "salon", "zk motors", "adres"]
        }
    
    def should_skip_rag(self, query: str) -> bool:
        query_lower = query.lower().strip()
        greetings = ["cześć", "hej", "witam", "dzień dobry", "siema", "hello"]
        return query_lower in greetings or query_lower in ["pa", "bye", "dzięki"]
    
    def detect_intent(self, query: str) -> Dict[str, Any]:
        query_lower = query.lower()
        if self.should_skip_rag(query):
            return {"skip_rag": True, "primary_intent": "greeting", "detected_models": [], "is_technical": False, "confidence": 1.0}
        
        detected_models = []
        for model in self.bmw_models:
            if re.search(r'\b' + re.escape(model) + r'\b', query_lower):
                detected_models.append(model.upper())
        
        primary_intent = "general"
        is_technical = False
        confidence = 0.5
        keyword_count = 0
        
        for intent, keywords in self.intent_keywords.items():
            for keyword in keywords:
                if re.search(r'\b' + re.escape(keyword) + r'\b', query_lower):
                    keyword_count += 1
                    if intent == "technical":
                        is_technical = True
                    if primary_intent == "general":
                        primary_intent = intent
        
        if keyword_count > 0:
            confidence = min(0.8, 0.5 + (keyword_count * 0.1))
        if detected_models:
            confidence = min(0.9, confidence + 0.2)
        
        return {"skip_rag": False, "primary_intent": primary_intent, "detected_models": detected_models, "is_technical": is_technical, "confidence": confidence}

# ============================================
# VECTOR STORE - WERSJA BEZ ONNXRUNTIME
# ============================================

class VectorStoreService:
    def __init__(self):
        self.client = None
        self.collection = None
        self.collection_name = settings.CHROMA_COLLECTION_NAME
        self._init_client()
    
    def _init_client(self):
        if chromadb is None:
            log.error("ChromaDB not installed")
            return False
        
        try:
            persist_path = settings.CHROMA_DB_PATH_OBJ
            persist_path.mkdir(parents=True, exist_ok=True)
            log.info(f"🔌 Łączenie z ChromaDB: {persist_path}")
            
            self.client = chromadb.PersistentClient(
                path=str(persist_path),
                settings=Settings(anonymized_telemetry=False)
            )
            
            # Import embedding functions DOPIERO TUTAJ (po zainicjalizowaniu klienta)
            from chromadb.utils import embedding_functions
            
            embedding_fn = embedding_functions.CohereEmbeddingFunction(
                api_key=settings.COHERE_API_KEY,
                model_name=settings.COHERE_EMBED_MODEL
            )
            
            try:
                self.collection = self.client.get_collection(
                    name=self.collection_name,
                    embedding_function=embedding_fn
                )
                log.info(f"✅ Połączono z kolekcją: {self.collection_name} ({self.collection.count()} dokumentów)")
                return True
            except:
                log.warning(f"⚠️ Kolekcja '{self.collection_name}' nie istnieje")
                return False
                
        except Exception as e:
            log.error(f"❌ Błąd ChromaDB: {e}")
            return False
    
    async def search(self, query_text: str, top_k: int = 5, **kwargs) -> Tuple[List[Dict], List[float]]:
        if self.collection is None:
            return [], []
        try:
            results = self.collection.query(
                query_texts=[query_text],
                n_results=top_k,
                include=["documents", "metadatas", "distances"]
            )
            documents = []
            distances = []
            if results.get("documents") and results["documents"][0]:
                for i in range(len(results["documents"][0])):
                    documents.append({
                        "content": results["documents"][0][i],
                        "metadata": results["metadatas"][0][i] if results.get("metadatas") else {},
                        "id": results["ids"][0][i] if results.get("ids") else f"doc_{i}"
                    })
                    distances.append(results["distances"][0][i] if results.get("distances") else 0.0)
            return documents, distances
        except Exception as e:
            log.error(f"Search error: {e}")
            return [], []
    
    async def get_document_count(self) -> int:
        return self.collection.count() if self.collection else 0

# ============================================
# INICJALIZACJA BAZY Z CSV
# ============================================

async def initialize_vector_store_from_csv(
    csv_path: str,
    collection_name: str = None,
    force_recreate: bool = False
) -> bool:
    """Tworzy bazę wektorową z pliku CSV"""
    if chromadb is None:
        log.error("ChromaDB not installed")
        return False
    
    if not settings.COHERE_API_KEY:
        log.error("COHERE_API_KEY not set")
        return False
    
    if collection_name is None:
        collection_name = settings.CHROMA_COLLECTION_NAME
    
    csv_path_obj = Path(csv_path)
    if not csv_path_obj.exists():
        log.error(f"Plik CSV nie istnieje: {csv_path_obj}")
        return False
    
    log.info(f"📁 Plik źródłowy: {csv_path_obj}")
    
    try:
        import pandas as pd
        df = pd.read_csv(csv_path_obj)
        log.info(f"✅ Wczytano {len(df)} wierszy")
    except ImportError:
        log.error("Pandas not installed: pip install pandas")
        return False
    
    # Utwórz dokumenty
    documents = []
    for idx, row in df.iterrows():
        content_parts = []
        for col in df.columns:
            value = row[col]
            if pd.notna(value) and str(value).strip():
                content_parts.append(f"{col}: {value}")
        documents.append("\n".join(content_parts))
    
    log.info(f"📄 Utworzono {len(documents)} dokumentów")
    
    # Inicjalizuj ChromaDB
    db_path = settings.CHROMA_DB_PATH_OBJ
    db_path.mkdir(parents=True, exist_ok=True)
    
    client = chromadb.PersistentClient(
        path=str(db_path),
        settings=Settings(anonymized_telemetry=False)
    )
    
    # Import embedding functions TUTAJ (unikamy onnxruntime)
    from chromadb.utils import embedding_functions
    
    embedding_fn = embedding_functions.CohereEmbeddingFunction(
        api_key=settings.COHERE_API_KEY,
        model_name=settings.COHERE_EMBED_MODEL
    )
    
    if force_recreate:
        try:
            client.delete_collection(collection_name)
            log.info(f"🗑️ Usunięto starą kolekcję")
        except:
            pass
    
    try:
        collection = client.create_collection(
            name=collection_name,
            embedding_function=embedding_fn
        )
        log.info(f"✅ Utworzono kolekcję: {collection_name}")
    except Exception as e:
        log.error(f"Błąd tworzenia kolekcji: {e}")
        return False
    
    # Dodaj dokumenty
    batch_size = 50
    for i in range(0, len(documents), batch_size):
        end = min(i + batch_size, len(documents))
        ids = [f"doc_{j:06d}" for j in range(i, end)]
        texts = documents[i:end]
        collection.add(ids=ids, documents=texts)
        log.info(f"  ➕ Dodano {i+1}-{end} z {len(documents)}")
    
    log.info(f"✅ Gotowe! Kolekcja zawiera {collection.count()} dokumentów")
    return True

# ============================================
# RAG SERVICE
# ============================================

class RAGService:
    def __init__(self):
        self.vector_store = VectorStoreService()
        self.cache = SimpleCacheService(namespace="rag")
        self.intent_detector = IntentDetector()
        self.min_confidence = settings.SIMILARITY_THRESHOLD
        self.top_k_default = settings.TOP_K_DOCUMENTS
        
        self._stats = {"queries_processed": 0, "documents_retrieved": 0}
        log.info(f"✅ RAGService gotowy")
    
    def _distance_to_confidence(self, distance: float) -> float:
        return max(0.0, min(1.0, 1.0 - distance))
    
    async def retrieve_with_intent_check(self, query: str, top_k: int = None, **kwargs) -> Dict[str, Any]:
        if top_k is None:
            top_k = self.top_k_default
        
        self._stats["queries_processed"] += 1
        intent_info = self.intent_detector.detect_intent(query)
        
        if intent_info["skip_rag"]:
            return {"has_data": False, "skip_rag": True, "documents": [], "sources": []}
        
        documents, distances = await self.vector_store.search(query, top_k)
        self._stats["documents_retrieved"] += len(documents)
        
        processed = []
        for doc, dist in zip(documents, distances):
            processed.append({
                "content": doc["content"],
                "metadata": doc.get("metadata", {}),
                "score": self._distance_to_confidence(dist)
            })
        
        return {
            "has_data": len(processed) > 0,
            "skip_rag": False,
            "confidence": processed[0]["score"] if processed else 0,
            "intent": intent_info["primary_intent"],
            "detected_models": intent_info["detected_models"],
            "documents": processed,
            "sources": processed
        }
    
    async def health_check(self) -> Dict[str, Any]:
        return {
            "status": "healthy" if await self.vector_store.get_document_count() > 0 else "degraded",
            "documents_in_store": await self.vector_store.get_document_count()
        }
    
    async def get_stats(self) -> Dict[str, Any]:
        return {
            "documents_in_store": await self.vector_store.get_document_count(),
            "queries_processed": self._stats["queries_processed"],
            "documents_retrieved": self._stats["documents_retrieved"]
        }

# ============================================
# DEPENDENCY INJECTION
# ============================================

_rag_service_instance = None

async def get_rag_service() -> RAGService:
    global _rag_service_instance
    if _rag_service_instance is None:
        _rag_service_instance = RAGService()
    return _rag_service_instance