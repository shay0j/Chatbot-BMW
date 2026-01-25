"""
Serwis RAG (Retrieval-Augmented Generation) dla BMW Assistant.
Łączy ChromaDB z Cohere embeddings i dostarcza kontekst dla LLM.
"""
import asyncio
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import hashlib
import json
from pathlib import Path
import re
from datetime import timedelta 

import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
import numpy as np

from app.core.config import settings

# ============================================
# SIMPLE LOGGER (fallback)
# ============================================

try:
    from app.utils.logger import log, PerformanceLogger
except ImportError:
    # Simple fallback logger
    import sys
    from datetime import datetime
    from contextlib import contextmanager
    import time
    
    class SimpleLogger:
        def __init__(self, name: str = "rag_service"):
            self.name = name
        
        def _log(self, level: str, message: str):
            timestamp = datetime.now().strftime("%H:%M:%S")
            print(f"{timestamp} - {self.name} - {level} - {message}", file=sys.stdout)
        
        def debug(self, message: str):
            self._log("DEBUG", message)
        
        def info(self, message: str):
            self._log("INFO", message)
        
        def warning(self, message: str):
            self._log("WARNING", message)
        
        def error(self, message: str):
            self._log("ERROR", message)
    
    log = SimpleLogger()
    
    class PerformanceLogger:
        @staticmethod
        def measure(name: str):
            @contextmanager
            def timer():
                start_time = time.time()
                try:
                    yield
                finally:
                    elapsed = time.time() - start_time
                    log.info(f"⏱️  {name}: {elapsed:.3f}s")
            return timer()

# ============================================
# CACHE SERVICE (simplified)
# ============================================

class SimpleCacheService:
    """Prosty cache w pamięci"""
    
    def __init__(self, namespace: str = "default"):
        self.namespace = namespace
        self.cache = {}
    
    async def get(self, key: str):
        """Pobiera wartość z cache"""
        full_key = f"{self.namespace}_{key}"
        if full_key in self.cache:
            entry = self.cache[full_key]
            if datetime.now() < entry['expires']:
                return entry['value']
            else:
                del self.cache[full_key]
        return None
    
    async def set(self, key: str, value, ttl: int = 300):
        """Ustawia wartość w cache"""
        full_key = f"{self.namespace}_{key}"
        self.cache[full_key] = {
            'value': value,
            'expires': datetime.now() + timedelta(seconds=ttl)
        }
    
    async def delete(self, key: str):
        """Usuwa wartość z cache"""
        full_key = f"{self.namespace}_{key}"
        if full_key in self.cache:
            del self.cache[full_key]


# ============================================
# INTENT DETECTOR
# ============================================

class IntentDetector:
    """Detektor intencji zoptymalizowany dla BMW"""
    
    def __init__(self):
        # Przywitania - pomiń RAG
        self.skip_intents = {
            "greeting": ["cześć", "hej", "witam", "dzień dobry", "siema", "hello", "hi", "yo"],
            "farewell": ["do widzenia", "pa", "nara", "bye", "dzięki", "dziękuję"],
            "small_talk": ["jak się masz", "co słychać", "proszę", "ok", "okej", "dobrze"]
        }
        
        # Wykrywanie modeli BMW
        self.bmw_models = [
            "x1", "x2", "x3", "x4", "x5", "x6", "x7", "xm",
            "i3", "i4", "i5", "i7", "i8", "ix", "ix3", "ix5",
            "m2", "m3", "m4", "m5", "m8",
            "seria 1", "seria 2", "seria 3", "seria 4", 
            "seria 5", "seria 7", "seria 8",
            "z4", "m235", "m240", "m340", "m440", "m550"
        ]
        
        # Słowa kluczowe dla intencji
        self.intent_keywords = {
            "technical": ["moc", "silnik", "skrzynia", "prędkość", "przyspieszenie", 
                         "spalanie", "zasięg", "bateria", "akumulator", "kw", "km",
                         "nm", "rpm", "hp", "ps", "kwh", "0-100", "0-60", "spalanie",
                         "napęd", "4x4", "awd", "rwd", "fwd", "pojemność", "moment"],
            "price": ["cena", "koszt", "cen", "zł", "euro", "dolar", "opłata", 
                     "płatność", "kredyt", "leasing", "rata", "finansowanie"],
            "model": ["model", "modele", "wersja", "seria", "typ", "wariant", "edycja"],
            "test_drive": ["test drive", "jazda próbna", "próbna jazda", "przejażdżka"],
            "dealer": ["dealer", "salon", "showroom", "zk motors", "kontakt", "adres"],
            "service": ["serwis", "naprawa", "gwarancja", "przegląd", "warsztat"]
        }
    
    def should_skip_rag(self, query: str) -> bool:
        """Czy pominąć RAG dla tego zapytania?"""
        query_lower = query.lower().strip()
        
        # Puste zapytanie
        if not query_lower:
            return True
        
        # Bardzo krótkie zapytania (1-2 słowa)
        words = query_lower.split()
        
        # Dla 1-2 słów sprawdź dokładne dopasowanie
        if len(words) <= 2:
            # Dokładne dopasowanie dla przywitań
            exact_greetings = ["cześć", "hej", "witam", "siema", "hello", "hi", "yo"]
            for greeting in exact_greetings:
                if greeting == query_lower:
                    return True
            
            # Dokładne dopasowanie dla pożegnań
            exact_farewells = ["pa", "nara", "bye", "dzięki", "dziękuję"]
            for farewell in exact_farewells:
                if farewell == query_lower:
                    return True
            
            # Dzień dobry / do widzenia (frazy)
            if query_lower in ["dzień dobry", "do widzenia"]:
                return True
        
        # Mała rozmowa (frazy) - sprawdź czy jest w query
        small_talk_phrases = ["jak się masz", "co słychać", "proszę", "ok", "okej", "dobrze"]
        for phrase in small_talk_phrases:
            if phrase in query_lower:
                return True
        
        # Tylko emotikony/znaki
        if re.match(r'^[\W_]+$', query_lower) and len(query_lower) < 10:
            return True
        
        return False
    
    def detect_intent(self, query: str) -> Dict[str, Any]:
        """Wykrywa intencję i modele w zapytaniu"""
        query_lower = query.lower()
        
        # Podstawowa detekcja
        if self.should_skip_rag(query):
            return {
                "skip_rag": True,
                "primary_intent": "greeting",
                "detected_models": [],
                "is_technical": False,
                "confidence": 1.0
            }
        
        # Wykryj modele BMW
        detected_models = []
        for model in self.bmw_models:
            # Sprawdź czy model jest w zapytaniu (jako osobne słowo)
            if re.search(r'\b' + re.escape(model) + r'\b', query_lower):
                detected_models.append(model.upper())
        
        # Wykryj intencję
        primary_intent = "general"
        is_technical = False
        confidence = 0.5  # Domyślna pewność dla ogólnych zapytań
        
        # Zwiększ confidence jeśli znajdziesz konkretne słowa kluczowe
        keyword_count = 0
        for intent, keywords in self.intent_keywords.items():
            for keyword in keywords:
                if re.search(r'\b' + re.escape(keyword) + r'\b', query_lower):
                    keyword_count += 1
                    if intent == "technical":
                        is_technical = True
                    if primary_intent == "general":  # Pierwsza wykryta intencja
                        primary_intent = intent
        
        # Dostosuj confidence na podstawie liczby słów kluczowych
        if keyword_count > 0:
            confidence = min(0.8, 0.5 + (keyword_count * 0.1))
        
        # Zwiększ confidence jeśli znaleziono model BMW
        if detected_models:
            confidence = min(0.9, confidence + 0.2)
        
        return {
            "skip_rag": False,
            "primary_intent": primary_intent,
            "detected_models": detected_models,
            "is_technical": is_technical,
            "confidence": confidence
        }


# ============================================
# VECTOR STORE SERVICE
# ============================================

class VectorStoreService:
    """Serwis do zarządzania ChromaDB z Cohere embeddings"""
    
    def __init__(self):
        self.client = None
        self.collection = None
        self.collection_name = settings.CHROMA_COLLECTION_NAME
        self.embedding_function = None
        self._init_client()
    
    def _init_client(self):
        """Inicjalizuje klienta ChromaDB z Cohere embeddings"""
        try:
            # Użyj ścieżki z configu
            persist_path = settings.CHROMA_DB_PATH_OBJ
            
            log.info(f"Connecting to ChromaDB at: {persist_path}")
            
            if not persist_path.exists():
                log.error(f"ChromaDB path does not exist: {persist_path}")
                return False
            
            # Inicjalizuj klienta
            self.client = chromadb.PersistentClient(
                path=str(persist_path),
                settings=Settings(anonymized_telemetry=False)
            )
            
            # Inicjalizuj funkcję embeddingową
            self.embedding_function = embedding_functions.CohereEmbeddingFunction(
                api_key=settings.COHERE_API_KEY,
                model_name=settings.COHERE_EMBED_MODEL
            )
            
            # Połącz się z istniejącą kolekcją
            try:
                self.collection = self.client.get_collection(
                    name=self.collection_name,
                    embedding_function=self.embedding_function
                )
                doc_count = self.collection.count()
                log.info(f"Connected to collection: {self.collection_name} with {doc_count} documents")
                return True
                
            except Exception as e:
                log.error(f"Failed to get collection '{self.collection_name}': {str(e)}")
                # Sprawdź dostępne kolekcje
                collections = self.client.list_collections()
                log.info(f"Available collections: {[c.name for c in collections]}")
                return False
                
        except Exception as e:
            log.error(f"ChromaDB initialization error: {str(e)}")
            return False
    
    async def search(
        self,
        query_text: str,
        top_k: int = 5,
        filter_metadata: Optional[Dict[str, Any]] = None,
        min_score: float = 0.0
    ) -> Tuple[List[Dict[str, Any]], List[float]]:
        """
        Wyszukuje podobne dokumenty w ChromaDB.
        """
        if self.collection is None:
            log.warning("ChromaDB not initialized")
            return [], []
        
        try:
            # ChromaDB przyjmuje query_texts i sam tworzy embeddingi
            results = self.collection.query(
                query_texts=[query_text],
                n_results=top_k,
                where=filter_metadata,
                include=["documents", "metadatas", "distances"]
            )
            
            documents = []
            distances = []
            
            if results.get("documents") and results["documents"][0]:
                for i in range(len(results["documents"][0])):
                    doc_text = results["documents"][0][i]
                    metadata = results["metadatas"][0][i] if results.get("metadatas") else {}
                    distance = results["distances"][0][i] if results.get("distances") else 0.0
                    
                    # ChromaDB używa dystansów: mniejsze = lepiej
                    if distance >= min_score:
                        doc_data = {
                            "content": doc_text,
                            "metadata": metadata,
                            "id": results["ids"][0][i] if results.get("ids") else f"doc_{i}"
                        }
                        documents.append(doc_data)
                        distances.append(float(distance))
            
            log.debug(f"Found {len(documents)} documents for query: '{query_text[:50]}...'")
            return documents, distances
            
        except Exception as e:
            log.error(f"ChromaDB search error: {str(e)}")
            return [], []
    
    async def get_document_count(self) -> int:
        """Zwraca liczbę dokumentów w kolekcji"""
        if self.collection is None:
            return 0
        
        try:
            return self.collection.count()
        except Exception as e:
            log.error(f"Failed to get document count: {str(e)}")
            return 0


# ============================================
# RAG SERVICE
# ============================================

class RAGService:
    """
    Główny serwis RAG używa ChromaDB z Cohere embeddings.
    """
    
    def __init__(self):
        self.vector_store = VectorStoreService()
        self.cache = SimpleCacheService(namespace="rag")
        self.intent_detector = IntentDetector()
        
        # Konfiguracja Z CONFIGU
        self.min_confidence = getattr(settings, 'SIMILARITY_THRESHOLD', 0.5)
        self.max_distance = 1.0 - self.min_confidence
        self.top_k_default = getattr(settings, 'TOP_K_DOCUMENTS', 3)
        
        # Statystyki
        self._stats = {
            "queries_processed": 0,
            "documents_retrieved": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "intent_skipped": 0,
            "low_confidence_skipped": 0,
            "high_quality_results": 0
        }
        
        log.info(f"RAGService initialized with min_confidence={self.min_confidence}, top_k={self.top_k_default}")
    
    def _distance_to_confidence(self, distance: float) -> float:
        """Konwertuje dystans ChromaDB na confidence (0-1)"""
        # Dla cosine: distance = 1 - cosine_similarity
        # Więc confidence = 1 - distance
        confidence = 1.0 - distance
        return float(max(0.0, min(1.0, confidence)))
    
    def _generate_cache_key(self, query: str, top_k: int) -> str:
        """Generuje klucz cache"""
        query_hash = hashlib.md5(query.lower().encode()).hexdigest()[:8]
        return f"rag_{query_hash}_{top_k}"
    
    def _calculate_dynamic_threshold(self, query: str, intent_info: Dict[str, Any]) -> float:
        """
        Dynamicznie dostosowuje próg confidence na podstawie zapytania.
        Dla zapytań o modele BMW obniża próg, aby zwracały dokumenty.
        """
        query_lower = query.lower()
        
        # DOMYŚLNIE: używamy niskiego progu dla wszystkich zapytań o BMW
        base_threshold = 0.5  # ZAWSZE 0.5 zamiast 0.6
        
        # Sprawdź czy to zapytanie o model BMW
        bmw_model_patterns = [
            r'\bx[1-7]\b', r'\bxm\b', 
            r'\bi[3-8]\b', r'\bix[0-9]?\b',
            r'\bm[2-8]\b',
            r'\bseria\s+[1-8]\b'
        ]
        
        for pattern in bmw_model_patterns:
            if re.search(pattern, query_lower):
                log.debug(f"BMW model pattern '{pattern}' found, using threshold 0.4")
                return 0.4
        
        # Jeśli wykryto modele w intencji
        if intent_info.get("detected_models"):
            log.debug(f"Detected models in intent: {intent_info['detected_models']}, using threshold 0.4")
            return 0.4
        
        # Krótkie zapytania - niższy próg
        if len(query_lower.split()) <= 3:
            return max(0.4, base_threshold - 0.1)
        
        return base_threshold
    
    def _convert_to_serializable(self, obj: Any) -> Any:
        """Konwertuje obiekty NumPy na serializowalne typy Pythona"""
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, dict):
            return {k: self._convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_to_serializable(item) for item in obj]
        else:
            return obj
    
    async def retrieve_with_intent_check(
        self,
        query: str,
        top_k: int = None,
        confidence_threshold: float = None
    ) -> Dict[str, Any]:
        """
        Główna metoda RAG z detekcją intencji.
        """
        if top_k is None:
            top_k = self.top_k_default
        
        start_time = datetime.now()
        self._stats["queries_processed"] += 1
        
        try:
            # 1. Detekcja intencji
            intent_info = self.intent_detector.detect_intent(query)
            
            # DEBUG: Sprawdź co wykrył intent detector
            log.debug(f"Intent detection for '{query}': skip_rag={intent_info['skip_rag']}, intent={intent_info['primary_intent']}")
            
            # 2. Użyj dynamicznego progu jeśli nie podano
            if confidence_threshold is None:
                confidence_threshold = self._calculate_dynamic_threshold(query, intent_info)
            
            if intent_info["skip_rag"]:
                self._stats["intent_skipped"] += 1
                log.info(f"Skipping RAG for greeting query: '{query}'")
                result = {
                    "has_data": False,
                    "skip_rag": True,
                    "below_threshold": False,
                    "confidence": 0.0,
                    "intent": intent_info["primary_intent"],
                    "detected_models": intent_info["detected_models"],
                    "tech": intent_info["is_technical"],
                    "documents": [],
                    "sources": [],
                    "processing_time": (datetime.now() - start_time).total_seconds()
                }
                return self._convert_to_serializable(result)
            
            # 3. Sprawdź cache
            cache_key = self._generate_cache_key(query, top_k)
            cached = await self.cache.get(cache_key)
            
            if cached:
                self._stats["cache_hits"] += 1
                try:
                    if isinstance(cached, str):
                        result = json.loads(cached)
                    else:
                        result = cached
                    
                    # Użyj dynamicznego progu również dla cache
                    confidence = result.get("confidence", 0.0)
                    result["below_threshold"] = confidence < confidence_threshold
                    result["has_data"] = not result["below_threshold"] and len(result.get("documents", [])) > 0
                    
                    result["processing_time"] = (datetime.now() - start_time).total_seconds()
                    return result
                    
                except Exception as e:
                    log.warning(f"Cache error: {str(e)}")
                    await self.cache.delete(cache_key)
            
            self._stats["cache_misses"] += 1
            
            # 4. Wyszukaj w ChromaDB
            documents, distances = await self.vector_store.search(
                query_text=query,
                top_k=top_k,
                min_score=0.0
            )
            
            # 5. Przetwórz wyniki
            processed_docs = []
            sources = []
            confidences = []
            has_tech_data = False
            
            for i, (doc_data, distance) in enumerate(zip(documents, distances)):
                confidence = self._distance_to_confidence(distance)
                confidences.append(confidence)
                
                # Sprawdź czy to dane techniczne
                metadata = doc_data.get("metadata", {})
                content = doc_data.get("content", "").lower()
                
                is_tech = any(keyword in content for keyword in 
                            ["kw", "km", "nm", "rpm", "silnik", "moc", "prędkość", "przyspieszenie", "0-100"])
                
                if is_tech:
                    has_tech_data = True
                    self._stats["high_quality_results"] += 1
                
                # Dokument
                doc_result = {
                    "content": doc_data["content"],
                    "metadata": metadata,
                    "score": confidence,
                    "distance": distance,
                    "id": doc_data.get("id", f"doc_{i}")
                }
                processed_docs.append(doc_result)
                
                # Źródło
                source = {
                    "title": metadata.get("title", f"Document {i+1}"),
                    "content": doc_data["content"][:150] + ("..." if len(doc_data["content"]) > 150 else ""),
                    "url": metadata.get("url", ""),
                    "source": metadata.get("source_file", "unknown"),
                    "score": confidence,
                    "distance": distance
                }
                sources.append(source)
            
            self._stats["documents_retrieved"] += len(processed_docs)
            
            # 6. Oblicz średni confidence
            if confidences:
                avg_confidence = float(np.mean(confidences))
            else:
                avg_confidence = 0.0
            
            # 7. Filtruj dokumenty na podstawie progu
            filtered_docs = []
            filtered_sources = []
            
            for doc, src, conf in zip(processed_docs, sources, confidences):
                if conf >= confidence_threshold:
                    filtered_docs.append(doc)
                    filtered_sources.append(src)
            
            # 8. Przygotuj wynik
            has_data = len(filtered_docs) > 0
            below_threshold = avg_confidence < confidence_threshold
            
            result = {
                "has_data": has_data,
                "skip_rag": False,
                "below_threshold": below_threshold,
                "confidence": avg_confidence,
                "intent": intent_info["primary_intent"],
                "detected_models": intent_info["detected_models"],
                "tech": has_tech_data,
                "documents": filtered_docs,
                "sources": filtered_sources,
                "documents_retrieved": len(processed_docs),
                "documents_filtered": len(filtered_docs),
                "confidence_threshold_used": confidence_threshold,
                "processing_time": (datetime.now() - start_time).total_seconds()
            }
            
            # 9. Konwersja do typów serializowalnych
            serializable_result = self._convert_to_serializable(result)
            
            # 10. Zapisz w cache tylko jeśli mamy dane
            if processed_docs:
                try:
                    await self.cache.set(
                        cache_key,
                        json.dumps(serializable_result),
                        ttl=1800
                    )
                except Exception as e:
                    log.warning(f"Failed to cache result: {str(e)}")
            
            # 11. Loguj
            processing_time = (datetime.now() - start_time).total_seconds()
            log.info(
                f"RAG: query='{query[:50]}...', "
                f"docs_found={len(processed_docs)}, "
                f"docs_returned={len(filtered_docs)}, "
                f"conf={avg_confidence:.3f}, "
                f"threshold={confidence_threshold:.3f}, "
                f"intent={intent_info['primary_intent']}, "
                f"time={processing_time:.3f}s"
            )
            
            return serializable_result
            
        except Exception as e:
            log.error(f"RAG retrieval error: {str(e)}")
            error_result = {
                "has_data": False,
                "skip_rag": False,
                "below_threshold": True,
                "confidence": 0.0,
                "intent": "error",
                "detected_models": [],
                "tech": False,
                "documents": [],
                "sources": [],
                "error": str(e),
                "processing_time": (datetime.now() - start_time).total_seconds()
            }
            return self._convert_to_serializable(error_result)
    
    async def retrieve_simple(
        self,
        query: str,
        top_k: int = None,
        confidence_threshold: float = None
    ) -> Dict[str, Any]:
        """Alias dla kompatybilności"""
        return await self.retrieve_with_intent_check(query, top_k, confidence_threshold)
    
    async def retrieve(self, query: str) -> Dict[str, Any]:
        """
        Prosta metoda retrieve dla testów.
        Zwraca tylko has_data, confidence i documents.
        """
        result = await self.retrieve_with_intent_check(query)
        return {
            "has_data": result["has_data"],
            "confidence": result["confidence"],
            "documents": result["documents"],
            "skip_rag": result.get("skip_rag", False)
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """Sprawdza zdrowie serwisu"""
        try:
            doc_count = await self.vector_store.get_document_count()
            
            health = {
                "status": "healthy" if doc_count > 0 else "degraded",
                "documents_in_store": doc_count,
                "vector_store": "chromadb",
                "embedding_service": "cohere",
                "confidence_threshold": self.min_confidence,
                "timestamp": datetime.now().isoformat()
            }
            
            return self._convert_to_serializable(health)
            
        except Exception as e:
            error_result = {
                "status": "unhealthy",
                "error": str(e),
                "vector_store": "chromadb",
                "embedding_service": "cohere"
            }
            return self._convert_to_serializable(error_result)
    
    async def get_stats(self) -> Dict[str, Any]:
        """Zwraca statystyki"""
        doc_count = await self.vector_store.get_document_count()
        
        cache_hits = self._stats["cache_hits"]
        cache_misses = self._stats["cache_misses"]
        cache_total = cache_hits + cache_misses
        cache_rate = cache_hits / cache_total if cache_total > 0 else 0
        
        stats = {
            "documents_in_store": doc_count,
            "queries_processed": self._stats["queries_processed"],
            "documents_retrieved": self._stats["documents_retrieved"],
            "cache_hits": cache_hits,
            "cache_misses": cache_misses,
            "cache_hit_rate": round(cache_rate, 3),
            "intent_skipped": self._stats["intent_skipped"],
            "low_confidence_skipped": self._stats["low_confidence_skipped"],
            "high_quality_results": self._stats["high_quality_results"],
            "min_confidence_threshold": self.min_confidence,
            "top_k_default": self.top_k_default
        }
        
        return self._convert_to_serializable(stats)


# ============================================
# DI - DEPENDENCY INJECTION
# ============================================

_rag_service_instance = None

async def get_rag_service() -> RAGService:
    """Zwraca instancję RAGService (singleton)"""
    global _rag_service_instance
    
    if _rag_service_instance is None:
        _rag_service_instance = RAGService()
        log.info("RAGService initialized")
        
        # Sprawdź połączenie
        health = await _rag_service_instance.health_check()
        if health["status"] == "healthy":
            log.info(f"RAGService connected to {health['documents_in_store']} documents")
        else:
            log.warning(f"RAGService health check: {health['status']}")
            if "error" in health:
                log.error(f"RAGService error: {health['error']}")
    
    return _rag_service_instance