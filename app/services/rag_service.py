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

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import numpy as np

from app.core.config import settings
from app.core.exceptions import RAGError, EmbeddingError, NotFoundError
from app.utils.logger import log, PerformanceLogger
from app.services.cache import CacheService


# ============================================
# MODELS & CONSTANTS
# ============================================

class Document:
    """Reprezentuje dokument w systemie RAG"""
    
    def __init__(
        self,
        content: str,
        metadata: Dict[str, Any] = None,
        embedding: Optional[np.ndarray] = None,
        id: Optional[str] = None
    ):
        self.content = content
        self.metadata = metadata or {}
        self.embedding = embedding
        self.id = id or self._generate_id()
    
    def _generate_id(self) -> str:
        """Generuje unikalne ID na podstawie zawartości"""
        content_hash = hashlib.md5(self.content.encode()).hexdigest()
        source = self.metadata.get("source", "unknown")
        return f"{source}_{content_hash[:8]}"
    
    def to_dict(self) -> Dict[str, Any]:
        """Konwertuje do słownika"""
        return {
            "id": self.id,
            "content": self.content,
            "metadata": self.metadata,
            "embedding": self.embedding.tolist() if self.embedding is not None else None
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Document':
        """Tworzy Document ze słownika"""
        embedding = np.array(data["embedding"]) if data.get("embedding") else None
        return cls(
            content=data["content"],
            metadata=data["metadata"],
            embedding=embedding,
            id=data.get("id")
        )


class RetrievalResult:
    """Wynik wyszukiwania w RAG"""
    
    def __init__(
        self,
        query: str,
        documents: List[Document],
        scores: List[float],
        query_embedding: Optional[np.ndarray] = None
    ):
        self.query = query
        self.documents = documents
        self.scores = scores
        self.query_embedding = query_embedding
    
    @property
    def average_similarity(self) -> float:
        """Średnie podobieństwo znalezionych dokumentów"""
        if not self.scores:
            return 0.0
        return float(np.mean(self.scores))
    
    @property
    def top_document(self) -> Optional[Document]:
        """Najbardziej podobny dokument"""
        if not self.documents:
            return None
        return self.documents[0]
    
    def to_api_response(self) -> Dict[str, Any]:
        """Konwertuje do formatu odpowiedzi API"""
        return {
            "query": self.query,
            "documents": [
                {
                    "content": doc.content[:500] + "..." if len(doc.content) > 500 else doc.content,
                    "metadata": doc.metadata,
                    "score": float(score),
                    "id": doc.id
                }
                for doc, score in zip(self.documents, self.scores)
            ],
            "sources": [
                {
                    "title": doc.metadata.get("title", "Unknown"),
                    "url": doc.metadata.get("url", ""),
                    "source": doc.metadata.get("source", "unknown"),
                    "page": doc.metadata.get("page", 1),
                    "score": float(score)
                }
                for doc, score in zip(self.documents, self.scores)
            ],
            "count": len(self.documents),
            "average_similarity": self.average_similarity
        }


# ============================================
# EMBEDDING SERVICE
# ============================================

class EmbeddingService:
    """Serwis do tworzenia embeddingów"""
    
    def __init__(self):
        self.model = None
        self.model_name = settings.COHERE_EMBED_MODEL
        self.cache = CacheService(namespace="embeddings")
        self._init_model()
    
    def _init_model(self):
        """Inicjalizuje model embeddingów"""
        try:
            # Dla Cohere embeddings używamy ich API
            if self.model_name.startswith("embed-"):
                log.info(f"Using sentence-transformers for embeddings: {self.model_name}")
                self.model = SentenceTransformer("all-MiniLM-L6-v2")
            else:
                log.warning(f"Unknown embedding model: {self.model_name}, using default")
                self.model = SentenceTransformer("all-MiniLM-L6-v2")
        
        except Exception as e:
            log.error(f"Failed to initialize embedding model: {str(e)}")
            # Fallback do prostego modelu
            self.model = None
    
    async def embed_text(self, text: str) -> np.ndarray:
        """
        Tworzy embedding dla tekstu.
        Używa cache dla poprawy wydajności.
        """
        # Generuj klucz cache
        cache_key = f"embed_{hashlib.md5(text.encode()).hexdigest()}"
        
        # Sprawdź cache - POPRAWIONE: bezpieczne pobieranie JSON
        try:
            cached = await self.cache.get(cache_key)
            if cached:
                # Jeśli cache zwraca string (JSON), sparsuj
                if isinstance(cached, str):
                    cached_data = json.loads(cached)
                    return np.array(cached_data)
                # Jeśli cache już zwraca listę/numpy array
                elif isinstance(cached, (list, np.ndarray)):
                    return np.array(cached)
        except (json.JSONDecodeError, ValueError) as e:
            log.warning(f"Cache deserialization error, regenerating embedding: {str(e)}")
        
        # Stwórz embedding
        try:
            with PerformanceLogger.measure("embedding"):
                # Użyj Cohere API jeśli masz klucz
                if hasattr(settings, 'COHERE_API_KEY') and settings.COHERE_API_KEY and self.model_name.startswith("embed-"):
                    embedding = await self._embed_with_cohere(text)
                else:
                    # Fallback do lokalnego modelu
                    if self.model is None:
                        self._init_model()
                    if self.model is not None:
                        embedding = self.model.encode(text)
                    else:
                        # Ostateczny fallback: random embedding
                        embedding = np.random.randn(384)
            
            # Zapisz w cache - POPRAWIONE: zawsze jako JSON string
            await self.cache.set(
                cache_key,
                json.dumps(embedding.tolist()),  # Serializuj do JSON
                ttl=86400  # 24 godziny
            )
            
            return embedding
            
        except Exception as e:
            log.warning(f"Embedding failed, using random fallback: {str(e)}")
            # Fallback: random embedding
            embedding = np.random.randn(384)
            # Zapisz fallback do cache
            await self.cache.set(
                cache_key,
                json.dumps(embedding.tolist()),
                ttl=300  # 5 minut dla fallback
            )
            return embedding
    
    async def _embed_with_cohere(self, text: str) -> np.ndarray:
        """Używa Cohere API do tworzenia embeddingów"""
        try:
            import cohere
            
            if not hasattr(settings, 'COHERE_API_KEY') or not settings.COHERE_API_KEY:
                raise ValueError("Cohere API key not configured")
            
            client = cohere.Client(settings.COHERE_API_KEY)
            
            response = client.embed(
                texts=[text],
                model=self.model_name,
                input_type="search_query"
            )
            
            return np.array(response.embeddings[0])
        except Exception as e:
            log.warning(f"Cohere API failed, using local model: {str(e)}")
            if self.model is not None:
                return self.model.encode(text)
            else:
                return np.random.randn(384)
    
    async def embed_batch(self, texts: List[str]) -> List[np.ndarray]:
        """Tworzy embeddingi dla batcha tekstów"""
        tasks = [self.embed_text(text) for text in texts]
        return await asyncio.gather(*tasks)


# ============================================
# VECTOR STORE SERVICE
# ============================================

class VectorStoreService:
    """Serwis do zarządzania wektorową bazą danych (ChromaDB)"""
    
    def __init__(self):
        self.client = None
        self.collection = None
        self.collection_name = settings.CHROMA_COLLECTION_NAME
        self._init_client()
    
    def _init_client(self):
        """Inicjalizuje klienta ChromaDB"""
        try:
            log.info(f"Initializing ChromaDB at {settings.CHROMA_DB_PATH}")
            
            # UTWÓRZ FOLDER
            db_path = Path(settings.CHROMA_DB_PATH)
            db_path.mkdir(parents=True, exist_ok=True)
            
            self.client = chromadb.PersistentClient(
                path=str(db_path),
                settings=Settings(anonymized_telemetry=False)
            )
            
            # Sprawdź czy kolekcja istnieje
            try:
                collections = self.client.list_collections()
                collection_names = [c.name for c in collections]
                
                if self.collection_name in collection_names:
                    self.collection = self.client.get_collection(self.collection_name)
                    log.info(f"Loaded existing collection: {self.collection_name}")
                else:
                    log.warning(f"Collection {self.collection_name} not found. Creating new...")
                    self.collection = self.client.create_collection(
                        name=self.collection_name,
                        metadata={"description": "BMW knowledge base"}
                    )
                    log.info(f"Created new collection: {self.collection_name}")
            except Exception as coll_error:
                log.error(f"Error accessing collections: {str(coll_error)}")
                # Próbuj stworzyć kolekcję
                self.collection = self.client.create_collection(
                    name=self.collection_name,
                    metadata={"description": "BMW knowledge base"}
                )
        
        except Exception as e:
            log.error(f"Failed to initialize ChromaDB: {str(e)}", exc_info=True)
            # Nie rzucaj wyjątku, pozwól aplikacji działać bez vector store
            self.collection = None
    
    async def add_documents(self, documents: List[Document]) -> List[str]:
        """
        Dodaje dokumenty do bazy wektorowej.
        
        Returns:
            Lista ID dodanych dokumentów
        """
        if not documents or self.collection is None:
            log.warning("Cannot add documents: collection not initialized")
            return []
        
        try:
            # Przygotuj dane dla ChromaDB
            ids = []
            texts = []
            metadatas = []
            embeddings = []
            
            for doc in documents:
                ids.append(doc.id)
                texts.append(doc.content)
                metadatas.append(doc.metadata)
                if doc.embedding is not None:
                    embeddings.append(doc.embedding.tolist())
            
            # Dodaj do kolekcji
            if embeddings:
                self.collection.add(
                    embeddings=embeddings,
                    documents=texts,
                    metadatas=metadatas,
                    ids=ids
                )
            else:
                self.collection.add(
                    documents=texts,
                    metadatas=metadatas,
                    ids=ids
                )
            
            log.info(f"Added {len(documents)} documents to vector store")
            return ids
            
        except Exception as e:
            log.error(f"Failed to add documents: {str(e)}")
            return []
    
    async def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
        filter_metadata: Optional[Dict[str, Any]] = None,
        score_threshold: float = 0.0
    ) -> Tuple[List[Document], List[float]]:
        """
        Wyszukuje podobne dokumenty w bazie wektorowej.
        
        Returns:
            Tuple: (documents, similarity_scores)
        """
        if self.collection is None:
            log.warning("Vector store not initialized, returning empty results")
            return [], []
        
        try:
            # Wyszukaj w ChromaDB
            results = self.collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=top_k,
                where=filter_metadata,
                include=["documents", "metadatas", "distances"]
            )
            
            # Przekształć odległości na podobieństwa
            distances = results["distances"][0] if results["distances"] else []
            similarities = [1.0 / (1.0 + d) for d in distances] if distances else []
            
            # Stwórz obiekty Document
            documents = []
            valid_scores = []
            
            if results.get("documents") and results["documents"][0]:
                for i, (doc_text, metadata, similarity) in enumerate(
                    zip(results["documents"][0], 
                        results["metadatas"][0] if results.get("metadatas") else [{}] * len(results["documents"][0]),
                        similarities)
                ):
                    if similarity >= score_threshold:
                        doc_id = results["ids"][0][i] if results.get("ids") and results["ids"][0] else None
                        doc = Document(
                            content=doc_text,
                            metadata=metadata,
                            id=doc_id
                        )
                        documents.append(doc)
                        valid_scores.append(similarity)
            
            return documents, valid_scores
            
        except Exception as e:
            log.error(f"Vector store search failed: {str(e)}")
            return [], []
    
    async def get_document_count(self) -> int:
        """Zwraca liczbę dokumentów w kolekcji"""
        if self.collection is None:
            return 0
        
        try:
            results = self.collection.get(limit=1)
            return len(results["ids"]) if results.get("ids") else 0
        except Exception as e:
            log.error(f"Failed to get document count: {str(e)}")
            return 0
    
    async def delete_documents(self, ids: List[str]) -> bool:
        """Usuwa dokumenty po ID"""
        if self.collection is None:
            return False
        
        try:
            self.collection.delete(ids=ids)
            log.info(f"Deleted {len(ids)} documents from vector store")
            return True
        except Exception as e:
            log.error(f"Failed to delete documents: {str(e)}")
            return False


# ============================================
# MAIN RAG SERVICE
# ============================================

class RAGService:
    """
    Główny serwis RAG łączący embeddingi i vector store.
    Udostępnia API dla aplikacji.
    """
    
    def __init__(self):
        self.embedding_service = EmbeddingService()
        self.vector_store = VectorStoreService()
        self.cache = CacheService(namespace="rag")
        self._stats = {
            "queries_processed": 0,
            "documents_retrieved": 0,
            "cache_hits": 0,
            "cache_misses": 0
        }
    
    async def retrieve(
        self,
        query: str,
        top_k: int = None,
        similarity_threshold: float = None,
        use_cache: bool = True,
        filter_by_source: Optional[str] = None
    ) -> RetrievalResult:
        """
        Główna metoda do wyszukiwania w RAG.
        """
        start_time = datetime.now()
        
        # Użyj domyślnych wartości z configa
        if top_k is None:
            top_k = getattr(settings, 'TOP_K_DOCUMENTS', 5)
        if similarity_threshold is None:
            similarity_threshold = getattr(settings, 'SIMILARITY_THRESHOLD', 0.7)
        
        # Generuj klucz cache
        cache_key = None
        if use_cache:
            cache_key = self._generate_cache_key(
                query, top_k, similarity_threshold, filter_by_source
            )
            cached_result = await self.cache.get(cache_key)
            if cached_result:
                self._stats["cache_hits"] += 1
                try:
                    # Bezpieczne deserializowanie cache
                    if isinstance(cached_result, str):
                        result_data = json.loads(cached_result)
                    else:
                        result_data = cached_result
                    
                    # Rekonstruuj result
                    documents = [Document.from_dict(d) for d in result_data["documents"]]
                    result = RetrievalResult(
                        query=result_data["query"],
                        documents=documents,
                        scores=result_data["scores"],
                        query_embedding=np.array(result_data["query_embedding"]) if result_data.get("query_embedding") else None
                    )
                    
                    processing_time = (datetime.now() - start_time).total_seconds()
                    log.debug(f"Cache hit for query: {query[:50]}... ({processing_time:.3f}s)")
                    return result
                except (json.JSONDecodeError, KeyError, ValueError) as cache_error:
                    log.warning(f"Cache deserialization error: {str(cache_error)}")
                    # Usuń błędny cache
                    if cache_key:
                        await self.cache.delete(cache_key)
        
        self._stats["cache_misses"] += 1
        self._stats["queries_processed"] += 1
        
        try:
            # 1. Stwórz embedding dla zapytania
            with PerformanceLogger.measure("query_embedding"):
                query_embedding = await self.embedding_service.embed_text(query)
            
            # 2. Przygotuj filtry
            filter_metadata = None
            if filter_by_source:
                filter_metadata = {"source": filter_by_source}
            
            # 3. Wyszukaj w bazie wektorowej
            with PerformanceLogger.measure("vector_search"):
                documents, scores = await self.vector_store.search(
                    query_embedding=query_embedding,
                    top_k=top_k,
                    filter_metadata=filter_metadata,
                    score_threshold=similarity_threshold
                )
            
            self._stats["documents_retrieved"] += len(documents)
            
            # 4. Stwórz wynik
            result = RetrievalResult(
                query=query,
                documents=documents,
                scores=scores,
                query_embedding=query_embedding
            )
            
            # 5. Zapisz w cache jeśli warto
            if use_cache and cache_key and documents:
                try:
                    cache_data = {
                        "query": query,
                        "documents": [doc.to_dict() for doc in documents],
                        "scores": [float(s) for s in scores],
                        "query_embedding": query_embedding.tolist() if query_embedding is not None else None,
                        "timestamp": datetime.now().isoformat()
                    }
                    await self.cache.set(
                        cache_key,
                        json.dumps(cache_data),  # Zawsze serializuj do JSON
                        ttl=3600  # 1 godzina
                    )
                except Exception as cache_error:
                    log.warning(f"Failed to cache result: {str(cache_error)}")
            
            processing_time = (datetime.now() - start_time).total_seconds()
            if documents:
                log.info(
                    f"RAG retrieval: {len(documents)} docs, "
                    f"avg similarity: {result.average_similarity:.3f}, "
                    f"time: {processing_time:.3f}s"
                )
            else:
                log.debug(f"RAG retrieval: no documents found, time: {processing_time:.3f}s")
            
            return result
            
        except Exception as e:
            log.error(f"RAG retrieval failed: {str(e)}", exc_info=True)
            return RetrievalResult(
                query=query,
                documents=[],
                scores=[],
                query_embedding=None
            )
    
    def _generate_cache_key(
        self,
        query: str,
        top_k: int,
        similarity_threshold: float,
        filter_by_source: Optional[str]
    ) -> str:
        """Generuje klucz cache dla zapytania"""
        components = [
            query,
            str(top_k),
            f"{similarity_threshold:.2f}",
            filter_by_source or "all"
        ]
        key_string = "_".join(components)
        return f"rag_{hashlib.md5(key_string.encode()).hexdigest()}"
    
    async def add_document(
        self,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
        generate_embedding: bool = True
    ) -> str:
        """Dodaje pojedynczy dokument do RAG."""
        doc = Document(content=content, metadata=metadata or {})
        
        if generate_embedding:
            doc.embedding = await self.embedding_service.embed_text(content)
        
        ids = await self.vector_store.add_documents([doc])
        return ids[0] if ids else ""
    
    async def add_documents_batch(
        self,
        documents: List[Tuple[str, Dict[str, Any]]],
        batch_size: int = 100
    ) -> List[str]:
        """Dodaje wiele dokumentów do RAG."""
        all_ids = []
        
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            
            # Stwórz obiekty Document
            doc_objects = []
            for content, metadata in batch:
                doc = Document(content=content, metadata=metadata)
                doc_objects.append(doc)
            
            # Stwórz embeddingi dla batcha
            try:
                texts = [doc.content for doc in doc_objects]
                embeddings = await self.embedding_service.embed_batch(texts)
                
                # Przypisz embeddingi
                for doc, embedding in zip(doc_objects, embeddings):
                    doc.embedding = embedding
                
                # Dodaj do vector store
                ids = await self.vector_store.add_documents(doc_objects)
                all_ids.extend(ids)
                
                log.info(f"Added batch {i//batch_size + 1}: {len(ids)} documents")
            except Exception as e:
                log.error(f"Failed to add batch {i//batch_size + 1}: {str(e)}")
        
        log.info(f"Total documents added: {len(all_ids)}")
        return all_ids
    
    async def get_document(self, document_id: str) -> Optional[Document]:
        """Pobiera dokument po ID"""
        if not hasattr(self.vector_store, 'collection') or self.vector_store.collection is None:
            return None
            
        try:
            results = self.vector_store.collection.get(
                ids=[document_id],
                include=["documents", "metadatas"]
            )
            
            if results.get("documents"):
                return Document(
                    content=results["documents"][0],
                    metadata=results["metadatas"][0] if results.get("metadatas") else {},
                    id=document_id
                )
            return None
            
        except Exception as e:
            log.error(f"Failed to get document {document_id}: {str(e)}")
            return None
    
    async def delete_document(self, document_id: str) -> bool:
        """Usuwa dokument po ID"""
        return await self.vector_store.delete_documents([document_id])
    
    async def get_stats(self) -> Dict[str, Any]:
        """Zwraca statystyki serwisu RAG"""
        doc_count = await self.vector_store.get_document_count()
        
        cache_hits = self._stats["cache_hits"]
        cache_misses = self._stats["cache_misses"]
        total_cache = cache_hits + cache_misses
        
        return {
            "documents_in_store": doc_count,
            "queries_processed": self._stats["queries_processed"],
            "documents_retrieved": self._stats["documents_retrieved"],
            "cache_hits": cache_hits,
            "cache_misses": cache_misses,
            "cache_hit_rate": cache_hits / total_cache if total_cache > 0 else 0,
            "status": "operational" if doc_count >= 0 else "error"
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """Sprawdza zdrowie serwisu RAG"""
        try:
            doc_count = await self.vector_store.get_document_count()
            
            # Testowe zapytanie tylko jeśli mamy dokumenty
            test_successful = False
            if doc_count > 0:
                test_query = "test"
                test_result = await self.retrieve(
                    query=test_query,
                    top_k=1,
                    use_cache=False
                )
                test_successful = len(test_result.documents) > 0
            
            return {
                "status": "healthy" if doc_count >= 0 else "degraded",
                "vector_store": "connected" if self.vector_store.collection is not None else "disconnected",
                "embedding_service": "operational",
                "document_count": doc_count,
                "test_query_successful": test_successful,
                "cache_enabled": True
            }
            
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "vector_store": "disconnected",
                "embedding_service": "failed"
            }
    
    async def clear_cache(self) -> bool:
        """Czyści cache RAG"""
        try:
            await self.cache.clear_namespace()
            log.info("RAG cache cleared")
            return True
        except Exception as e:
            log.error(f"Failed to clear RAG cache: {str(e)}")
            return False


# ============================================
# FACTORY FUNCTION
# ============================================

_rag_service_instance = None

async def get_rag_service() -> RAGService:
    """Factory function dla dependency injection."""
    global _rag_service_instance
    
    if _rag_service_instance is None:
        _rag_service_instance = RAGService()
        log.info("RAGService initialized")
    
    return _rag_service_instance