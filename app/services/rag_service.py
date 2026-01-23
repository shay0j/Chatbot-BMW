"""
Serwis RAG (Retrieval-Augmented Generation) dla BMW Assistant.
Łączy ChromaDB z Cohere embeddings i dostarcza kontekst dla LLM.
"""
import asyncio
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import hashlib
import json

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import numpy as np

from app.core.config import settings
from app.core.exceptions import RAGError, EmbeddingError, NotFoundError
from app.utils.logger import log, PerformanceLogger
from app.services.cache import CacheService

# ============================================
# 🎯 MODELS & CONSTANTS
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
# 🧠 EMBEDDING SERVICE
# ============================================

class EmbeddingService:
    """Serwis do tworzenia embeddingów"""
    
    def __init__(self):
        self.model = None
        self.model_name = settings.COHERE_EMBED_MODEL.value
        self.cache = CacheService(namespace="embeddings")
        self._init_model()
    
    def _init_model(self):
        """Inicjalizuje model embeddingów"""
        try:
            # Dla Cohere embeddings używamy ich API
            # Możesz też użyć sentence-transformers lokalnie
            if self.model_name.startswith("embed-"):
                # Użyjemy sentence-transformers jako fallback
                # W produkcji użyj Cohere API dla lepszej jakości
                log.info(f"Using sentence-transformers for embeddings")
                self.model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
            else:
                raise EmbeddingError(f"Unsupported embedding model: {self.model_name}")
        
        except Exception as e:
            raise EmbeddingError(f"Failed to initialize embedding model: {str(e)}")
    
    async def embed_text(self, text: str) -> np.ndarray:
        """
        Tworzy embedding dla tekstu.
        Używa cache dla poprawy wydajności.
        """
        # Generuj klucz cache
        cache_key = f"embed_{hashlib.md5(text.encode()).hexdigest()}"
        
        # Sprawdź cache
        cached = await self.cache.get(cache_key)
        if cached:
            return np.array(json.loads(cached))
        
        # Stwórz embedding
        try:
            with PerformanceLogger.measure("embedding"):
                # Użyj Cohere API jeśli masz klucz
                if settings.COHERE_API_KEY and self.model_name.startswith("embed-"):
                    embedding = await self._embed_with_cohere(text)
                else:
                    # Fallback do lokalnego modelu
                    embedding = self.model.encode(text)
            
            # Zapisz w cache
            await self.cache.set(
                cache_key,
                json.dumps(embedding.tolist()),
                ttl=86400  # 24 godziny
            )
            
            return embedding
            
        except Exception as e:
            raise EmbeddingError(f"Failed to create embedding: {str(e)}")
    
    async def _embed_with_cohere(self, text: str) -> np.ndarray:
        """Używa Cohere API do tworzenia embeddingów"""
        import cohere
        
        client = cohere.Client(settings.COHERE_API_KEY)
        
        response = client.embed(
            texts=[text],
            model=self.model_name,
            input_type="search_query"
        )
        
        return np.array(response.embeddings[0])
    
    async def embed_batch(self, texts: List[str]) -> List[np.ndarray]:
        """Tworzy embeddingi dla batcha tekstów"""
        tasks = [self.embed_text(text) for text in texts]
        return await asyncio.gather(*tasks)


# ============================================
# 🗄️ VECTOR STORE SERVICE
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
            
            self.client = chromadb.PersistentClient(
                path=str(settings.CHROMA_DB_PATH),
                settings=Settings(anonymized_telemetry=False)
            )
            
            # Sprawdź czy kolekcja istnieje
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
        
        except Exception as e:
            raise RAGError(f"Failed to initialize ChromaDB: {str(e)}")
    
    async def add_documents(self, documents: List[Document]) -> List[str]:
        """
        Dodaje dokumenty do bazy wektorowej.
        
        Returns:
            Lista ID dodanych dokumentów
        """
        if not documents:
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
            raise RAGError(f"Failed to add documents: {str(e)}")
    
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
        try:
            # Wyszukaj w ChromaDB
            results = self.collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=top_k,
                where=filter_metadata,
                include=["documents", "metadatas", "distances"]
            )
            
            # Przekształć odległości na podobieństwa (1 - distance)
            # ChromaDB zwraca odległości L2, więc konwertujemy
            distances = results["distances"][0] if results["distances"] else []
            similarities = [1.0 / (1.0 + d) for d in distances]  # Przekształcenie
            
            # Stwórz obiekty Document
            documents = []
            valid_scores = []
            
            if results["documents"] and results["documents"][0]:
                for i, (doc_text, metadata, similarity) in enumerate(
                    zip(results["documents"][0], results["metadatas"][0], similarities)
                ):
                    if similarity >= score_threshold:
                        doc = Document(
                            content=doc_text,
                            metadata=metadata,
                            id=results["ids"][0][i] if results["ids"] else None
                        )
                        documents.append(doc)
                        valid_scores.append(similarity)
            
            return documents, valid_scores
            
        except Exception as e:
            raise RAGError(f"Failed to search vector store: {str(e)}")
    
    async def get_document_count(self) -> int:
        """Zwraca liczbę dokumentów w kolekcji"""
        try:
            # Pobierz wszystkie ID (może być nieefektywne dla dużych baz)
            results = self.collection.get(limit=1)
            return len(results["ids"]) if results["ids"] else 0
        except:
            return 0
    
    async def delete_documents(self, ids: List[str]) -> bool:
        """Usuwa dokumenty po ID"""
        try:
            self.collection.delete(ids=ids)
            log.info(f"Deleted {len(ids)} documents from vector store")
            return True
        except Exception as e:
            log.error(f"Failed to delete documents: {str(e)}")
            return False


# ============================================
# 🚀 MAIN RAG SERVICE
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
        
        Args:
            query: Zapytanie użytkownika
            top_k: Liczba dokumentów do zwrócenia
            similarity_threshold: Minimalne podobieństwo (0.0-1.0)
            use_cache: Czy używać cache
            filter_by_source: Filtruj po źródle (np. "bmw.pl")
        
        Returns:
            RetrievalResult z dokumentami i podobieństwami
        """
        start_time = datetime.now()
        
        # Użyj domyślnych wartości z configa
        if top_k is None:
            top_k = settings.TOP_K_DOCUMENTS
        if similarity_threshold is None:
            similarity_threshold = settings.SIMILARITY_THRESHOLD
        
        # Generuj klucz cache
        cache_key = None
        if use_cache:
            cache_key = self._generate_cache_key(
                query, top_k, similarity_threshold, filter_by_source
            )
            cached_result = await self.cache.get(cache_key)
            if cached_result:
                self._stats["cache_hits"] += 1
                result_data = json.loads(cached_result)
                
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
                cache_data = {
                    "query": query,
                    "documents": [doc.to_dict() for doc in documents],
                    "scores": [float(s) for s in scores],
                    "query_embedding": query_embedding.tolist() if query_embedding is not None else None,
                    "timestamp": datetime.now().isoformat()
                }
                await self.cache.set(
                    cache_key,
                    json.dumps(cache_data),
                    ttl=3600  # 1 godzina
                )
            
            processing_time = (datetime.now() - start_time).total_seconds()
            log.info(
                f"RAG retrieval: {len(documents)} docs, "
                f"avg similarity: {result.average_similarity:.3f}, "
                f"time: {processing_time:.3f}s"
            )
            
            return result
            
        except Exception as e:
            log.error(f"RAG retrieval failed: {str(e)}", exc_info=True)
            raise RAGError(f"Retrieval failed: {str(e)}")
    
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
        """
        Dodaje pojedynczy dokument do RAG.
        
        Returns:
            ID dodanego dokumentu
        """
        doc = Document(content=content, metadata=metadata or {})
        
        if generate_embedding:
            doc.embedding = await self.embedding_service.embed_text(content)
        
        ids = await self.vector_store.add_documents([doc])
        return ids[0] if ids else None
    
    async def add_documents_batch(
        self,
        documents: List[Tuple[str, Dict[str, Any]]],
        batch_size: int = 100
    ) -> List[str]:
        """
        Dodaje wiele dokumentów do RAG.
        
        Args:
            documents: Lista tupli (content, metadata)
            batch_size: Rozmiar batcha dla embeddingów
        
        Returns:
            Lista ID dodanych dokumentów
        """
        all_ids = []
        
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            
            # Stwórz obiekty Document
            doc_objects = []
            for content, metadata in batch:
                doc = Document(content=content, metadata=metadata)
                doc_objects.append(doc)
            
            # Stwórz embeddingi dla batcha
            texts = [doc.content for doc in doc_objects]
            embeddings = await self.embedding_service.embed_batch(texts)
            
            # Przypisz embeddingi
            for doc, embedding in zip(doc_objects, embeddings):
                doc.embedding = embedding
            
            # Dodaj do vector store
            ids = await self.vector_store.add_documents(doc_objects)
            all_ids.extend(ids)
            
            log.info(f"Added batch {i//batch_size + 1}: {len(ids)} documents")
        
        log.info(f"Total documents added: {len(all_ids)}")
        return all_ids
    
    async def get_document(self, document_id: str) -> Optional[Document]:
        """Pobiera dokument po ID"""
        try:
            results = self.vector_store.collection.get(
                ids=[document_id],
                include=["documents", "metadatas"]
            )
            
            if results["documents"]:
                return Document(
                    content=results["documents"][0],
                    metadata=results["metadatas"][0],
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
        
        return {
            "documents_in_store": doc_count,
            "queries_processed": self._stats["queries_processed"],
            "documents_retrieved": self._stats["documents_retrieved"],
            "cache_hits": self._stats["cache_hits"],
            "cache_misses": self._stats["cache_misses"],
            "cache_hit_rate": (
                self._stats["cache_hits"] / (self._stats["cache_hits"] + self._stats["cache_misses"])
                if (self._stats["cache_hits"] + self._stats["cache_misses"]) > 0 else 0
            )
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """Sprawdza zdrowie serwisu RAG"""
        try:
            doc_count = await self.vector_store.get_document_count()
            
            # Testowe zapytanie
            test_query = "BMW"
            test_result = await self.retrieve(
                query=test_query,
                top_k=1,
                use_cache=False
            )
            
            return {
                "status": "healthy",
                "vector_store": "connected",
                "embedding_service": "operational",
                "document_count": doc_count,
                "test_query_successful": len(test_result.documents) > 0,
                "cache_enabled": True
            }
            
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "vector_store": "disconnected" if "ChromaDB" in str(e) else "unknown",
                "embedding_service": "failed"
            }
    
    async def clear_cache(self) -> bool:
        """Czyści cache RAG"""
        try:
            await self.cache.clear()
            log.info("RAG cache cleared")
            return True
        except Exception as e:
            log.error(f"Failed to clear RAG cache: {str(e)}")
            return False


# ============================================
# 🔌 FACTORY FUNCTION
# ============================================

_rag_service_instance = None

async def get_rag_service() -> RAGService:
    """
    Factory function dla dependency injection.
    Używaj z FastAPI Depends.
    
    Usage:
        @app.get("/search")
        async def search(rag_service: RAGService = Depends(get_rag_service)):
            ...
    """
    global _rag_service_instance
    
    if _rag_service_instance is None:
        _rag_service_instance = RAGService()
        log.info("RAGService initialized")
    
    return _rag_service_instance