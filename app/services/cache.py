"""
Serwis cache dla BMW Assistant.
Wykorzystuje Redis do cache'owania odpowiedzi, embeddingów i promptów.
"""
import json
import asyncio
from typing import Any, Optional, Union, Dict  # DODAJ 'Dict' TUTAJ
from datetime import timedelta

import redis.asyncio as redis
from redis.asyncio import Redis

from app.core.config import settings
from app.core.exceptions import CacheError
from app.utils.logger import log

# ============================================
# 🎯 CACHE SERVICE
# ============================================

class CacheService:
    """
    Serwis do zarządzania cache w Redis.
    Obsługuje różne przestrzenie nazw (namespaces) dla różnych typów danych.
    """
    
    def __init__(self, namespace: str = "default", client: Optional[Redis] = None):
        """
        Args:
            namespace: Przestrzeń nazw dla kluczy cache (np. 'rag', 'llm', 'embeddings')
            client: Opcjonalny klient Redis (jeśli None, tworzy nowy)
        """
        self.namespace = namespace
        self.client = client
        self._is_connected = False
    
    async def connect(self) -> bool:
        """Nawiazywanie połączenia z Redis"""
        if self.client and self._is_connected:
            return True
        
        try:
            if not settings.REDIS_URL:
                log.warning("REDIS_URL not configured, using in-memory cache")
                self.client = None
                self._is_connected = False
                return False
            
            self.client = redis.from_url(
                settings.REDIS_URL,
                encoding="utf-8",
                decode_responses=True
            )
            
            # Test połączenia
            await self.client.ping()
            self._is_connected = True
            
            log.info(f"Redis cache connected for namespace: {self.namespace}")
            return True
            
        except Exception as e:
            log.warning(f"Redis connection failed: {str(e)}. Using in-memory cache.")
            self.client = None
            self._is_connected = False
            return False
    
    def _build_key(self, key: str) -> str:
        """Buduje klucz cache z namespace"""
        return f"bmw_assistant:{self.namespace}:{key}"
    
    async def get(self, key: str) -> Optional[Any]:
        """
        Pobiera wartość z cache.
        
        Returns:
            Zdeserializowana wartość lub None jeśli nie istnieje
        """
        try:
            if not await self.connect():
                return None
            
            cache_key = self._build_key(key)
            value = await self.client.get(cache_key)
            
            if value is None:
                return None
            
            # Deserializuj JSON
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                # Jeśli to nie JSON, zwróć jako string
                return value
                
        except Exception as e:
            log.error(f"Cache get failed for key {key}: {str(e)}")
            return None
    
    async def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None,
        serialize: bool = True
    ) -> bool:
        """
        Zapisuje wartość w cache.
        
        Args:
            key: Klucz cache
            value: Wartość do zapisania
            ttl: Time to live w sekundach (None = bez wygaśnięcia)
            serialize: Czy serializować do JSON
        
        Returns:
            True jeśli sukces, False w przeciwnym razie
        """
        try:
            if not await self.connect():
                return False
            
            cache_key = self._build_key(key)
            
            # Serializuj jeśli to obiekt
            if serialize and not isinstance(value, (str, int, float, bool)):
                value_to_store = json.dumps(value, ensure_ascii=False)
            else:
                value_to_store = value
            
            if ttl:
                await self.client.setex(cache_key, ttl, value_to_store)
            else:
                await self.client.set(cache_key, value_to_store)
            
            return True
            
        except Exception as e:
            log.error(f"Cache set failed for key {key}: {str(e)}")
            return False
    
    async def delete(self, key: str) -> bool:
        """Usuwa klucz z cache"""
        try:
            if not await self.connect():
                return False
            
            cache_key = self._build_key(key)
            result = await self.client.delete(cache_key)
            return result > 0
            
        except Exception as e:
            log.error(f"Cache delete failed for key {key}: {str(e)}")
            return False
    
    async def exists(self, key: str) -> bool:
        """Sprawdza czy klucz istnieje w cache"""
        try:
            if not await self.connect():
                return False
            
            cache_key = self._build_key(key)
            return await self.client.exists(cache_key) > 0
            
        except Exception as e:
            log.error(f"Cache exists check failed for key {key}: {str(e)}")
            return False
    
    async def increment(self, key: str, amount: int = 1) -> Optional[int]:
        """Inkrementuje wartość (dla liczników)"""
        try:
            if not await self.connect():
                return None
            
            cache_key = self._build_key(key)
            return await self.client.incrby(cache_key, amount)
            
        except Exception as e:
            log.error(f"Cache increment failed for key {key}: {str(e)}")
            return None
    
    async def clear_namespace(self) -> bool:
        """Czyści wszystkie klucze w namespace"""
        try:
            if not await self.connect():
                return False
            
            pattern = self._build_key("*")
            keys = await self.client.keys(pattern)
            
            if keys:
                await self.client.delete(*keys)
            
            log.info(f"Cleared cache namespace: {self.namespace}")
            return True
            
        except Exception as e:
            log.error(f"Failed to clear cache namespace {self.namespace}: {str(e)}")
            return False
    
    async def get_ttl(self, key: str) -> Optional[int]:
        """Pobiera pozostały czas życia klucza w sekundach"""
        try:
            if not await self.connect():
                return None
            
            cache_key = self._build_key(key)
            ttl = await self.client.ttl(cache_key)
            return ttl if ttl >= 0 else None
            
        except Exception as e:
            log.error(f"Failed to get TTL for key {key}: {str(e)}")
            return None
    
    async def get_stats(self) -> Dict[str, Any]:
        """Zwraca statystyki cache"""
        try:
            if not await self.connect():
                return {"status": "disconnected", "namespace": self.namespace}
            
            pattern = self._build_key("*")
            keys = await self.client.keys(pattern)
            
            return {
                "status": "connected",
                "namespace": self.namespace,
                "key_count": len(keys),
                "redis_url": settings.REDIS_URL
            }
            
        except Exception as e:
            return {
                "status": "error",
                "namespace": self.namespace,
                "error": str(e)
            }
    
    async def close(self):
        """Zamyka połączenie z Redis"""
        if self.client and self._is_connected:
            await self.client.close()
            self._is_connected = False
            log.debug(f"Redis connection closed for namespace: {self.namespace}")


# ============================================
# 🧠 IN-MEMORY CACHE (fallback)
# ============================================

class InMemoryCache:
    """
    Prosty cache w pamięci jako fallback gdy Redis nie działa.
    TTL obsługiwany przez ręczne czyszczenie.
    """
    
    def __init__(self):
        self._cache = {}
        self._expiry = {}
    
    async def get(self, key: str) -> Optional[Any]:
        """Pobiera z cache w pamięci"""
        import time
        
        # Sprawdź czy wygasł
        if key in self._expiry:
            expiry_time = self._expiry[key]
            if time.time() > expiry_time:
                del self._cache[key]
                del self._expiry[key]
                return None
        
        return self._cache.get(key)
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Zapisuje w cache w pamięci"""
        import time
        
        self._cache[key] = value
        
        if ttl:
            self._expiry[key] = time.time() + ttl
        elif key in self._expiry:
            del self._expiry[key]
        
        return True
    
    async def delete(self, key: str) -> bool:
        """Usuwa z cache w pamięci"""
        if key in self._cache:
            del self._cache[key]
        if key in self._expiry:
            del self._expiry[key]
        return True
    
    async def clear(self):
        """Czyści cały cache"""
        self._cache.clear()
        self._expiry.clear()
    
    async def get_stats(self) -> Dict[str, Any]:
        """Statystyki in-memory cache"""
        import time
        
        # Licz wygasłe klucze
        expired = 0
        current_time = time.time()
        
        for expiry_time in self._expiry.values():
            if current_time > expiry_time:
                expired += 1
        
        return {
            "type": "in_memory",
            "total_keys": len(self._cache),
            "keys_with_ttl": len(self._expiry),
            "expired_keys": expired,
            "memory_usage": "N/A"
        }


# ============================================
# 🏭 CACHE FACTORY
# ============================================

class CacheFactory:
    """Fabryka tworząca odpowiedni cache service"""
    
    @staticmethod
    def create_cache(namespace: str = "default") -> Union[CacheService, InMemoryCache]:
        """
        Tworzy cache service w zależności od konfiguracji.
        
        Jeśli Redis jest skonfigurowany i dostępny, używa Redis.
        W przeciwnym razie używa in-memory cache.
        """
        if settings.REDIS_URL:
            return CacheService(namespace=namespace)
        else:
            log.info(f"Using in-memory cache for namespace: {namespace}")
            return InMemoryCache()


# ============================================
# 🔧 CACHE DECORATORS
# ============================================

def cache_result(ttl: int = 3600, namespace: str = "default"):
    """
    Dekorator do cache'owania wyników funkcji.
    
    Args:
        ttl: Czas życia cache w sekundach
        namespace: Przestrzeń nazw cache
    
    Usage:
        @cache_result(ttl=300, namespace="rag")
        async def expensive_operation(query: str):
            ...
    """
    def decorator(func):
        async def wrapper(*args, **kwargs):
            # Generuj klucz cache na podstawie argumentów
            import inspect
            import hashlib
            
            # Pobierz nazwę funkcji
            func_name = func.__name__
            
            # Konwertuj argumenty na string do hashowania
            args_str = str(args) + str(sorted(kwargs.items()))
            args_hash = hashlib.md5(args_str.encode()).hexdigest()[:8]
            
            cache_key = f"{func_name}_{args_hash}"
            
            # Utwórz cache service
            cache = CacheFactory.create_cache(namespace)
            
            # Sprawdź cache
            cached = await cache.get(cache_key)
            if cached is not None:
                log.debug(f"Cache hit for {func_name}")
                return cached
            
            # Wykonaj funkcję
            result = await func(*args, **kwargs)
            
            # Zapisz w cache
            await cache.set(cache_key, result, ttl=ttl)
            log.debug(f"Cache miss for {func_name}, cached result")
            
            return result
        
        return wrapper
    
    return decorator


# ============================================
# 🔌 GLOBAL CACHE INSTANCES
# ============================================

# Globalne instancje cache dla różnych celów
rag_cache = CacheFactory.create_cache("rag")
llm_cache = CacheFactory.create_cache("llm")
embedding_cache = CacheFactory.create_cache("embeddings")
session_cache = CacheFactory.create_cache("sessions")


# ============================================
# 🚀 INITIALIZATION
# ============================================

async def init_cache() -> bool:
    """Inicjalizuje cache (testuje połączenia)"""
    try:
        # Testuj Redis jeśli skonfigurowany
        if settings.REDIS_URL:
            cache = CacheService()
            connected = await cache.connect()
            
            if connected:
                log.info("✅ Redis cache initialized successfully")
                return True
            else:
                log.warning("⚠️ Redis not available, using in-memory cache")
                return False
        else:
            log.info("ℹ️  Redis not configured, using in-memory cache")
            return False
            
    except Exception as e:
        log.error(f"❌ Cache initialization failed: {str(e)}")
        return False


async def close_cache():
    """Zamyka połączenia cache"""
    # Tutaj można dodać zamknięcie wszystkich cache services
    pass