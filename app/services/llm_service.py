"""
Serwis LLM (Large Language Model) dla BMW Assistant.
Integracja z Cohere API i zarządzanie modelami językowymi.
"""
import asyncio
from typing import Dict, Any, Optional, List
from datetime import datetime
import json
import hashlib

import cohere
from cohere import Client

from app.core.config import settings
from app.core.exceptions import LLMError, RateLimitExceeded, APIError
from app.utils.logger import log, PerformanceLogger
from app.services.cache import CacheService

# ============================================
# 🎯 MODELS & CONSTANTS
# ============================================

class LLMResponse:
    """Reprezentuje odpowiedź z modelu językowego"""
    
    def __init__(
        self,
        text: str,
        model: str,
        tokens_used: Dict[str, int],
        finish_reason: str = "complete",
        raw_response: Optional[Any] = None
    ):
        self.text = text
        self.model = model
        self.tokens_used = tokens_used
        self.finish_reason = finish_reason
        self.raw_response = raw_response
    
    def to_dict(self) -> Dict[str, Any]:
        """Konwertuje do słownika"""
        return {
            "text": self.text,
            "model": self.model,
            "tokens_used": self.tokens_used,
            "finish_reason": self.finish_reason,
            "timestamp": datetime.now().isoformat()
        }


class LLMRequest:
    """Reprezentuje request do modelu językowego"""
    
    def __init__(
        self,
        prompt: str,
        model: str = None,
        temperature: float = None,
        max_tokens: int = None,
        stop_sequences: Optional[List[str]] = None,
        **kwargs
    ):
        self.prompt = prompt
        self.model = model or settings.COHERE_CHAT_MODEL.value
        self.temperature = temperature or settings.LLM_TEMPERATURE
        self.max_tokens = max_tokens or settings.MAX_TOKENS
        self.stop_sequences = stop_sequences or []
        self.extra_params = kwargs
    
    def to_cohere_params(self) -> Dict[str, Any]:
        """Konwertuje do parametrów Cohere API"""
        params = {
            "message": self.prompt,
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        
        if self.stop_sequences:
            params["stop_sequences"] = self.stop_sequences
        
        # Dodaj dodatkowe parametry
        params.update(self.extra_params)
        
        return params


# ============================================
# 🚀 LLM SERVICE
# ============================================

class LLMService:
    """
    Główny serwis LLM do komunikacji z Cohere API.
    Obsługuje caching, rate limiting, fallback i monitoring.
    """
    
    def __init__(self):
        self.client = None
        self.cache = CacheService(namespace="llm")
        self.rate_limiter = RateLimiter()
        self._stats = {
            "requests_sent": 0,
            "tokens_used": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "errors": 0
        }
        self._init_client()
    
    def _init_client(self):
        """Inicjalizuje klienta Cohere"""
        try:
            if not settings.COHERE_API_KEY:
                raise LLMError("COHERE_API_KEY is not configured")
            
            self.client = Client(settings.COHERE_API_KEY)
            
            # Test połączenia
            self.client.chat(
                message="Test connection",
                model="command",
                max_tokens=1
            )
            
            log.info(f"Cohere client initialized with model: {settings.COHERE_CHAT_MODEL.value}")
            
        except Exception as e:
            log.error(f"Failed to initialize Cohere client: {str(e)}")
            
            # W development możemy użyć mock clienta
            if settings.IS_DEVELOPMENT:
                log.warning("Using mock LLM client in development mode")
                self.client = MockCohereClient()
            else:
                raise LLMError(f"Failed to initialize LLM service: {str(e)}")
    
    async def generate(
        self,
        prompt: str,
        model: str = None,
        temperature: float = None,
        max_tokens: int = None,
        use_cache: bool = True,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        **kwargs
    ) -> LLMResponse:
        """
        Główna metoda generowania odpowiedzi.
        
        Args:
            prompt: Prompt do wysłania
            model: Model Cohere (command-r, command-r-plus, command)
            temperature: Kreatywność (0.0-1.0)
            max_tokens: Maksymalna liczba tokenów
            use_cache: Czy używać cache
            conversation_history: Historia konwersacji
            **kwargs: Dodatkowe parametry dla Cohere API
        
        Returns:
            LLMResponse z odpowiedzią modelu
        """
        start_time = datetime.now()
        
        # Sprawdź rate limiting
        await self.rate_limiter.check_limit()
        
        # Przygotuj request
        request = LLMRequest(
            prompt=prompt,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs
        )
        
        # Dodaj historię konwersacji jeśli dostępna
        if conversation_history:
            request.extra_params["chat_history"] = conversation_history
        
        # Sprawdź cache
        cache_key = None
        if use_cache:
            cache_key = self._generate_cache_key(request)
            cached_response = await self.cache.get(cache_key)
            if cached_response:
                self._stats["cache_hits"] += 1
                response_data = json.loads(cached_response)
                response = LLMResponse(
                    text=response_data["text"],
                    model=response_data["model"],
                    tokens_used=response_data["tokens_used"],
                    finish_reason=response_data.get("finish_reason", "complete")
                )
                
                processing_time = (datetime.now() - start_time).total_seconds()
                log.debug(f"LLM cache hit ({processing_time:.3f}s): {prompt[:50]}...")
                return response
        
        self._stats["cache_misses"] += 1
        self._stats["requests_sent"] += 1
        
        try:
            # Wywołaj Cohere API
            with PerformanceLogger.measure("cohere_api_call"):
                cohere_params = request.to_cohere_params()
                
                # Synchronous call - Cohere SDK nie ma async
                # W prawdziwej aplikacji rozważ użycie httpx dla async
                response = self.client.chat(**cohere_params)
            
            # Przetwórz odpowiedź
            llm_response = LLMResponse(
                text=response.text,
                model=request.model,
                tokens_used={
                    "input_tokens": response.meta.tokens.input_tokens,
                    "output_tokens": response.meta.tokens.output_tokens,
                    "total_tokens": response.meta.tokens.input_tokens + response.meta.tokens.output_tokens
                } if hasattr(response, 'meta') and hasattr(response.meta, 'tokens') else {},
                finish_reason=getattr(response, 'finish_reason', 'complete'),
                raw_response=response
            )
            
            self._stats["tokens_used"] += llm_response.tokens_used.get("total_tokens", 0)
            
            # Zapisz w cache jeśli warto
            if use_cache and cache_key:
                cache_data = llm_response.to_dict()
                await self.cache.set(
                    cache_key,
                    json.dumps(cache_data),
                    ttl=3600  # 1 godzina dla odpowiedzi
                )
            
            processing_time = (datetime.now() - start_time).total_seconds()
            log.info(
                f"LLM generated {len(llm_response.text)} chars, "
                f"tokens: {llm_response.tokens_used.get('total_tokens', 'N/A')}, "
                f"time: {processing_time:.3f}s"
            )
            
            return llm_response
            
        except cohere.errors.RateLimitError as e:
            self._stats["errors"] += 1
            log.warning(f"Cohere rate limit exceeded: {str(e)}")
            raise RateLimitExceeded(service="Cohere API", detail=str(e))
            
        except cohere.errors.ClientError as e:
            self._stats["errors"] += 1
            log.error(f"Cohere client error: {str(e)}")
            raise LLMError(f"Cohere API error: {str(e)}")
            
        except Exception as e:
            self._stats["errors"] += 1
            log.error(f"LLM generation failed: {str(e)}", exc_info=True)
            raise LLMError(f"Generation failed: {str(e)}")
    
    def _generate_cache_key(self, request: LLMRequest) -> str:
        """Generuje klucz cache dla requestu"""
        # Używamy hash promptu i parametrów
        key_string = f"{request.prompt}_{request.model}_{request.temperature}_{request.max_tokens}"
        
        # Dodaj parametry dodatkowe
        for key, value in sorted(request.extra_params.items()):
            key_string += f"_{key}_{value}"
        
        return f"llm_{hashlib.md5(key_string.encode()).hexdigest()}"
    
    async def generate_streaming(self, prompt: str, **kwargs):
        """
        Generuje odpowiedź w trybie streaming.
        Not implemented - Cohere SDK nie wspiera async streaming.
        """
        raise NotImplementedError("Streaming not implemented with current Cohere SDK")
    
    async def embed_text(self, text: str) -> List[float]:
        """
        Tworzy embedding dla tekstu za pomocą Cohere.
        Alternatywa dla EmbeddingService jeśli chcemy tylko Cohere.
        """
        try:
            response = self.client.embed(
                texts=[text],
                model=settings.COHERE_EMBED_MODEL.value,
                input_type="search_document"
            )
            return response.embeddings[0]
        except Exception as e:
            raise EmbeddingError(f"Cohere embedding failed: {str(e)}")
    
    async def get_available_models(self) -> List[Dict[str, Any]]:
        """Zwraca listę dostępnych modeli Cohere"""
        # Cohere nie ma endpointu do listowania modeli,
        # więc zwracamy hardcoded listę
        models = [
            {
                "id": "command-r",
                "name": "Command R",
                "provider": "Cohere",
                "max_tokens": 128000,
                "context_length": 128000,
                "supports_chat": True,
                "supports_embeddings": False
            },
            {
                "id": "command-r-plus",
                "name": "Command R+",
                "provider": "Cohere",
                "max_tokens": 128000,
                "context_length": 128000,
                "supports_chat": True,
                "supports_embeddings": False
            },
            {
                "id": "command",
                "name": "Command",
                "provider": "Cohere",
                "max_tokens": 4096,
                "context_length": 4096,
                "supports_chat": True,
                "supports_embeddings": False
            },
            {
                "id": "embed-multilingual-v3.0",
                "name": "Embed Multilingual v3",
                "provider": "Cohere",
                "max_tokens": 512,
                "context_length": 512,
                "supports_chat": False,
                "supports_embeddings": True
            }
        ]
        
        return models
    
    async def get_stats(self) -> Dict[str, Any]:
        """Zwraca statystyki użycia LLM"""
        return {
            "requests_sent": self._stats["requests_sent"],
            "tokens_used": self._stats["tokens_used"],
            "cache_hits": self._stats["cache_hits"],
            "cache_misses": self._stats["cache_misses"],
            "errors": self._stats["errors"],
            "cache_hit_rate": (
                self._stats["cache_hits"] / (self._stats["cache_hits"] + self._stats["cache_misses"])
                if (self._stats["cache_hits"] + self._stats["cache_misses"]) > 0 else 0
            )
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """Sprawdza zdrowie serwisu LLM"""
        try:
            # Testowe zapytanie
            test_response = await self.generate(
                prompt="Respond with only: OK",
                model="command",
                max_tokens=5,
                temperature=0.1,
                use_cache=False
            )
            
            return {
                "status": "healthy",
                "provider": "Cohere",
                "model": settings.COHERE_CHAT_MODEL.value,
                "test_successful": test_response.text.strip() == "OK",
                "api_key_configured": bool(settings.COHERE_API_KEY)
            }
            
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "provider": "Cohere",
                "api_key_configured": bool(settings.COHERE_API_KEY)
            }
    
    async def clear_cache(self) -> bool:
        """Czyści cache LLM"""
        try:
            await self.cache.clear()
            log.info("LLM cache cleared")
            return True
        except Exception as e:
            log.error(f"Failed to clear LLM cache: {str(e)}")
            return False


# ============================================
# 🛡️ RATE LIMITER
# ============================================

class RateLimiter:
    """Prosty rate limiter dla Cohere API"""
    
    def __init__(self, requests_per_minute: int = 60):
        self.requests_per_minute = requests_per_minute
        self.request_times = []
    
    async def check_limit(self):
        """Sprawdza czy nie przekroczono limitu"""
        from datetime import datetime, timedelta
        
        now = datetime.now()
        one_minute_ago = now - timedelta(minutes=1)
        
        # Oczyść stare requesty
        self.request_times = [t for t in self.request_times if t > one_minute_ago]
        
        # Sprawdź limit
        if len(self.request_times) >= self.requests_per_minute:
            wait_time = (self.request_times[0] + timedelta(minutes=1) - now).total_seconds()
            raise RateLimitExceeded(
                detail=f"Rate limit exceeded. Try again in {wait_time:.1f} seconds."
            )
        
        # Dodaj nowy request
        self.request_times.append(now)
    
    async def get_remaining_requests(self) -> int:
        """Zwraca pozostałą liczbę requestów w oknie czasowym"""
        from datetime import datetime, timedelta
        
        now = datetime.now()
        one_minute_ago = now - timedelta(minutes=1)
        
        recent_requests = [t for t in self.request_times if t > one_minute_ago]
        return max(0, self.requests_per_minute - len(recent_requests))


# ============================================
# 🧪 MOCK CLIENT (dla developmentu)
# ============================================

class MockCohereClient:
    """Mock klienta Cohere dla developmentu bez API key"""
    
    def chat(self, **kwargs):
        """Mock odpowiedzi chat"""
        class MockResponse:
            def __init__(self):
                self.text = """To jest mockowana odpowiedź BMW Assistant. 
                
Jako asystent BMW, mogę pomóc Ci z informacjami o:
- Modelach BMW (serie 1-8, X, i, M)
- Specyfikacjach technicznych
- Wyposażeniu i pakietach
- Cenach i promocjach
- Test drive i dealershipach

W prawdziwej aplikacji ta odpowiedź byłaby generowana przez Cohere API z użyciem RAG."""
                
                class Meta:
                    class Tokens:
                        input_tokens = 150
                        output_tokens = 200
                
                self.meta = Meta()
                self.finish_reason = "complete"
        
        return MockResponse()
    
    def embed(self, **kwargs):
        """Mock odpowiedzi embed"""
        class MockEmbedResponse:
            def __init__(self):
                self.embeddings = [[0.1] * 384]  # Mock embedding
        
        return MockEmbedResponse()


# ============================================
# 🔌 FACTORY FUNCTION
# ============================================

_llm_service_instance = None

async def get_llm_service() -> LLMService:
    """
    Factory function dla dependency injection.
    
    Usage:
        @app.get("/generate")
        async def generate(llm_service: LLMService = Depends(get_llm_service)):
            ...
    """
    global _llm_service_instance
    
    if _llm_service_instance is None:
        _llm_service_instance = LLMService()
        log.info("LLMService initialized")
    
    return _llm_service_instance