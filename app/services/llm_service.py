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
from app.core.exceptions import LLMError, RateLimitExceeded, APIError, EmbeddingError
from app.utils.logger import log, PerformanceLogger
from app.services.cache import CacheService


# ============================================
# MODELS & CONSTANTS
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
        self.model = model or settings.COHERE_CHAT_MODEL
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
# LLM SERVICE
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
                log.warning("COHERE_API_KEY not configured, using mock client")
                self.client = MockCohereClient()
                return
            
            self.client = Client(settings.COHERE_API_KEY)
            
            # Test połączenia
            try:
                self.client.chat(
                    message="Test connection",
                    model=settings.COHERE_CHAT_MODEL,
                    max_tokens=1
                )
                log.info(f"Cohere client initialized with model: {settings.COHERE_CHAT_MODEL}")
            except Exception as test_error:
                # Jeśli test failuje, sprawdź czy to stary model
                if "was removed" in str(test_error) or "not found" in str(test_error):
                    log.error(f"Model error: {test_error}")
                    log.warning(f"Falling back to mock client")
                    self.client = MockCohereClient()
                else:
                    raise test_error
            
        except Exception as e:
            log.error(f"Failed to initialize Cohere client: {str(e)}")
            
            # W development możemy użyć mock clienta
            if settings.IS_DEVELOPMENT or settings.USE_MOCK_LLM:
                log.warning("Using mock LLM client")
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
        ZAWSZE wymusza odpowiedź po polsku.
        """
        start_time = datetime.now()
        
        # Sprawdź rate limiting
        await self.rate_limiter.check_limit()
        
        # SILNA INSTRUKCJA JĘZYKA - ZAWSZE PO POLSKU
        polish_instruction = """ABSOLUTNIE WAŻNE: ODPOWIADAJ WYŁĄCZNIE PO POLSKU.

ZASADY ODPOWIEDZI PO POLSKU:
1. Używaj TYLKO języka polskiego
2. Używaj polskich znaków: ą, ć, ę, ł, ń, ó, ś, ź, ż
3. Używaj polskiej gramatyki i składni
4. NIGDY nie używaj angielskiego ani innych języków
5. NIGDY nie tłumacz na inne języki
6. Jeśli nie wiesz jak coś powiedzieć po polsku, napisz: "Przepraszam, nie potrafię odpowiedzieć"

FORMALNE ZASADY:
- Odpowiadaj w oficjalnym, profesjonalnym tonie
- Używaj form grzecznościowych: "Proszę", "Dziękuję", "Przepraszam"
- Używaj polskich nazw modeli BMW (np. "seria 3", "BMW X5", nie "series 3")
- Używaj polskich jednostek (km, litry, zł, koni mechanicznych)
- Zawsze podawaj informacje o ZK Motors jako oficjalnym dealerze

PAMIĘTAJ: Jesteś Leo, asystentem ZK Motors w Polsce. 
Twoi klienci mówią wyłącznie po polsku. Twoja odpowiedź MUSI być w 100% po polsku.

Jeśli złamiesz którąkolwiek z tych zasad, popełnisz błąd.
"""
        
        # Dodaj instrukcję języka do promptu
        enhanced_prompt = f"{polish_instruction}\n\n{prompt}"
        
        # Przygotuj request z enhanced prompt
        request = LLMRequest(
            prompt=enhanced_prompt,
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
            # Jeśli używamy mock clienta
            if isinstance(self.client, MockCohereClient):
                # Dla mock clienta również dodajemy instrukcję języka
                mock_prompt = f"ODPOWIADAJ WYŁĄCZNIE PO POLSKU: {prompt}"
                response = self.client.chat(**{
                    **request.to_cohere_params(),
                    "message": mock_prompt
                })
                
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
            else:
                # Wywołaj prawdziwe Cohere API z enhanced prompt
                with PerformanceLogger.measure("cohere_api_call"):
                    cohere_params = request.to_cohere_params()
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
            
            # SPRAWDŹ CZY ODPOWIEDŹ JEST PO POLSKU
            is_polish = self._check_if_polish(llm_response.text)
            if not is_polish:
                log.warning(f"LLM responded in non-Polish. Response: {llm_response.text[:100]}")
                # Jeśli nie po polsku, wymuś polską odpowiedź
                llm_response.text = self._force_polish_response(llm_response.text)
            
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
            
        except cohere.RateLimitError as e:
            self._stats["errors"] += 1
            log.warning(f"Cohere rate limit exceeded: {str(e)}")
            raise RateLimitExceeded(service="Cohere API", detail=str(e))
            
        except cohere.CohereError as e:
            self._stats["errors"] += 1
            log.error(f"Cohere client error: {str(e)}")
            # Fallback do mock w development
            if settings.IS_DEVELOPMENT:
                log.warning("Falling back to mock response due to API error")
                return await self._mock_fallback(prompt, request.model)
            raise LLMError(f"Cohere API error: {str(e)}")
            
        except Exception as e:
            self._stats["errors"] += 1
            log.error(f"LLM generation failed: {str(e)}", exc_info=True)
            # Fallback do mock w development
            if settings.IS_DEVELOPMENT:
                log.warning("Falling back to mock response due to error")
                return await self._mock_fallback(prompt, request.model)
            raise LLMError(f"Generation failed: {str(e)}")
    
    def _check_if_polish(self, text: str) -> bool:
        """Sprawdza czy tekst jest po polsku"""
        if not text:
            return False
        
        # Polskie znaki
        polish_chars = set('ąćęłńóśźżĄĆĘŁŃÓŚŹŻ')
        text_chars = set(text)
        
        # Sprawdź czy ma polskie znaki
        has_polish_chars = bool(polish_chars & text_chars)
        
        # Polskie słowa kluczowe
        polish_words = ['proszę', 'dziękuję', 'przepraszam', 'witam', 'cześć', 'dzień', 'dobry']
        text_lower = text.lower()
        has_polish_words = any(word in text_lower for word in polish_words)
        
        # Angielskie słowa które nie powinny występować (z wyjątkiem marki BMW/MINI)
        english_common = ['the', 'and', 'for', 'with', 'this', 'that', 'have', 'from', 'are', 'you', 'your']
        
        # Sprawdź angielskie słowa (ignoruj BMW/MINI które są markami)
        words = text_lower.split()
        english_count = 0
        for word in words:
            if word in english_common and word not in ['bmw', 'mini', 'zk']:
                english_count += 1
        
        # Jeśli ma polskie znaki i słowa, a mało angielskich słów
        return (has_polish_chars or has_polish_words) and english_count < 3
    
    def _force_polish_response(self, original_text: str) -> str:
        """Wymusza polską odpowiedź gdy LLM odpowiada po angielsku"""
        log.warning("Forcing Polish response due to non-Polish LLM output")
        
        # Jeśli odpowiedź zawiera przydatne informacje, dodaj polski komentarz
        if "BMW" in original_text or "MINI" in original_text:
            return f"Przepraszam, wystąpił problem z odpowiedzią w języku polskim. Oto przetłumaczona informacja:\n\n{original_text}\n\nProszę skontaktować się z ZK Motors po szczegóły."
        else:
            return "Przepraszam, wystąpił problem z generowaniem odpowiedzi po polsku. Proszę spróbować ponownie lub skontaktować się z ZK Motors pod numerem telefonu dostępnym na stronie."
    
    async def _mock_fallback(self, prompt: str, model: str) -> LLMResponse:
        """Fallback do mock odpowiedzi gdy API failuje"""
        mock_client = MockCohereClient()
        response = mock_client.chat(message=f"ODPOWIADAJ PO POLSKU: {prompt}", model=model)
        
        return LLMResponse(
            text=response.text,
            model=model,
            tokens_used={"input_tokens": 100, "output_tokens": 200, "total_tokens": 300},
            finish_reason="complete",
            raw_response=response
        )
    
    def _generate_cache_key(self, request: LLMRequest) -> str:
        """Generuje klucz cache dla requestu"""
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
        """
        try:
            # Jeśli używamy mock clienta
            if isinstance(self.client, MockCohereClient) or not settings.COHERE_API_KEY:
                # Return random embeddings for mock
                import numpy as np
                return list(np.random.randn(384))
            
            response = self.client.embed(
                texts=[text],
                model=settings.COHERE_EMBED_MODEL,
                input_type="search_document"
            )
            return response.embeddings[0]
        except Exception as e:
            log.error(f"Cohere embedding failed: {str(e)}")
            raise EmbeddingError(f"Cohere embedding failed: {str(e)}")
    
    async def get_available_models(self) -> List[Dict[str, Any]]:
        """Zwraca listę dostępnych modeli Cohere"""
        models = [
            {
                "id": "command-r",
                "name": "Command R",
                "provider": "Cohere",
                "max_tokens": 128000,
                "context_length": 128000,
                "supports_chat": True,
                "supports_embeddings": False,
                "status": "available"
            },
            {
                "id": "command-r-plus",
                "name": "Command R+",
                "provider": "Cohere",
                "max_tokens": 128000,
                "context_length": 128000,
                "supports_chat": True,
                "supports_embeddings": False,
                "status": "available"
            },
            {
                "id": "command-light",
                "name": "Command Light",
                "provider": "Cohere",
                "max_tokens": 4096,
                "context_length": 4096,
                "supports_chat": True,
                "supports_embeddings": False,
                "status": "available"
            },
            {
                "id": "embed-multilingual-v3.0",
                "name": "Embed Multilingual v3",
                "provider": "Cohere",
                "max_tokens": 512,
                "context_length": 512,
                "supports_chat": False,
                "supports_embeddings": True,
                "status": "available"
            }
        ]
        
        return models
    
    async def get_stats(self) -> Dict[str, Any]:
        """Zwraca statystyki użycia LLM"""
        total_cache = self._stats["cache_hits"] + self._stats["cache_misses"]
        cache_hit_rate = self._stats["cache_hits"] / total_cache if total_cache > 0 else 0
        
        return {
            "requests_sent": self._stats["requests_sent"],
            "tokens_used": self._stats["tokens_used"],
            "cache_hits": self._stats["cache_hits"],
            "cache_misses": self._stats["cache_misses"],
            "errors": self._stats["errors"],
            "cache_hit_rate": cache_hit_rate,
            "using_mock": isinstance(self.client, MockCohereClient),
            "api_configured": bool(settings.COHERE_API_KEY)
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """Sprawdza zdrowie serwisu LLM"""
        try:
            # Jeśli używamy mocka
            if isinstance(self.client, MockCohereClient):
                return {
                    "status": "healthy",
                    "provider": "Mock (Development)",
                    "model": settings.COHERE_CHAT_MODEL,
                    "test_successful": True,
                    "api_key_configured": bool(settings.COHERE_API_KEY),
                    "using_mock": True
                }
            
            # Testowe zapytanie z aktualnym modelem
            test_response = await self.generate(
                prompt="Odpowiedz tylko: OK",
                model=settings.COHERE_CHAT_MODEL,
                max_tokens=5,
                temperature=0.1,
                use_cache=False
            )
            
            # Sprawdź czy odpowiedź jest po polsku
            is_polish = self._check_if_polish(test_response.text)
            
            return {
                "status": "healthy" if is_polish else "degraded",
                "provider": "Cohere",
                "model": settings.COHERE_CHAT_MODEL,
                "test_successful": test_response.text.strip() == "OK",
                "polish_language": is_polish,
                "api_key_configured": bool(settings.COHERE_API_KEY),
                "using_mock": False
            }
            
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "provider": "Cohere" if settings.COHERE_API_KEY else "Mock",
                "api_key_configured": bool(settings.COHERE_API_KEY),
                "using_mock": isinstance(self.client, MockCohereClient)
            }
    
    async def clear_cache(self) -> bool:
        """Czyści cache LLM"""
        try:
            await self.cache.clear_namespace()
            log.info("LLM cache cleared")
            return True
        except Exception as e:
            log.error(f"Failed to clear LLM cache: {str(e)}")
            return False


# ============================================
# RATE LIMITER
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
# MOCK CLIENT (dla developmentu)
# ============================================

class MockCohereClient:
    """Mock klienta Cohere dla developmentu bez API key"""
    
    def chat(self, **kwargs):
        """Mock odpowiedzi chat - zawsze po polsku"""
        class MockResponse:
            def __init__(self):
                self.text = """Cześć! Jestem Leo, asystentem ZK Motors, oficjalnego dealera BMW i MINI.

Mogę pomóc Ci w:
- Wyborze modelu BMW lub MINI dopasowanego do Twoich potrzeb
- Specyfikacjach technicznych poszczególnych modeli
- Informacjach o test drive w salonach ZK Motors
- Aktualnych promocjach i ofertach finansowania
- Porównaniu modeli elektrycznych i spalinowych

W prawdziwej aplikacji ta odpowiedź byłaby generowana przez Cohere API z użyciem RAG i zawsze w języku polskim."""
                
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
# FACTORY FUNCTION
# ============================================

_llm_service_instance = None

async def get_llm_service() -> LLMService:
    """
    Factory function dla dependency injection.
    """
    global _llm_service_instance
    
    if _llm_service_instance is None:
        _llm_service_instance = LLMService()
        log.info("LLMService initialized")
    
    return _llm_service_instance