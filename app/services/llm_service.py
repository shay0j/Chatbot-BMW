"""
Serwis LLM (Large Language Model) dla BMW Assistant.
Integracja z Cohere API i zarządzanie modelami językowymi.
"""
import asyncio
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
import json
import hashlib

import cohere

from app.core.config import settings
from app.utils.logger import log


# ============================================
# EXCEPTIONS
# ============================================

class LLMError(Exception):
    """Błąd LLM"""
    pass


class RateLimitExceeded(Exception):
    """Przekroczono limit zapytań"""
    def __init__(self, service: str = "Cohere API", detail: str = ""):
        self.service = service
        self.detail = detail
        super().__init__(f"Rate limit exceeded for {service}. {detail}")


# ============================================
# RATE LIMITER
# ============================================

class RateLimiter:
    """Prosty rate limiter"""
    
    def __init__(self, requests_per_minute: int = 60):
        self.requests_per_minute = requests_per_minute
        self.request_timestamps: List[datetime] = []
        
    async def check_limit(self):
        """Sprawdza limit zapytań"""
        now = datetime.now()
        minute_ago = now - timedelta(minutes=1)
        
        # Usuń stare timestampy
        self.request_timestamps = [
            ts for ts in self.request_timestamps 
            if ts > minute_ago
        ]
        
        # Sprawdź czy nie przekraczamy limitu
        if len(self.request_timestamps) >= self.requests_per_minute:
            raise RateLimitExceeded()
        
        # Dodaj nowy timestamp
        self.request_timestamps.append(now)


# ============================================
# LLM SERVICE
# ============================================

class LLMService:
    """
    Główny serwis LLM do komunikacji z Cohere API.
    """
    
    def __init__(self):
        self.client = None
        self.cache = {}
        self.cache_ttl = 300  # 5 minut
        self.rate_limiter = RateLimiter(requests_per_minute=60)
        
        # Statystyki
        self.stats = {
            "requests_sent": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "errors": 0,
            "tokens_used": 0
        }
        
        self._init_client()
    
    def _init_client(self):
        """Inicjalizuje klienta Cohere"""
        try:
            if not settings.COHERE_API_KEY:
                log.warning("COHERE_API_KEY not configured, using mock client")
                self.client = None
                return
            
            self.client = cohere.Client(settings.COHERE_API_KEY)
            log.info(f"Cohere client initialized with model: {settings.COHERE_CHAT_MODEL}")
            
        except Exception as e:
            log.error(f"Failed to initialize Cohere client: {str(e)}")
            self.client = None
    
    def _generate_cache_key(self, prompt: str, model: str, temperature: float) -> str:
        """Generuje klucz cache"""
        key_string = f"{prompt}_{model}_{temperature}"
        return hashlib.md5(key_string.encode()).hexdigest()[:16]
    
    def _get_cached_response(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Pobiera zcacheowaną odpowiedź"""
        if cache_key in self.cache:
            entry = self.cache[cache_key]
            if datetime.now() - entry['timestamp'] < timedelta(seconds=self.cache_ttl):
                return entry['response']
        return None
    
    def _cache_response(self, cache_key: str, response: Dict[str, Any]):
        """Cache'uje odpowiedź"""
        self.cache[cache_key] = {
            'response': response,
            'timestamp': datetime.now()
        }
        
        # Oczyść stary cache
        self._clean_old_cache()
    
    def _clean_old_cache(self):
        """Czyści stary cache"""
        now = datetime.now()
        keys_to_remove = []
        
        for key, entry in self.cache.items():
            if now - entry['timestamp'] > timedelta(seconds=self.cache_ttl):
                keys_to_remove.append(key)
        
        for key in keys_to_remove:
            del self.cache[key]
    
    def _mock_response(self, prompt: str) -> str:
        """Mock odpowiedzi dla developmentu"""
        # Prosta logika dla common pytań
        prompt_lower = prompt.lower()
        
        if "witaj" in prompt_lower or "cześć" in prompt_lower or "dzień dobry" in prompt_lower:
            return "Witaj! Jestem BMW Assistant. Jak mogę Ci pomóc z Twoim BMW?"
        
        elif "serwis" in prompt_lower or "przegląd" in prompt_lower:
            return ("BMW zaleca regularne przeglądy serwisowe co 15-20 tys. km lub raz do roku. "
                    "Dla Twojego modelu BMW X5 z 2022 roku najbliższy przegląd powinien zostać "
                    "wykonany za około 3 tys. km lub do końca kwartału.")
        
        elif "awaria" in prompt_lower or "problem" in prompt_lower:
            return ("W przypadku awarii, zalecam kontakt z najbliższym autoryzowanym serwisem BMW. "
                    "Jeśli problem dotyczy silnika lub bezpieczeństwa, nie kontynuuj jazdy i "
                    "zadzwoń pod numer BMW Assistance: 123-456-789.")
        
        elif "paliwo" in prompt_lower or "spalanie" in prompt_lower:
            return ("Średnie spalanie dla BMW X5 wynosi około 8-12 l/100km w zależności od stylu jazdy "
                    "i warunków. Dla optymalnego zużycia paliwa zalecam płynną jazdę, unikanie "
                    "gwałtownych przyspieszeń i regularne sprawdzanie ciśnienia w oponach.")
        
        elif "cena" in prompt_lower or "koszt" in prompt_lower:
            return ("Koszt przeglądu serwisowego dla BMW X5 zaczyna się od 1500 zł. "
                    "Cena może się różnić w zależności od zakresu prac i wymienianych części. "
                    "Dokładną wycenę przygotuje dla Ciebie wybrany serwis BMW.")
        
        elif "gwarancja" in prompt_lower or "reklamacja" in prompt_lower:
            return ("BMW oferuje 2-letnią gwarancję bez limitu kilometrów. W przypadku reklamacji "
                    "skontaktuj się z autoryzowanym serwisem BMW, który dokona diagnostyki. "
                    "Pamiętaj o regularnych przeglądach - to warunek utrzymania gwarancji.")
        
        elif "opony" in prompt_lower or "zimowe" in prompt_lower:
            return ("Dla BMW X5 zalecane opony zimowe to 275/45 R20 110V. Ważne jest stosowanie "
                    "opon zimowych od 16 października do 15 kwietnia, zgodnie z polskim prawem. "
                    "BMW zaleca wymianę wszystkich 4 opon na raz dla optymalnej przyczepności.")
        
        else:
            return ("Dziękuję za pytanie. Jako BMW Assistant specjalizuję się w tematach związanych "
                    "z obsługą, serwisem i funkcjami pojazdów BMW. Czy możesz sprecyzować pytanie "
                    "dotyczące konkretnego aspektu Twojego BMW?")
    
    async def generate(
        self,
        prompt: str,
        model: str = None,
        temperature: float = None,
        max_tokens: int = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generuje odpowiedź za pomocą Cohere API.
        
        Returns:
            Słownik z odpowiedzią i metadanymi
        """
        start_time = datetime.now()
        
        # Sprawdź rate limiting
        try:
            await self.rate_limiter.check_limit()
        except RateLimitExceeded as e:
            log.warning(f"Rate limit: {e}")
            # W development możemy kontynuować
            if not settings.IS_DEVELOPMENT:
                raise
        
        # Użyj domyślnych wartości
        model = model or settings.COHERE_CHAT_MODEL
        temperature = temperature or settings.LLM_TEMPERATURE
        max_tokens = max_tokens or 500
        
        # Sprawdź cache
        cache_key = self._generate_cache_key(prompt, model, temperature)
        cached = self._get_cached_response(cache_key)
        
        if cached:
            self.stats["cache_hits"] += 1
            processing_time = (datetime.now() - start_time).total_seconds()
            log.debug(f"LLM cache hit ({processing_time:.3f}s): {prompt[:50]}...")
            return cached
        
        self.stats["cache_misses"] += 1
        self.stats["requests_sent"] += 1
        
        try:
            # Jeśli nie mamy klienta (brak API key), użyj mocka
            if self.client is None:
                log.warning("Using mock LLM response (no API key)")
                response_text = self._mock_response(prompt)
                tokens_used = {"input_tokens": 100, "output_tokens": 200, "total_tokens": 300}
            else:
                # Wywołaj prawdziwe Cohere API
                with log.measure("cohere_api_call"):
                    response = self.client.chat(
                        message=prompt,
                        model=model,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        **kwargs
                    )
                
                response_text = response.text
                
                # Pobierz tokeny jeśli dostępne
                if hasattr(response, 'meta') and hasattr(response.meta, 'tokens'):
                    tokens_used = {
                        "input_tokens": response.meta.tokens.input_tokens,
                        "output_tokens": response.meta.tokens.output_tokens,
                        "total_tokens": response.meta.tokens.input_tokens + response.meta.tokens.output_tokens
                    }
                else:
                    tokens_used = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
            
            self.stats["tokens_used"] += tokens_used.get("total_tokens", 0)
            
            # Przygotuj wynik
            result = {
                "text": response_text,
                "tokens_used": tokens_used,
                "model": model,
                "processing_time": (datetime.now() - start_time).total_seconds(),
                "success": True
            }
            
            # Cache'uj odpowiedź
            self._cache_response(cache_key, result)
            
            processing_time = (datetime.now() - start_time).total_seconds()
            log.info(
                f"LLM generated {len(response_text)} chars, "
                f"tokens: {tokens_used.get('total_tokens', 'N/A')}, "
                f"time: {processing_time:.2f}s"
            )
            
            return result
            
        except cohere.RateLimitError as e:
            self.stats["errors"] += 1
            log.error(f"Cohere rate limit exceeded: {str(e)}")
            raise RateLimitExceeded(service="Cohere API", detail=str(e))
            
        except Exception as e:
            self.stats["errors"] += 1
            log.error(f"LLM generation failed: {str(e)}")
            
            # Fallback do mock odpowiedzi
            response_text = self._mock_response(prompt)
            
            result = {
                "text": response_text,
                "tokens_used": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
                "model": model,
                "processing_time": (datetime.now() - start_time).total_seconds(),
                "success": False,
                "error": str(e)
            }
            
            return result
    
    def get_stats(self) -> Dict[str, Any]:
        """Zwraca statystyki użycia LLM"""
        return {
            **self.stats,
            "cache_size": len(self.cache),
            "cache_ttl": self.cache_ttl
        }
    
    def clear_cache(self):
        """Czyści cache"""
        self.cache.clear()
        log.info("LLM cache cleared")


# Singleton instance
llm_service = LLMService()