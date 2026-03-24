import os
import httpx
import json
from typing import Optional, Dict, Any, List, AsyncGenerator
from datetime import datetime
from loguru import logger

class GLMService:
    """Serwis do obsługi GLM-4.7 API od Z.ai"""
    
    def __init__(self, api_key: str = None, model: str = None, base_url: str = None):
        self.api_key = api_key or os.getenv("GLM_API_KEY")
        self.model = model or os.getenv("GLM_MODEL", "glm-4.7")
        self.base_url = base_url or os.getenv("GLM_BASE_URL", "https://open.bigmodel.cn/api/paas/v4")
        
        if not self.api_key:
            logger.warning("GLM_API_KEY nie znaleziony - użyj ustaw go w .env")
    
    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2000,
        top_p: float = 0.9,
        stop: Optional[List[str]] = None,
        stream: bool = False
    ) -> Dict[str, Any]:
        """
        Generuje odpowiedź z GLM
        
        Args:
            prompt: Prompt użytkownika
            system_prompt: System prompt (opcjonalny)
            temperature: Kreatywność (0.0-1.0)
            max_tokens: Maksymalna długość odpowiedzi
            top_p: Sampling parameter
            stop: Lista stringów przerywających generację
            stream: Czy streamować odpowiedź
            
        Returns:
            Odpowiedź z API
        """
        if not self.api_key:
            return {"error": "Brak klucza API GLM", "text": "Przepraszam, nie mogę teraz odpowiedzieć (brak API)."}
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": top_p,
            "stream": stream
        }
        
        if stop:
            payload["stop"] = stop
        
        try:
            logger.info(f"GLM request: {prompt[:50]}...")
            start_time = datetime.now()
            
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(
                    f"{self.base_url}/chat/completions",
                    headers=headers,
                    json=payload
                )
                
                elapsed = (datetime.now() - start_time).total_seconds()
                
                if response.status_code == 200:
                    data = response.json()
                    content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
                    
                    # Logowanie zużycia tokenów
                    usage = data.get("usage", {})
                    logger.info(f"GLM response in {elapsed:.2f}s, tokens: {usage.get('total_tokens', '?')}")
                    
                    return {
                        "success": True,
                        "text": content,
                        "usage": usage,
                        "model": data.get("model", self.model),
                        "elapsed": elapsed
                    }
                else:
                    error_text = await response.aread()
                    logger.error(f"GLM error {response.status_code}: {error_text}")
                    return {
                        "success": False,
                        "error": f"HTTP {response.status_code}: {error_text}",
                        "text": "Przepraszam, wystąpił błąd podczas generowania odpowiedzi."
                    }
                    
        except Exception as e:
            logger.error(f"GLM exception: {e}")
            return {
                "success": False,
                "error": str(e),
                "text": "Przepraszam, nie mogę teraz odpowiedzieć (błąd połączenia)."
            }
    
    async def generate_stream(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2000
    ) -> AsyncGenerator[str, None]:
        """Streaming odpowiedzi z GLM"""
        if not self.api_key:
            yield "data: " + json.dumps({"error": "Brak API key"}) + "\n\n"
            yield "data: [DONE]\n\n"
            return
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": True
        }
        
        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                async with client.stream("POST", f"{self.base_url}/chat/completions", headers=headers, json=payload) as response:
                    async for line in response.aiter_lines():
                        if line.startswith("data: "):
                            if line.strip() == "data: [DONE]":
                                yield "data: [DONE]\n\n"
                                break
                            
                            try:
                                data = json.loads(line[6:])
                                content = data.get("choices", [{}])[0].get("delta", {}).get("content", "")
                                if content:
                                    yield f"data: {json.dumps({'content': content})}\n\n"
                            except:
                                continue
        except Exception as e:
            logger.error(f"GLM stream error: {e}")
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
            yield "data: [DONE]\n\n"
    
    async def health_check(self) -> Dict[str, Any]:
        """Sprawdza czy GLM API działa"""
        if not self.api_key:
            return {"status": "unavailable", "error": "No API key"}
        
        try:
            # Proste testowe zapytanie
            result = await self.generate(
                prompt="Odpowiedz jednym słowem: 2+2=?",
                max_tokens=10,
                temperature=0.1
            )
            
            return {
                "status": "healthy" if result.get("success") else "degraded",
                "model": self.model,
                "api_key_present": True,
                "test_response": result.get("text", "")[:50]
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "model": self.model
            }
    
    async def get_stats(self) -> Dict[str, Any]:
        """Statystyki serwisu"""
        return {
            "model": self.model,
            "base_url": self.base_url,
            "api_key_present": bool(self.api_key),
            "max_tokens_default": 2000
        }

# Singleton dla GLM
_glm_service_instance = None

def get_glm_service():
    """Zwraca singleton GLM service"""
    global _glm_service_instance
    if _glm_service_instance is None:
        _glm_service_instance = GLMService()
    return _glm_service_instance