import os
import asyncio
import time
import json
import base64
import re
import sys
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import httpx
from loguru import logger
from dotenv import load_dotenv
from pydantic import BaseModel
import secrets
import traceback

# ============================================
# IMPORT NOWEGO RAG SERVICE
# ============================================

# Dodaj ścieżkę do app w sys.path jeśli potrzebne
current_dir = Path(__file__).parent.absolute()
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

try:
    from app.services.rag_service import get_rag_service as get_rag_service_new
    from app.services.rag_service import RAGService
    RAG_AVAILABLE = True
    print("✅ Nowy RAG service załadowany")
except Exception as e:
    print(f"⚠️ Ostrzeżenie: Could not import new RAG module: {e}")
    RAG_AVAILABLE = False
    
    # Dummy RAG dla fallback
    class RAGService:
        async def retrieve_with_intent_check(self, query, top_k=3, confidence_threshold=0.5):
            return {
                "has_data": False,
                "skip_rag": False,
                "below_threshold": True,
                "confidence": 0.0,
                "intent": "general",
                "detected_models": [],
                "tech": False,
                "documents": [],
                "sources": []
            }
        async def health_check(self):
            return {"status": "unavailable", "is_dummy": True}
        async def get_stats(self):
            return {"total_chunks": 0, "is_dummy": True}
    
    async def get_rag_service_new():
        return RAGService()

# ============================================
# RAG SERVICE SINGLETON (dla kompatybilności)
# ============================================

_rag_service_instance = None

async def get_rag_service():
    """Zwraca singleton RAG service (nowa wersja)"""
    global _rag_service_instance
    if _rag_service_instance is None:
        print("Tworzę singleton RAG service (nowy)...")
        _rag_service_instance = await get_rag_service_new()
    return _rag_service_instance

# ============================================
# KONFIGURACJA I ZALADOWANIE ZMIENNYCH ŚRODOWISKOWYCH
# ============================================
load_dotenv()

print("\n" + "="*60)
print("🔍 DEBUG - Environment variables for LiveChat:")
print(f"LIVECHAT_CLIENT_ID: {os.getenv('LIVECHAT_CLIENT_ID')}")
print(f"LIVECHAT_CLIENT_SECRET: {os.getenv('LIVECHAT_CLIENT_SECRET')[:5] if os.getenv('LIVECHAT_CLIENT_SECRET') else 'None'}... (częściowo ukryty)")
print(f"LIVECHAT_REDIRECT_URI: {os.getenv('LIVECHAT_REDIRECT_URI')}")
print(f"LIVECHAT_REGION: {os.getenv('LIVECHAT_REGION')}")
print(f"LIVECHAT_TARGET_GROUP_ID: {os.getenv('LIVECHAT_TARGET_GROUP_ID')}")
print(f"LIVECHAT_ORGANIZATION_ID: {os.getenv('LIVECHAT_ORGANIZATION_ID')}")
print(f"LIVECHAT_ACCOUNT_ID: {os.getenv('LIVECHAT_ACCOUNT_ID')}")
print(f"LIVECHAT_EMAIL: {os.getenv('LIVECHAT_EMAIL')}")
print("="*60 + "\n")

NGROK_URL = "https://crinklier-ruddily-leonore.ngrok-free.dev"

BOT_CONFIG_FILE = "bot_config.json"

def load_bot_config():
    if os.path.exists(BOT_CONFIG_FILE):
        try:
            with open(BOT_CONFIG_FILE, 'r') as f:
                return json.load(f)
        except:
            return {}
    return {}

def save_bot_config(config):
    with open(BOT_CONFIG_FILE, 'w') as f:
        json.dump(config, f, indent=2)

class TokenResponse(BaseModel):
    access_token: str
    refresh_token: Optional[str] = None
    expires_in: int
    token_type: str = "bearer"

class BotCreateResponse(BaseModel):
    id: str
    secret: str

class IncomingEvent(BaseModel):
    chat_id: str
    event: Dict[str, Any]
    author_id: Optional[str] = None

@dataclass
class MessageHandler:
    chat_id: str
    message: str
    author_id: str

class LiveChatAuth:
    
    def __init__(self):
        self.client_id = os.getenv("LIVECHAT_CLIENT_ID")
        self.client_secret = os.getenv("LIVECHAT_CLIENT_SECRET")
        self.redirect_uri = os.getenv("LIVECHAT_REDIRECT_URI", "http://localhost:8000/callback")
        self.region = os.getenv("LIVECHAT_REGION", "us")
        self.organization_id = os.getenv("LIVECHAT_ORGANIZATION_ID")
        self.account_id = os.getenv("LIVECHAT_ACCOUNT_ID")
        self.email = os.getenv("LIVECHAT_EMAIL")
        
        if not self.client_id or not self.client_secret:
            logger.warning("Brak LIVECHAT_CLIENT_ID lub LIVECHAT_CLIENT_SECRET w .env")
        
        self.accounts_url = "https://accounts.livechatinc.com"
        self.api_url = "https://api.livechatinc.com"
        
        self.oauth_token = "eu-west3:oLOLIreQRpV7zhu4MuWS0h38GYo"
        
        self.http_client = httpx.AsyncClient(timeout=30.0)
    
    async def get_valid_token(self) -> str:
        return self.oauth_token
    
    def is_authenticated(self) -> bool:
        return bool(self.oauth_token)
    
    async def close(self):
        await self.http_client.aclose()

auth_client = LiveChatAuth()

class LiveChatAPIClient:
    
    def __init__(self):
        self.base_url = "https://api.livechatinc.com"
        self.http_client = httpx.AsyncClient(timeout=30.0)
    
    async def _request(self, method: str, path: str, **kwargs) -> Any:
        token = await auth_client.get_valid_token()
        
        headers = kwargs.pop("headers", {})
        headers["Authorization"] = f"Bearer {token}"
        headers["Content-Type"] = "application/json"
        
        url = f"{self.base_url}{path}"
        
        try:
            print(f"\n🔍 API Request: {method} {url}")
            
            response = await self.http_client.request(
                method, url, headers=headers, **kwargs
            )
            
            print(f"📊 Response status: {response.status_code}")
            
            if response.status_code >= 400:
                error_msg = f"Błąd API {response.status_code}"
                try:
                    error_data = response.json()
                    error_msg += f": {json.dumps(error_data)}"
                    print(f"❌ {error_msg}")
                except:
                    error_msg += f": {response.text}"
                    print(f"❌ {error_msg}")
                response.raise_for_status()
            
            response_body = response.text
            if response_body.strip().startswith('{') or response_body.strip().startswith('['):
                return response.json()
            else:
                return response_body
                    
        except Exception as e:
            print(f"💥 Błąd: {e}")
            raise
    
    async def create_bot(self, name: str) -> Tuple[str, str]:
        """Tworzy nowego bota w LiveChat"""
        token = await auth_client.get_valid_token()
        
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "name": name,
            "default_group_priority": "normal",
            "max_chats_count": 5,
            "timezone": "Europe/Warsaw"
        }
        
        print(f"\n🤖 Tworzenie nowego bota o nazwie {name}")
        print(f"📦 Payload: {json.dumps(payload, indent=2)}")
        
        async with httpx.AsyncClient() as client:
            response = await client.post(
                "https://api.livechatinc.com/v3.6/configuration/action/create_bot",
                headers=headers,
                json=payload
            )
            
            print(f"📊 Response status: {response.status_code}")
            
            if response.status_code == 200 or response.status_code == 201:
                data = response.json()
                bot_id = data.get("id")
                bot_secret = data.get("secret")
                if bot_id and bot_secret:
                    print(f"✅ Bot utworzony! ID: {bot_id}")
                    return bot_id, bot_secret
                else:
                    raise Exception("Brak ID lub sekretu w odpowiedzi")
            else:
                error_data = response.json()
                raise Exception(f"Błąd tworzenia bota: {error_data}")
    
    async def get_bot_token(self, bot_id: str, bot_secret: str) -> Optional[str]:
        """Generuje token dla bota"""
        try:
            token = await auth_client.get_valid_token()
            
            headers = {
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "bot_id": bot_id,
                "client_id": auth_client.client_id,
                "organization_id": auth_client.organization_id,
                "bot_secret": bot_secret
            }
            
            print(f"\n🔑 Generowanie tokena dla bota {bot_id}")
            
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    "https://api.livechatinc.com/v3.6/configuration/action/issue_bot_token",
                    headers=headers,
                    json=payload
                )
                
                print(f"📊 Response status: {response.status_code}")
                
                if response.status_code == 200:
                    data = response.json()
                    bot_token = data.get("token")
                    print(f"✅ Token bota wygenerowany: {bot_token[:20]}...")
                    return bot_token
                else:
                    print(f"❌ Błąd generowania tokena: {response.text}")
                    return None
                    
        except Exception as e:
            print(f"❌ Błąd: {e}")
            return None
    
    async def set_routing_status(self, bot_id: str, status: str = "accepting_chats") -> Dict[str, Any]:
        agent_token = await auth_client.get_valid_token()
        
        headers = {
            "Authorization": f"Bearer {agent_token}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "id": bot_id,
            "status": status
        }
        
        url = f"{self.base_url}/v3.6/agent/action/set_routing_status"
        
        print(f"\n🤖 Ustawianie statusu routingu dla bota {bot_id} na {status}")
        print(f"📦 Payload: {json.dumps(payload, indent=2)}")
        
        async with httpx.AsyncClient() as client:
            response = await client.post(url, headers=headers, json=payload)
            print(f"📊 Response status: {response.status_code}")
            
            if response.status_code == 200 or response.status_code == 202:
                print(f"✅ Bot aktywowany (status: {status})")
                return {"success": True}
            else:
                try:
                    error_data = response.json()
                    print(f"❌ Błąd aktywacji: {json.dumps(error_data, indent=2)}")
                    return {"success": False, "error": error_data}
                except:
                    print(f"❌ Błąd aktywacji: {response.text}")
                    return {"success": False, "error": response.text}
    
    async def join_chat(self, chat_id: str, bot_id: str) -> Dict[str, Any]:
        agent_token = await auth_client.get_valid_token()
        
        headers = {
            "Authorization": f"Bearer {agent_token}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "chat_id": chat_id,
            "user_id": bot_id,
            "user_type": "agent",
            "visibility": "all",
            "ignore_requester_presence": True
        }
        
        url = f"{self.base_url}/v3.6/agent/action/add_user_to_chat"
        
        print(f"\n👋 Dołączanie do czatu {chat_id}")
        print(f"📦 Payload: {json.dumps(payload, indent=2)}")
        
        async with httpx.AsyncClient() as client:
            response = await client.post(url, headers=headers, json=payload)
            print(f"📊 Response status: {response.status_code}")
            
            try:
                response_data = response.json()
                print(f"📄 Response body: {json.dumps(response_data, indent=2)}")
            except:
                response_data = response.text
                print(f"📄 Response body: {response_data}")
            
            if response.status_code == 200 or response.status_code == 202:
                print(f"✅ Bot dołączył do czatu!")
                return {"success": True}
            else:
                print(f"❌ Błąd dołączania: {response_data}")
                return {"success": False, "error": response_data}
    
    async def send_message_as_bot(self, chat_id: str, text: str, bot_token: str) -> Dict[str, Any]:
        """Wysyła wiadomość jako bot przez Web API - używając tokena bota"""
        
        headers = {
            "Authorization": f"Bearer {bot_token}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "chat_id": chat_id,
            "event": {
                "type": "message",
                "text": text,
                "visibility": "all"
            }
        }
        
        url = f"{self.base_url}/v3.6/agent/action/send_event"
        
        print(f"\n📤 Wysyłanie wiadomości do czatu {chat_id}")
        print(f"📋 Headers: Authorization: Bearer {bot_token[:20]}...")
        print(f"📦 Payload: {json.dumps(payload, indent=2)}")
        
        async with httpx.AsyncClient() as client:
            response = await client.post(url, headers=headers, json=payload)
            print(f"📊 Response status: {response.status_code}")
            
            if response.status_code == 202:
                try:
                    response_data = response.json()
                    event_id = response_data.get("event_id", "unknown")
                    logger.info(f"✅ Wiadomość wysłana! Event ID: {event_id}")
                    print(f"✅ Sukces! Event ID: {event_id}")
                    return {"success": True, "event_id": event_id}
                except:
                    logger.info(f"✅ Wiadomość wysłana!")
                    print(f"✅ Sukces!")
                    return {"success": True}
            else:
                try:
                    error_data = response.json()
                    print(f"❌ Błąd: {json.dumps(error_data, indent=2)}")
                except:
                    print(f"❌ Błąd: {response.text}")
                return {"success": False, "error": response.text}
    
    async def test_connection(self) -> tuple[bool, str]:
        try:
            print("\n🔍 Testowanie połączenia z API...")
            result = await self._request(
                "POST",
                "/v3.6/configuration/action/list_agents",
                json={}
            )
            
            if isinstance(result, dict) and "agents" in result:
                agents_count = len(result.get("agents", []))
                print(f"✅ Połączenie działa. Znaleziono {agents_count} agentów")
                return True, f"OK, {agents_count} agentów"
            else:
                print(f"✅ Połączenie działa")
                return True, "OK"
                
        except Exception as e:
            error_msg = str(e)
            print(f"❌ Problem z połączeniem: {error_msg}")
            return False, error_msg

    async def transfer_chat(self, chat_id: str, target_group_id: int, bot_id: str) -> Dict[str, Any]:
        agent_token = await auth_client.get_valid_token()
        
        headers = {
            "Authorization": f"Bearer {agent_token}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "id": chat_id,
            "target": {
                "type": "group",
                "ids": [target_group_id]
            },
            "ignore_agents_availability": False
        }
        
        url = f"{self.base_url}/v3.6/agent/action/transfer_chat"
        
        print(f"\n🔄 Przekazywanie czatu {chat_id} do grupy {target_group_id}")
        print(f"📦 Payload: {json.dumps(payload, indent=2)}")
        
        async with httpx.AsyncClient() as client:
            response = await client.post(url, headers=headers, json=payload)
            print(f"📊 Response status: {response.status_code}")
            
            if response.status_code == 202:
                logger.info(f"✅ Czat {chat_id} przekazany do grupy {target_group_id}")
                return {"success": True}
            else:
                logger.error(f"❌ Błąd transferu: {response.text}")
                return {"success": False, "error": response.text}
    
    async def close(self):
        await self.http_client.aclose()

api_client = LiveChatAPIClient()

# ============================================
# GLM SERVICE
# ============================================

class GLMService:
    """Serwis do obsługi GLM-4.7 API od Z.ai"""
    
    def __init__(self, api_key: str = None, model: str = None, base_url: str = None):
        self.api_key = api_key or os.getenv("GLM_API_KEY")
        self.model = model or os.getenv("GLM_MODEL", "glm-4.7-flash")
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
        stop: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Generuje odpowiedź z GLM
        """
        if not self.api_key:
            return {"success": False, "text": "Przepraszam, nie mogę teraz odpowiedzieć (brak API)."}
        
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
            "stream": False
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
                    logger.info(f"✅ GLM response in {elapsed:.2f}s, tokens: {usage.get('total_tokens', '?')}")
                    
                    return {
                        "success": True,
                        "text": content,
                        "usage": usage,
                        "model": data.get("model", self.model),
                        "elapsed": elapsed
                    }
                else:
                    error_text = await response.aread()
                    logger.error(f"❌ GLM error {response.status_code}: {error_text}")
                    return {
                        "success": False,
                        "text": "Przepraszam, wystąpił błąd podczas generowania odpowiedzi.",
                        "error": f"HTTP {response.status_code}: {error_text}"
                    }
                    
        except Exception as e:
            logger.error(f"❌ GLM exception: {e}")
            return {
                "success": False,
                "text": "Przepraszam, nie mogę teraz odpowiedzieć (błąd połączenia).",
                "error": str(e)
            }
    
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
            "api_key_present": bool(self.api_key)
        }

# Singleton dla GLM
_glm_service_instance = None

def get_glm_service():
    """Zwraca singleton GLM service"""
    global _glm_service_instance
    if _glm_service_instance is None:
        _glm_service_instance = GLMService()
    return _glm_service_instance

# ============================================
# ZAAWANSOWANA KLASA BOTA Z RAG (NOWY) I GLM
# ============================================

class YourBot:
    
    def __init__(self):
        logger.info("Inicjalizacja bota z nowym RAG i GLM...")
        self.conversation_state = {}
        self.rag_service = None
        self.glm_service = get_glm_service()
        
        # Inicjalizuj RAG asynchronicznie
        asyncio.create_task(self._init_rag())
        
        logger.info("✅ Bot z GLM gotowy")
    
    async def _init_rag(self):
        """Inicjalizuje RAG service"""
        try:
            self.rag_service = await get_rag_service()
            health = await self.rag_service.health_check()
            stats = await self.rag_service.get_stats()
            
            if health.get("status") == "healthy":
                logger.info(f"✅ Nowy RAG service zainicjalizowany: {stats.get('total_chunks', 0)} dokumentów")
            else:
                logger.warning(f"⚠️ Nowy RAG service w stanie: {health.get('status')}")
        except Exception as e:
            logger.error(f"❌ Nowy RAG init failed: {e}")
    
    async def process_message(self, text: str, chat_id: str = None) -> tuple[str, bool]:
        text_lower = text.lower().strip()
        
        # Inicjalizuj stan dla nowej rozmowy
        if chat_id and chat_id not in self.conversation_state:
            self.conversation_state[chat_id] = {
                "failed_attempts": 0,
                "last_topic": None,
                "context": []
            }
        
        if chat_id:
            state = self.conversation_state[chat_id]
        else:
            state = {"failed_attempts": 0}
        
        # Sprawdź czy to prośba o konsultanta
        handoff_keywords = ['konsultant', 'człowiek', 'agent', 'handoff', 'konsultanta', 'człowiekiem']
        if any(keyword in text_lower for keyword in handoff_keywords):
            return "Łączę z konsultantem...", True
        
        # Użyj nowego RAG jeśli dostępny
        rag_results = {
            "has_data": False,
            "skip_rag": False,
            "below_threshold": False,
            "confidence": 0.0,
            "intent": "general",
            "detected_models": [],
            "tech": False,
            "documents": [],
            "sources": []
        }
        
        if self.rag_service:
            try:
                rag_results = await self.rag_service.retrieve_with_intent_check(
                    query=text,
                    top_k=3,
                    confidence_threshold=0.5
                )
                
                if rag_results.get("has_data"):
                    print(f"✅ Nowy RAG znalazł dane, confidence: {rag_results.get('confidence', 0):.2f}")
                elif rag_results.get("skip_rag"):
                    print(f"⏭️ Nowy RAG pomija (intent: {rag_results.get('intent', 'unknown')})")
                else:
                    print(f"ℹ️ Nowy RAG: {rag_results.get('documents_retrieved', 0)} dokumentów, conf: {rag_results.get('confidence', 0):.2f}")
            except Exception as e:
                logger.error(f"Nowy RAG error: {e}")
        
        # Przygotuj kontekst z RAG
        rag_context = ""
        sources = []
        
        if rag_results.get("has_data") and rag_results.get("documents"):
            valid_docs = rag_results.get("documents", [])
            
            if valid_docs:
                context_parts = []
                for i, doc in enumerate(valid_docs[:3], 1):
                    content = doc.get('content', '')[:500] + "..." if len(doc.get('content', '')) > 500 else doc.get('content', '')
                    metadata = doc.get('metadata', {})
                    source = metadata.get('title', 'Źródło')[:50]
                    context_parts.append(f"[{i}] Z {source}:\n{content}")
                    
                    # Zapisz źródła dla odpowiedzi
                    sources.append({
                        "title": metadata.get('title', 'Źródło')[:100],
                        "content": content[:200] + "..." if len(content) > 200 else content,
                        "relevance": doc.get('score', 0.0)
                    })
                
                rag_context = "\n\n".join(context_parts)
        
        # Wykryj typ pytania (jeśli RAG nie podał intent)
        question_type = rag_results.get('intent', 'general')
        if question_type == 'general':
            if any(word in text_lower for word in ['specyfikacj', 'dane', 'parametr', 'silnik', 'moc', 'km']):
                question_type = "specyfikacje"
            elif any(word in text_lower for word in ['różni', 'różnica', 'porównaj', 'vs', 'contra']):
                question_type = "porownanie"
            elif any(word in text_lower for word in ['rodzin', 'dzieci', 'osób', 'przestrzeń']):
                question_type = "rodzinny"
            elif any(word in text_lower for word in ['sport', 'szybk', 'mocny', 'm pakiet']):
                question_type = "sportowy"
            elif any(word in text_lower for word in ['elektryczn', 'ev', 'elektryk', 'hybryd']):
                question_type = "elektryczny"
            elif any(word in text_lower for word in ['cen', 'koszt', 'drogi', 'wyposażenie']):
                question_type = "cena"
        
        # Wykryj modele BMW (jeśli RAG nie podał)
        detected_models = rag_results.get('detected_models', [])
        if not detected_models:
            bmw_models = ['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'xm',
                         'i3', 'i4', 'i5', 'i7', 'i8', 'ix',
                         'm2', 'm3', 'm4', 'm5', 'm8', 'z4',
                         'seria 2', 'seria 3', 'seria 4', 'seria 5', 'seria 7', 'seria 8']
            
            for model in bmw_models:
                if model in text_lower:
                    model_upper = model.upper()
                    if 'SERIA' in model_upper:
                        model_upper = model_upper.replace('SERIA', 'Seria')
                    detected_models.append(model_upper)
        
        # System prompt dla GLM
        system_prompt = """Jesteś Leo - ekspertem BMW w ZK Motors, oficjalnym dealerze BMW i MINI.

ZASADY:
1. Odpowiadaj KONKRETNIE i RZETELNIE - maksymalnie 5-6 zdań
2. Używaj DANYCH Z KONTEKSTU - nie wymyślaj informacji
3. Bądź profesjonalny i pomocny
4. Jeśli brakuje danych w kontekście - powiedz to i zaproś do salonu
5. Zawsze kończ zachętą do kontaktu z ZK Motors
6. Używaj polskiego języka, bądź uprzejmy
7. Jesteś przedstawicielem ZK Motors - oficjalnego dealera BMW"""

        # Kontekst z RAG
        context_section = "BRAK DANYCH W BAZIE"
        if rag_context:
            context_section = f"DANE Z BAZY WIEDZY BMW:\n{rag_context}"
        
        # Historia (ostatnie 2 wiadomości)
        history_section = ""
        if chat_id and len(state.get("context", [])) > 0:
            recent = state["context"][-2:]
            history_lines = []
            for entry in recent:
                role = "Klient" if entry.get("role") == "user" else "Ty"
                history_lines.append(f"{role}: {entry.get('content', '')}")
            if history_lines:
                history_section = "OSTATNIA ROZMOWA:\n" + "\n".join(history_lines)
        
        # Informacje o wykrytych modelach
        models_info = ""
        if detected_models:
            models_info = f"Wykryte modele w pytaniu: {', '.join(detected_models)}"
        
        # Zbuduj prompt
        user_prompt = f"""PYTANIE KLIENTA: {text}

{models_info}

{context_section}

{history_section}

ODPOWIEDŹ (po polsku, 5-6 zdań):"""
        
        # Wywołaj GLM
        glm_result = await self.glm_service.generate(
            prompt=user_prompt,
            system_prompt=system_prompt,
            temperature=0.7,
            max_tokens=1000
        )
        
        if glm_result.get("success"):
            response = glm_result.get("text", "")
            print(f"✅ GLM wygenerował odpowiedź")
        else:
            print(f"⚠️ GLM błąd, używam fallback: {glm_result.get('error')}")
            # Fallback do reguł
            response = self._fallback_response(text_lower, question_type, detected_models, rag_context, state)
        
        # Zapisz kontekst
        if chat_id:
            state["context"].append({"role": "user", "content": text, "timestamp": datetime.now().isoformat()})
            state["context"].append({"role": "assistant", "content": response, "timestamp": datetime.now().isoformat()})
            # Ogranicz kontekst do 5 ostatnich wiadomości
            if len(state["context"]) > 10:
                state["context"] = state["context"][-10:]
            state["last_topic"] = question_type
            state["failed_attempts"] = 0
        
        return response, False
    
    def _fallback_response(self, text_lower: str, question_type: str, detected_models: List[str], 
                          rag_context: str, state: Dict) -> str:
        """Fallback odpowiedzi gdy GLM nie działa"""
        
        if question_type == "specyfikacje" and rag_context:
            return f"Na podstawie dostępnych danych:\n\n{rag_context[:300]}...\n\nCzy chcesz poznać więcej szczegółów? Zapraszamy do salonu ZK Motors!"
        elif question_type == "porownanie" and rag_context:
            return f"Porównanie modeli BMW:\n\n{rag_context[:300]}...\n\nKtóry z tych modeli Cię interesuje? Możesz umówić się na jazdę testową w ZK Motors."
        elif detected_models and rag_context:
            model_str = ', '.join(detected_models)
            return f"Oto informacje o {model_str}:\n\n{rag_context[:300]}...\n\nWięcej szczegółów w salonie ZK Motors!"
        elif any(word in text_lower for word in ['cześć', 'witaj', 'hej', 'dzień dobry']):
            return "Cześć! Jestem Leo, ekspertem BMW w ZK Motors. W czym mogę pomóc?"
        elif any(word in text_lower for word in ['dziękuję', 'dzięki', 'thx']):
            return "Proszę bardzo! Czy mogę pomóc w czymś jeszcze? Zapraszam do kontaktu."
        elif any(word in text_lower for word in ['godziny', 'otwarcia', 'czynne']):
            return "Jesteśmy czynni od poniedziałku do piątku w godzinach 9:00-17.00, a w soboty 9:00-14:00. Zapraszamy do salonu ZK Motors!"
        elif any(word in text_lower for word in ['adres', 'gdzie', 'siedziba']):
            return "Nasza siedziba znajduje się przy ul. Przykładowej 123 w Warszawie. Serdecznie zapraszamy do salonu!"
        elif 'bmw' in text_lower and not rag_context:
            return "BMW oferuje wiele modeli dopasowanych do różnych potrzeb. Który segment Cię interesuje? SUV-y (X1, X3, X5), sedany (Seria 3, 5), czy może sportowe auta (M2, M4)? Zapraszam do salonu!"
        else:
            state["failed_attempts"] = state.get("failed_attempts", 0) + 1
            
            if state["failed_attempts"] >= 3:
                return "Przepraszam, nie mogę pomóc. Łączę z konsultantem."
            
            if detected_models:
                return f"Przepraszam, nie znalazłem wystarczających informacji o {', '.join(detected_models)}. Czy możesz sprecyzować pytanie? A może chcesz umówić się na jazdę próbną w ZK Motors?"
            else:
                return "Przepraszam, nie zrozumiałem. Czy możesz powiedzieć inaczej?"

class LiveChatBotIntegration:
    
    def __init__(self, your_bot: YourBot):
        self.bot = your_bot
        self.bot_agent_id: Optional[str] = None
        self.bot_secret: Optional[str] = None
        self.bot_token: Optional[str] = None
        self.target_group_id = int(os.getenv("LIVECHAT_TARGET_GROUP_ID", "0"))
        self._running = False
        self.connection_ok = False
        self.webhook_url = f"{NGROK_URL}/webhook"
    
    async def start(self):
        if self._running:
            return
        
        try:
            config = load_bot_config()
            
            if "bot_id" in config and config["bot_id"]:
                self.bot_agent_id = config["bot_id"]
                self.bot_secret = config.get("bot_secret", "")
                print(f"✅ Wczytano bota z konfiguracji: {self.bot_agent_id}")
                print(f"🎯 Grupa docelowa: {self.target_group_id}")
                print(f"🌍 Webhook URL: {self.webhook_url}")
                
                if self.bot_secret:
                    print(f"\n🔑 Próba wygenerowania tokena bota...")
                    self.bot_token = await api_client.get_bot_token(self.bot_agent_id, self.bot_secret)
                    if self.bot_token:
                        print(f"✅ Token bota wygenerowany pomyślnie")
                    else:
                        print(f"⚠️ Nie udało się wygenerować tokena bota - sprawdź bot_secret w configu")
                else:
                    print(f"⚠️ Brak bot_secret w configu - nie można wygenerować tokena bota")
                
                ok, msg = await api_client.test_connection()
                self.connection_ok = ok
                
                if ok:
                    print(f"✅ Połączono z API LiveChat: {msg}")
                    
                    print(f"\n🤖 Próba aktywacji bota {self.bot_agent_id}...")
                    activation_result = await api_client.set_routing_status(self.bot_agent_id, "accepting_chats")
                    if activation_result.get("success"):
                        print(f"✅ Bot aktywowany pomyślnie")
                    else:
                        print(f"⚠️ Problem z aktywacją bota: {activation_result.get('error')}")
                        print(f"⚠️ Bot może nie działać poprawnie - sprawdź uprawnienia tokena")
                    
                else:
                    print(f"⚠️ Problem z połączeniem: {msg}")
                    print(f"⚠️ Bot uruchomiony ale może nie działać - sprawdź token")
                
                self._running = True
                logger.info("✅ Bot LiveChat zainicjalizowany")
                return
            
            error_msg = "❌ BRAK ID BOTA W KONFIGURACJI! Utwórz plik bot_config.json z Twoim ID bota"
            print(error_msg)
            raise Exception(error_msg)
            
        except Exception as e:
            logger.error(f"Błąd inicjalizacji: {e}")
            raise
    
    async def handle_webhook(self, payload: dict) -> None:
        try:
            action = payload.get("action")
            print(f"\n📨 Webhook action: {action}")
            
            if action == "incoming_event":
                await self._handle_incoming_event(payload)
            elif action == "incoming_chat":
                print(f"🆕 Nowy czat rozpoczęty")
                print(f"📦 Payload: {json.dumps(payload, indent=2)}")
            else:
                print(f"⏭️ Nieobsługiwana akcja: {action}")
                print(f"📦 Payload: {json.dumps(payload, indent=2)}")
                
        except Exception as e:
            logger.error(f"Błąd webhooka: {e}")
            traceback.print_exc()
    
    async def _handle_incoming_event(self, payload: dict):
        try:
            payload_data = payload.get("payload", {})
            event = payload_data.get("event", {})
            chat_id = payload_data.get("chat_id")
            
            print(f"📊 Chat ID: {chat_id}")
            print(f"📊 Event type: {event.get('type')}")
            
            if event.get("type") != "message":
                print(f"⏭️ Pomijam event typu: {event.get('type')}")
                return
            
            text = event.get("text", "")
            author_id = event.get("author_id")
            
            print(f"👤 Author ID: {author_id}")
            print(f"💬 Wiadomość: {text[:100]}...")
            
            if author_id == self.bot_agent_id:
                print(f"⏭️ Ignoruję wiadomość od samego bota")
                return
            
            if not self.bot_agent_id:
                print("❌ Brak ID bota - nie mogę odpowiedzieć")
                return
            
            if not self.connection_ok:
                print(f"⚠️ Brak połączenia z API - nie wysyłam odpowiedzi")
                return
            
            print(f"\n🔍 Próba dołączenia bota do czatu {chat_id}...")
            join_result = await api_client.join_chat(chat_id, self.bot_agent_id)
            
            if not join_result.get("success"):
                print(f"⚠️ Nie udało się dołączyć do czatu - próbuję mimo to...")
            else:
                print(f"✅ Bot dołączył do czatu")
            
            await asyncio.sleep(1)
            
            print(f"\n🤖 Przetwarzanie wiadomości z nowym RAG i GLM...")
            response, should_transfer = await self.bot.process_message(text, chat_id)
            print(f"💬 Odpowiedź: {response}")
            print(f"🔄 Transfer: {should_transfer}")
            
            if response:
                if not self.bot_token:
                    print(f"❌ Brak tokena bota - nie mogę wysłać wiadomości")
                else:
                    print(f"📤 Wysyłanie odpowiedzi z tokenem bota...")
                    result = await api_client.send_message_as_bot(chat_id, response, self.bot_token)
                    if result.get("success"):
                        print(f"✅ Odpowiedź wysłana!")
                    else:
                        print(f"❌ Błąd wysyłania: {result.get('error')}")
            
            if should_transfer:
                print(f"🔄 Przekazywanie czatu...")
                transfer_result = await api_client.transfer_chat(chat_id, self.target_group_id, self.bot_agent_id)
                if transfer_result.get("success"):
                    print(f"✅ Czat przekazany")
                else:
                    print(f"❌ Błąd transferu: {transfer_result.get('error')}")
                
        except Exception as e:
            logger.error(f"Błąd w _handle_incoming_event: {e}")
            traceback.print_exc()
    
    async def stop(self):
        self._running = False
        logger.info("Bot zatrzymany")
    
    def is_running(self) -> bool:
        return self._running
    
    def get_bot_agent_id(self) -> Optional[str]:
        return self.bot_agent_id
    
    def is_connected(self) -> bool:
        return self.connection_ok

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.sessions import SessionMiddleware

app = FastAPI(title="Bot LiveChat")

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])
app.add_middleware(SessionMiddleware, secret_key=os.getenv("SECRET_KEY", secrets.token_urlsafe(32)))

templates = Jinja2Templates(directory=os.path.join(os.path.dirname(__file__), "templates"))
bot_integration: Optional[LiveChatBotIntegration] = None

@app.get("/")
async def home():
    return RedirectResponse(url="/panel")

@app.get("/panel")
async def panel(request: Request):
    return templates.TemplateResponse("panel.html", {"request": request})

@app.get("/token-page", response_class=HTMLResponse)
async def token_page(request: Request):
    return templates.TemplateResponse("token-page.html", {"request": request})

@app.get("/test")
async def test():
    return {
        "status": "ok", 
        "message": "Webhook endpoint is accessible", 
        "time": str(datetime.now()),
        "ngrok_url": NGROK_URL,
        "webhook_url": f"{NGROK_URL}/webhook"
    }

@app.post("/webhook")
async def webhook_receiver(request: Request):
    try:
        body = await request.body()
        print(f"\n{'='*60}")
        print(f"📨 RAW WEBHOOK RECEIVED at {datetime.now().strftime('%H:%M:%S')}")
        print(f"Headers: {dict(request.headers)}")
        print(f"Raw body: {body.decode('utf-8', errors='ignore')[:500]}")
        
        try:
            payload = json.loads(body)
            print(f"📦 Parsed JSON: {json.dumps(payload, indent=2)}")
        except json.JSONDecodeError as e:
            print(f"❌ Failed to parse JSON: {e}")
            payload = {}
        
        if bot_integration and bot_integration.is_running():
            print(f"✅ Forwarding to bot handler")
            asyncio.create_task(bot_integration.handle_webhook(payload))
        else:
            print(f"⚠️ Bot not running, ignoring webhook")
            print(f"💡 Start the bot in the panel first: http://localhost:8000/panel")
        
        return JSONResponse(content={"status": "ok"}, status_code=200)
        
    except Exception as e:
        print(f"❌ Webhook error: {e}")
        traceback.print_exc()
        return JSONResponse(content={"status": "error", "message": str(e)}, status_code=500)

@app.get("/api/bot/status")
async def get_status():
    return {
        "running": bot_integration is not None and bot_integration.is_running(),
        "bot_agent_id": bot_integration.get_bot_agent_id() if bot_integration else None,
        "target_group_id": bot_integration.target_group_id if bot_integration else None,
        "connected": bot_integration.is_connected() if bot_integration else False,
        "webhook_url": bot_integration.webhook_url if bot_integration else None
    }

@app.get("/api/bot/config")
async def get_bot_config():
    config = load_bot_config()
    return {
        "has_bot_config": bool(config.get("bot_id")),
        "bot_id": config.get("bot_id"),
        "running_bot_id": bot_integration.get_bot_agent_id() if bot_integration else None
    }

@app.get("/api/auth/check")
async def check_auth():
    try:
        result = await api_client._request(
            "POST",
            "/v3.6/configuration/action/list_agents",
            json={}
        )
        
        if isinstance(result, dict):
            agents = result.get("agents", [])
            return {
                "valid": True,
                "message": f"Token ważny, znaleziono {len(agents)} agentów",
                "agents_count": len(agents)
            }
        else:
            return {
                "valid": True,
                "message": "Token ważny",
                "agents_count": 0
            }
            
    except Exception as e:
        error_str = str(e)
        return {
            "valid": False,
            "message": f"Token nieważny: {error_str[:200]}",
            "error": error_str
        }

@app.get("/api/webhooks/list")
async def list_webhooks():
    try:
        token = await auth_client.get_valid_token()
        
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "owner_client_id": auth_client.client_id
        }
        
        async with httpx.AsyncClient() as client:
            response = await client.post(
                "https://api.livechatinc.com/v3.6/configuration/action/list_webhooks",
                headers=headers,
                json=payload
            )
            result = response.json()
            print(f"📋 Webhooks list: {json.dumps(result, indent=2)}")
            return result
    except Exception as e:
        print(f"❌ Error listing webhooks: {e}")
        return {"error": str(e)}

@app.post("/api/bot/start")
async def start_bot():
    global bot_integration
    if bot_integration and bot_integration.is_running():
        return {"success": True, "message": "Bot już działa"}
    
    your_bot = YourBot()
    bot_integration = LiveChatBotIntegration(your_bot)
    asyncio.create_task(bot_integration.start())
    await asyncio.sleep(2)
    return {"success": True, "message": "Bot uruchomiony"}

@app.post("/api/bot/stop")
async def stop_bot():
    global bot_integration
    if bot_integration:
        await bot_integration.stop()
        bot_integration = None
    return {"success": True, "message": "Bot zatrzymany"}

@app.post("/register-webhook")
async def register_webhook():
    try:
        token = await auth_client.get_valid_token()
        webhook_url = f"{NGROK_URL}/webhook"
        
        print(f"\n🔍 Registering webhook with URL: {webhook_url}")
        
        print("📋 Sprawdzanie istniejących webhooków...")
        list_response = await list_webhooks()
        print(f"Istniejące webhooki: {json.dumps(list_response, indent=2)}")
        
        secret_key = secrets.token_urlsafe(32)
        
        payload = {
            "url": webhook_url,
            "action": "incoming_event",
            "description": "Bot webhook",
            "type": "bot",
            "owner_client_id": auth_client.client_id,
            "secret_key": secret_key
        }
        
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json"
        }
        
        print(f"📦 Rejestracja z payload: {json.dumps(payload, indent=2)}")
        
        async with httpx.AsyncClient() as client:
            response = await client.post(
                "https://api.livechatinc.com/v3.6/configuration/action/register_webhook",
                headers=headers,
                json=payload
            )
            result = response.json()
            print(f"✅ Wynik rejestracji: {json.dumps(result, indent=2)}")
            return result
    except Exception as e:
        print(f"❌ Błąd rejestracji webhooka: {e}")
        traceback.print_exc()
        return {"error": str(e)}

@app.post("/unregister-webhook")
async def unregister_webhook(request: Request):
    try:
        data = await request.json()
        token = await auth_client.get_valid_token()
        
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json"
        }
        
        if "id" in data:
            payload = {
                "id": data["id"],
                "owner_client_id": auth_client.client_id
            }
        elif "url" in data:
            payload = {
                "url": data["url"],
                "owner_client_id": auth_client.client_id
            }
        else:
            return {"error": "Wymagane 'id' lub 'url'"}
        
        print(f"\n🔍 Unregistering webhook with payload: {json.dumps(payload, indent=2)}")
        
        async with httpx.AsyncClient() as client:
            response = await client.post(
                "https://api.livechatinc.com/v3.6/configuration/action/unregister_webhook",
                headers=headers,
                json=payload
            )
            result = response.json()
            print(f"✅ Unregistered webhook: {result}")
            return result
    except Exception as e:
        print(f"❌ Error unregistering webhook: {e}")
        traceback.print_exc()
        return {"error": str(e)}

@app.post("/unregister-all-webhooks")
async def unregister_all_webhooks():
    try:
        token = await auth_client.get_valid_token()
        
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json"
        }
        
        list_response = await list_webhooks()
        
        webhooks = []
        if isinstance(list_response, dict) and "webhooks" in list_response:
            webhooks = list_response["webhooks"]
        elif isinstance(list_response, list):
            webhooks = list_response
        
        removed = []
        errors = []
        
        for webhook in webhooks:
            webhook_id = webhook.get("id") if isinstance(webhook, dict) else None
            
            if webhook_id:
                try:
                    payload = {
                        "id": webhook_id,
                        "owner_client_id": auth_client.client_id
                    }
                    
                    async with httpx.AsyncClient() as client:
                        response = await client.post(
                            "https://api.livechatinc.com/v3.6/configuration/action/unregister_webhook",
                            headers=headers,
                            json=payload
                        )
                        if response.status_code == 202:
                            removed.append(webhook_id)
                            print(f"✅ Usunięto webhook: {webhook_id}")
                        else:
                            errors.append(webhook_id)
                            print(f"❌ Nie udało się usunąć: {webhook_id}")
                except Exception as e:
                    errors.append(webhook_id)
                    print(f"❌ Błąd: {e}")
        
        return {
            "success": True,
            "removed": removed,
            "errors": errors,
            "message": f"Usunięto {len(removed)} webhooków"
        }
    except Exception as e:
        return {"error": str(e)}

@app.post("/create-bot")
async def create_bot_endpoint():
    """Endpoint do tworzenia nowego bota przez API"""
    try:
        bot_id, bot_secret = await api_client.create_bot("ChatbotBMW")
        
        # Automatycznie zapisz do configu
        save_bot_config({
            "bot_id": bot_id,
            "bot_secret": bot_secret
        })
        
        return {
            "success": True,
            "bot_id": bot_id,
            "bot_secret": bot_secret,
            "message": "Nowy bot utworzony i zapisany w bot_config.json"
        }
    except Exception as e:
        return {"error": str(e), "message": "Nie udało się utworzyć bota"}

@app.get("/register-webhook-simple")
async def register_webhook_simple():
    return await register_webhook()

@app.get("/unregister-webhook-by-id/{webhook_id}")
async def unregister_webhook_by_id(webhook_id: str):
    try:
        token = await auth_client.get_valid_token()
        
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "id": webhook_id,
            "owner_client_id": auth_client.client_id
        }
        
        async with httpx.AsyncClient() as client:
            response = await client.post(
                "https://api.livechatinc.com/v3.6/configuration/action/unregister_webhook",
                headers=headers,
                json=payload
            )
            return response.json()
    except Exception as e:
        return {"error": str(e)}

@app.get("/api/bot/routing-status/{bot_id}")
async def get_bot_routing_status(bot_id: str):
    try:
        token = await auth_client.get_valid_token()
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json"
        }
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"https://api.livechatinc.com/v3.6/agent/action/get_routing_status",
                headers=headers,
                json={"id": bot_id}
            )
            return response.json()
    except Exception as e:
        return {"error": str(e)}

# ============================================
# RAG ENDPOINTS (NOWY RAG)
# ============================================

@app.get("/rag/info")
async def get_rag_info():
    """Informacje o RAG"""
    try:
        rag_service = await get_rag_service()
        health = await rag_service.health_check()
        stats = await rag_service.get_stats()
        
        return {
            "healthy": health.get("status") == "healthy" and not stats.get("is_dummy", False),
            "documents": stats.get("documents_in_store", 0),
            "queries_processed": stats.get("queries_processed", 0),
            "cache_hit_rate": stats.get("cache_hit_rate", 0),
            "intent_skipped": stats.get("intent_skipped", 0),
            "confidence_threshold": stats.get("min_confidence_threshold", 0.5),
            "status": health.get("status", "unknown"),
            "available": RAG_AVAILABLE
        }
    except Exception as e:
        return {
            "healthy": False,
            "error": str(e),
            "available": RAG_AVAILABLE
        }

@app.get("/rag/test")
async def test_rag(query: str = "BMW X5 dane techniczne"):
    """Test RAG z podanym zapytaniem"""
    try:
        rag_service = await get_rag_service()
        result = await rag_service.retrieve_with_intent_check(
            query=query,
            top_k=3,
            confidence_threshold=0.5
        )
        
        return {
            "query": query,
            "has_data": result.get("has_data", False),
            "skip_rag": result.get("skip_rag", False),
            "confidence": result.get("confidence", 0.0),
            "intent": result.get("intent", "general"),
            "detected_models": result.get("detected_models", []),
            "documents_count": len(result.get("documents", [])),
            "sources_count": len(result.get("sources", [])),
            "processing_time": result.get("processing_time", 0)
        }
    except Exception as e:
        return {"error": str(e)}

# ============================================
# GLM ENDPOINTS
# ============================================

@app.get("/glm/info")
async def get_glm_info():
    """Informacje o GLM"""
    try:
        glm_service = get_glm_service()
        health = await glm_service.health_check()
        stats = await glm_service.get_stats()
        
        return {
            "available": stats.get("api_key_present", False),
            "model": stats.get("model", "unknown"),
            "status": health.get("status", "unknown"),
            "test_response": health.get("test_response", ""),
            "details": stats
        }
    except Exception as e:
        return {
            "available": False,
            "error": str(e)
        }

@app.post("/glm/test")
async def test_glm(query: str = "Powiedz coś o BMW X5"):
    """Test GLM z podanym zapytaniem"""
    try:
        glm_service = get_glm_service()
        
        result = await glm_service.generate(
            prompt=query,
            system_prompt="Jesteś ekspertem BMW. Odpowiadaj krótko i konkretnie.",
            max_tokens=500
        )
        
        return {
            "query": query,
            "success": result.get("success", False),
            "response": result.get("text", ""),
            "usage": result.get("usage", {}),
            "elapsed": result.get("elapsed", 0)
        }
    except Exception as e:
        return {"error": str(e)}

@app.on_event("startup")
async def startup():
    print("\n" + "="*60)
    print("🚀 SERWER BOTA LIVECHAT Z NOWYM RAG I GLM")
    print("="*60)
    print("📊 Panel: http://localhost:8000/panel")
    print("📥 Webhook: http://localhost:8000/webhook")
    print("🔑 Token page: http://localhost:8000/token-page")
    print("🧪 Test: http://localhost:8000/test")
    print("🌍 Ngrok URL:", NGROK_URL)
    print("🌍 Webhook URL:", f"{NGROK_URL}/webhook")
    print("📝 Rejestracja webhooka (POST): http://localhost:8000/register-webhook")
    print("📝 Rejestracja webhooka (GET test): http://localhost:8000/register-webhook-simple")
    print("🗑️ Usuń webhook po ID (GET): http://localhost:8000/unregister-webhook-by-id/ID")
    print("🗑️ Usuń wszystkie webhooki (POST): http://localhost:8000/unregister-all-webhooks")
    print("🤖 Utwórz nowego bota (POST): http://localhost:8000/create-bot")
    print("🔍 Sprawdź status routingu bota (GET): http://localhost:8000/api/bot/routing-status/ID_BOTA")
    print("📚 RAG info: http://localhost:8000/rag/info")
    print("🧪 Test RAG: http://localhost:8000/rag/test?query=BMW X5")
    print("🤖 GLM info: http://localhost:8000/glm/info")
    print("🧪 Test GLM (POST): http://localhost:8000/glm/test?query=BMW X5")
    print("="*60)
    
    # Sprawdź nowy RAG
    try:
        rag_service = await get_rag_service()
        rag_health = await rag_service.health_check()
        rag_stats = await rag_service.get_stats()
        
        if rag_health.get("status") == "healthy":
            print(f"✅ Nowy RAG: {rag_stats.get('documents_in_store', 0)} dokumentów")
        else:
            print(f"⚠️ Nowy RAG: {rag_health.get('status', 'unknown')}")
    except Exception as e:
        print(f"⚠️ Nowy RAG: nie udało się zainicjalizować ({e})")
    
    # Sprawdź GLM
    glm_service = get_glm_service()
    glm_health = await glm_service.health_check()
    glm_stats = await glm_service.get_stats()
    
    if glm_stats.get("api_key_present", False):
        print(f"✅ GLM: {glm_stats.get('model', 'unknown')} (API działa)")
    else:
        print("⚠️ GLM: BRAK KLUCZA API - dodaj GLM_API_KEY do .env")
    
    config = load_bot_config()
    if config.get("bot_id"):
        print(f"✅ Bot ID w configu: {config['bot_id']}")
        if config.get("bot_secret"):
            print(f"✅ Bot secret znaleziony w configu")
        else:
            print(f"⚠️ Brak bot_secret w configu - użyj /create-bot aby utworzyć nowego bota")
    else:
        print("❌ BRAK ID BOTA! Użyj /create-bot aby utworzyć nowego bota")
    
    token = auth_client.oauth_token
    if token and token != "NOWY_TOKEN":
        print(f"✅ Token OAuth skonfigurowany: {token[:20]}...")
        
        try:
            result = await api_client._request(
                "POST",
                "/v3.6/configuration/action/list_agents",
                json={}
            )
            if isinstance(result, dict):
                agents = result.get("agents", [])
                print(f"✅ Token działa! Znaleziono {len(agents)} agentów")
            else:
                print(f"✅ Token działa")
        except Exception as e:
            print(f"❌ Token NIE DZIAŁA: {e}")
            print("   Zdobądź nowy token na /token-page")
    else:
        print("❌ BRAK TOKENA OAuth! Wpisz token w linii 93")
    
    print("="*60 + "\n")

@app.on_event("shutdown")
async def shutdown():
    global bot_integration
    if bot_integration:
        await bot_integration.stop()
    await auth_client.close()
    await api_client.close()
    print("👋 Serwer zatrzymany")

async def main():
    logger.info("🚀 Uruchamianie bota z nowym RAG i GLM...")
    
    integration = LiveChatBotIntegration(YourBot())
    
    try:
        await integration.start()
        logger.info("✅ Bot zainicjalizowany. Naciśnij Ctrl+C aby zatrzymać.")
        while integration.is_running():
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        await integration.stop()
    finally:
        await auth_client.close()
        await api_client.close()

if __name__ == "__main__":
    if not os.getenv("RUNNING_IN_UVICORN"):
        asyncio.run(main())