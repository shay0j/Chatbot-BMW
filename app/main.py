# main.py - KOMPLETNY BOT Z INTEGRACJĄ LIVECHAT (wersja webhook + Web API)
import os
import asyncio
import time
import json
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass
from datetime import datetime

import httpx
from loguru import logger
from dotenv import load_dotenv
from pydantic import BaseModel
import secrets

# ============================================
# KONFIGURACJA I ZALADOWANIE ZMIENNYCH ŚRODOWISKOWYCH
# ============================================
load_dotenv()

# ============================================
# DEBUG - WYŚWIETL ZMIENNE ŚRODOWISKOWE
# ============================================
print("\n" + "="*60)
print("🔍 DEBUG - Environment variables for LiveChat:")
print(f"LIVECHAT_CLIENT_ID: {os.getenv('LIVECHAT_CLIENT_ID')}")
print(f"LIVECHAT_CLIENT_SECRET: {os.getenv('LIVECHAT_CLIENT_SECRET')[:5] if os.getenv('LIVECHAT_CLIENT_SECRET') else 'None'}... (częściowo ukryty)")
print(f"LIVECHAT_REDIRECT_URI: {os.getenv('LIVECHAT_REDIRECT_URI')}")
print(f"LIVECHAT_REGION: {os.getenv('LIVECHAT_REGION')}")
print(f"LIVECHAT_TARGET_GROUP_ID: {os.getenv('LIVECHAT_TARGET_GROUP_ID')}")
print(f"LIVECHAT_ORGANIZATION_ID: {os.getenv('LIVECHAT_ORGANIZATION_ID')}")
print("="*60 + "\n")

# ============================================
# MODELE PYDANTIC DLA LIVECHAT API
# ============================================

class TokenResponse(BaseModel):
    """Odpowiedź z tokenem OAuth"""
    access_token: str
    refresh_token: Optional[str] = None
    expires_in: int
    token_type: str = "bearer"

class BotCreateResponse(BaseModel):
    """Odpowiedź po utworzeniu bota"""
    id: str
    secret: str

class IncomingEvent(BaseModel):
    """Zdarzenie przychodzące (wiadomość)"""
    chat_id: str
    event: Dict[str, Any]
    author_id: Optional[str] = None

@dataclass
class MessageHandler:
    """Handler dla wiadomości"""
    chat_id: str
    message: str
    author_id: str

# ============================================
# KLASA AUTORYZACJI OAuth 2.0 (uproszczona)
# ============================================

class LiveChatAuth:
    """Klient autoryzacji OAuth 2.0 dla LiveChat"""
    
    def __init__(self):
        self.client_id = os.getenv("LIVECHAT_CLIENT_ID")
        self.client_secret = os.getenv("LIVECHAT_CLIENT_SECRET")
        self.redirect_uri = os.getenv("LIVECHAT_REDIRECT_URI", "http://localhost:8000/callback")
        self.region = os.getenv("LIVECHAT_REGION", "us")
        self.organization_id = os.getenv("LIVECHAT_ORGANIZATION_ID")
        
        if not self.client_id or not self.client_secret:
            logger.warning("Brak LIVECHAT_CLIENT_ID lub LIVECHAT_CLIENT_SECRET w .env")
        
        self.accounts_url = "https://accounts.livechatinc.com"
        self.api_url = "https://api.livechatinc.com"
        
        # Cache tokena
        self._token: Optional[Dict] = None
        self._token_expires_at: Optional[float] = None
        self.http_client = httpx.AsyncClient(timeout=30.0)
    
    async def get_valid_token(self) -> str:
        """Zwraca token od supportu (tymczasowo)"""
        print("🔑 Używam nowego tokena z webhooks.configuration:rw")
        return "eu-west3:LgrMu1VqLmix-r-7AE8HTyXyYoM"
    
    def is_authenticated(self) -> bool:
        """Sprawdza czy mamy ważny token"""
        return True
    
    async def close(self):
        """Zamyka klienta HTTP"""
        await self.http_client.aclose()

# Globalna instancja auth
auth_client = LiveChatAuth()

# ============================================
# KLIENT API LIVECHAT (dla Web API)
# ============================================

class LiveChatAPIClient:
    """Klient do REST API LiveChat"""
    
    def __init__(self):
        self.base_url = "https://api.livechatinc.com"
        self.http_client = httpx.AsyncClient(timeout=30.0)
    
    async def _request(self, method: str, path: str, **kwargs) -> Dict[str, Any]:
        """Wykonuje zapytanie z tokenem"""
        token = await auth_client.get_valid_token()
        
        headers = kwargs.pop("headers", {})
        headers["Authorization"] = f"Bearer {token}"
        headers["Content-Type"] = "application/json"
        
        url = f"{self.base_url}{path}"
        
        print(f"\n🔍 DEBUG - API Request: {method} {url}")
        print(f"📦 Request data: {json.dumps(kwargs.get('json', {}), indent=2, ensure_ascii=False)}")
        
        try:
            response = await self.http_client.request(
                method, url, headers=headers, **kwargs
            )
            
            print(f"📊 Response status: {response.status_code}")
            
            # PRÓBUJ ODCZYTAĆ TREŚĆ ODPOWIEDZI
            try:
                response_body = response.text
                print(f"📄 Response body: {response_body[:500]}")
                if response_body.strip().startswith('{'):
                    json_body = json.loads(response_body)
                    print(f"📋 JSON response: {json.dumps(json_body, indent=2, ensure_ascii=False)}")
            except:
                print("⚠️ Nie udało się odczytać odpowiedzi")
            
            if response.status_code >= 400:
                error_msg = f"Błąd API {response.status_code}"
                try:
                    error_data = response.json()
                    error_msg += f": {json.dumps(error_data)}"
                except:
                    error_msg += f": {response.text}"
                print(f"❌ {error_msg}")
                response.raise_for_status()
            
            return response.json()
            
        except Exception as e:
            print(f"💥 Błąd: {e}")
            raise
    
    async def create_bot(self, name: str) -> Tuple[str, str]:
        """Tworzy bota i zwraca (id, secret)"""
        print(f"\n🤖 Próba utworzenia bota z nazwą: {name}")
        
        data = {
            "name": name,
            "default_group_priority": "normal",
            "avatar_path": "https://example.com/avatar.png",
            "max_chats_count": 5,
            "timezone": "Europe/Warsaw"
        }
        
        try:
            result = await self._request(
                "POST",
                "/v3.6/configuration/action/create_bot",
                json=data
            )
            
            bot_id = result.get("id")
            bot_secret = result.get("secret")
            if bot_id and bot_secret:
                logger.info(f"✅ Bot utworzony: {bot_id}")
                print(f"🔑 Sekret bota: {bot_secret}")
                return bot_id, bot_secret
            else:
                raise Exception("Brak ID lub secret w odpowiedzi")
                
        except Exception as e:
            logger.error(f"❌ Nie udało się utworzyć bota: {e}")
            raise

    async def issue_bot_token(self, bot_id: str, bot_secret: str) -> str:
        """Generuje token dla bota"""
        print(f"\n🔑 Generowanie tokena dla bota: {bot_id}")
        
        client_id = auth_client.client_id
        organization_id = os.getenv("LIVECHAT_ORGANIZATION_ID")
        
        if not client_id:
            raise Exception("Brak LIVECHAT_CLIENT_ID w konfiguracji")
        if not organization_id:
            raise Exception("Brak LIVECHAT_ORGANIZATION_ID w .env")
        
        payload = {
            "bot_id": bot_id,
            "client_id": client_id,
            "organization_id": organization_id,
            "bot_secret": bot_secret
        }
        
        print(f"📦 Wysyłam payload: {payload}")
        
        try:
            result = await self._request(
                "POST",
                "/v3.6/configuration/action/issue_bot_token",
                json=payload
            )
            
            token = result.get("token")
            if token:
                logger.info(f"✅ Token bota wygenerowany")
                print(f"🔑 Token: {token[:20]}...")
                return token
            else:
                raise Exception("Brak tokena w odpowiedzi")
                
        except Exception as e:
            logger.error(f"❌ Nie udało się wygenerować tokena: {e}")
            raise

    async def send_message_as_bot(self, chat_id: str, text: str, bot_id: str) -> Dict[str, Any]:
        """Wysyła wiadomość jako bot przez Web API"""
        print(f"\n📤 Wysyłanie wiadomości jako bot {bot_id[:8]}... do czatu {chat_id}")
        
        token = await auth_client.get_valid_token()
        
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "Author-Id": bot_id  # Kluczowe – autoryzacja jako bot!
        }
        
        payload = {
            "chat_id": chat_id,
            "event": {
                "type": "message",
                "text": text
            }
        }
        
        url = f"{self.base_url}/v3.6/agent/action/send_event"
        
        async with httpx.AsyncClient() as client:
            response = await client.post(url, headers=headers, json=payload)
            if response.status_code == 202:
                logger.info(f"✅ Wiadomość wysłana")
                return {"success": True}
            else:
                logger.error(f"❌ Błąd wysyłania: {response.text}")
                return {"success": False, "error": response.text}

    async def transfer_chat(self, chat_id: str, target_group_id: int, bot_id: str) -> Dict[str, Any]:
        """Przekazuje czat do grupy agentów"""
        print(f"\n🔄 Przekazywanie czatu {chat_id} do grupy {target_group_id}")
        
        token = await auth_client.get_valid_token()
        
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "Author-Id": bot_id
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
        
        async with httpx.AsyncClient() as client:
            response = await client.post(url, headers=headers, json=payload)
            if response.status_code == 202:
                logger.info(f"✅ Czat przekazany")
                return {"success": True}
            else:
                logger.error(f"❌ Błąd transferu: {response.text}")
                return {"success": False, "error": response.text}

    async def list_agents(self) -> list:
        """Lista agentów"""
        try:
            result = await self._request(
                "POST", 
                "/v3.6/configuration/action/list_agents",
                json={}
            )
            agents = result.get("agents", [])
            print(f"📋 Znaleziono {len(agents)} agentów")
            return agents
        except Exception as e:
            logger.error(f"❌ Nie udało się pobrać listy agentów: {e}")
            return []
    
    async def close(self):
        """Zamyka klienta"""
        await self.http_client.aclose()

# Globalna instancja
api_client = LiveChatAPIClient()

# ============================================
# TWOJA KLASA BOTA
# ============================================

class YourBot:
    """Logika Twojego bota"""
    
    def __init__(self):
        logger.info("Inicjalizacja bota...")
        self.conversation_state = {}
        logger.info("Bot gotowy")
    
    async def process_message(self, text: str, chat_id: str = None) -> tuple[str, bool]:
        """Przetwarza wiadomość i zwraca (odpowiedź, czy transfer)"""
        text_lower = text.lower().strip()
        
        handoff_keywords = ['konsultant', 'człowiek', 'agent', 'handoff', 'konsultanta', 'człowiekiem']
        if any(keyword in text_lower for keyword in handoff_keywords):
            return "Łączę z konsultantem...", True
        
        if chat_id:
            if chat_id not in self.conversation_state:
                self.conversation_state[chat_id] = {"failed": 0}
            state = self.conversation_state[chat_id]
        else:
            state = {"failed": 0}
        
        if 'cześć' in text_lower or 'witaj' in text_lower or 'hej' in text_lower or 'dzień dobry' in text_lower:
            state["failed"] = 0
            return "Cześć! W czym mogę pomóc?", False
        elif 'dziękuję' in text_lower or 'dzięki' in text_lower or 'thx' in text_lower:
            state["failed"] = 0
            return "Proszę bardzo! Czy mogę pomóc w czymś jeszcze?", False
        elif 'godziny' in text_lower or 'otwarcia' in text_lower or 'czynne' in text_lower:
            state["failed"] = 0
            return "Jesteśmy czynni od poniedziałku do piątku w godzinach 9:00-17:00.", False
        elif 'adres' in text_lower or 'gdzie' in text_lower or 'siedziba' in text_lower:
            state["failed"] = 0
            return "Nasza siedziba znajduje się przy ul. Przykładowej 123 w Warszawie.", False
        else:
            state["failed"] = state.get("failed", 0) + 1
            if state["failed"] >= 3:
                return "Przepraszam, nie mogę pomóc. Łączę z konsultantem.", True
            return "Przepraszam, nie zrozumiałem. Czy możesz powiedzieć inaczej?", False

# ============================================
# INTEGRACJA Z LIVECHAT (webhook + Web API)
# ============================================

class LiveChatBotIntegration:
    """Integracja bota z LiveChat oparta na webhookach i Web API"""
    
    def __init__(self, your_bot: YourBot):
        self.bot = your_bot
        self.bot_agent_id: Optional[str] = None
        self.bot_secret: Optional[str] = None
        self.bot_token: Optional[str] = None
        self.target_group_id = int(os.getenv("LIVECHAT_TARGET_GROUP_ID", "0"))
        self._running = False
    
    async def start(self):
        """Inicjalizuje bota (tworzy go i generuje token)"""
        if self._running:
            return
        
        try:
            # 1. Utwórz bota (jeśli nie istnieje)
            self.bot_agent_id, self.bot_secret = await api_client.create_bot(name="ChatbotBMW")
            
            # 2. Wygeneruj token dla bota
            self.bot_token = await api_client.issue_bot_token(self.bot_agent_id, self.bot_secret)
            print(f"🔑 Token bota: {self.bot_token[:20]}...")
            
            self._running = True
            logger.info("✅ Bot LiveChat zainicjalizowany (gotowy do odbioru webhooków)")
            print("🌍 Serwer nasłuchuje na http://localhost:8000/webhook")
            
        except Exception as e:
            logger.error(f"Błąd inicjalizacji: {e}")
            raise
    
    async def handle_webhook(self, payload: dict) -> None:
        """Główna funkcja przetwarzająca webhooki"""
        try:
            action = payload.get("action")
            print(f"\n📨 Webhook action: {action}")
            
            if action == "incoming_event":
                await self._handle_incoming_event(payload)
            elif action == "incoming_chat":
                print("🆕 Nowy czat rozpoczęty")
            else:
                print(f"⏭️ Pomijam nieobsługiwane zdarzenie: {action}")
                
        except Exception as e:
            logger.error(f"Błąd przetwarzania webhooka: {e}")
    
    async def _handle_incoming_event(self, payload: dict):
        """Obsługuje zdarzenie nowej wiadomości"""
        try:
            event = payload.get("payload", {}).get("event", {})
            chat_id = payload.get("payload", {}).get("chat_id")
            
            if event.get("type") != "message":
                return
            
            text = event.get("text", "")
            author_id = event.get("author_id")
            
            # Ignoruj wiadomości od samego bota
            if author_id == self.bot_agent_id:
                return
            
            print(f"💬 Wiadomość od {author_id}: {text[:50]}...")
            
            # Przetwórz przez logikę bota
            response, should_transfer = await self.bot.process_message(text, chat_id)
            
            if should_transfer:
                # Wyślij wiadomość o transferze
                await api_client.send_message_as_bot(chat_id, response, self.bot_agent_id)
                # Wykonaj transfer
                await api_client.transfer_chat(chat_id, self.target_group_id, self.bot_agent_id)
            else:
                # Wyślij odpowiedź bota
                await api_client.send_message_as_bot(chat_id, response, self.bot_agent_id)
                
        except Exception as e:
            logger.error(f"Błąd w _handle_incoming_event: {e}")
    
    async def stop(self):
        """Zatrzymuje bota"""
        self._running = False
        logger.info("Bot zatrzymany")
    
    def is_running(self) -> bool:
        return self._running
    
    def get_bot_agent_id(self) -> Optional[str]:
        return self.bot_agent_id

# ============================================
# GŁÓWNA FUNKCJA
# ============================================

async def main():
    """Uruchamia bota w trybie standalone"""
    logger.info("🚀 Uruchamianie bota...")
    
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

# ============================================
# FASTAPI APLIKACJA DLA PANELU I WEBHOOKÓW
# ============================================

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

# ============================================
# ENDPOINT DO ODBIORU TOKENA Z IMPLICIT GRANT
# ============================================

@app.get("/token-page", response_class=HTMLResponse)
async def token_page(request: Request):
    """Strona do odbioru tokena z Implicit Grant"""
    return templates.TemplateResponse("token-page.html", {"request": request})

# ============================================
# ENDPOINT DLA WEBHOOKÓW (kluczowy!)
# ============================================

@app.post("/webhook")
async def webhook_receiver(request: Request):
    """Odbiera webhooki z LiveChat"""
    try:
        payload = await request.json()
        print(f"\n{'='*60}")
        print(f"📨 Webhook received at {datetime.now().strftime('%H:%M:%S')}")
        print(json.dumps(payload, indent=2))
        
        # Uruchom przetwarzanie w tle (nie blokuje odpowiedzi)
        if bot_integration:
            asyncio.create_task(bot_integration.handle_webhook(payload))
        else:
            print("⚠️ Bot nie jest zainicjalizowany")
        
        # Natychmiastowa odpowiedź – wymagane przez LiveChat!
        return JSONResponse(content={"status": "ok"}, status_code=200)
        
    except Exception as e:
        print(f"❌ Błąd webhooka: {e}")
        return JSONResponse(content={"status": "error", "message": str(e)}, status_code=500)

# ============================================
# ENDPOINTY DLA PANELU STEROWANIA
# ============================================

@app.get("/")
async def home(request: Request):
    if auth_client.is_authenticated():
        return RedirectResponse(url="/panel")
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/panel")
async def panel(request: Request):
    return templates.TemplateResponse("panel.html", {"request": request})

@app.get("/api/bot/status")
async def get_status():
    global bot_integration
    return {
        "running": bot_integration is not None and bot_integration.is_running(),
        "authenticated": True,
        "bot_agent_id": bot_integration.get_bot_agent_id() if bot_integration else None
    }

@app.post("/api/bot/start")
async def start_bot():
    global bot_integration
    if bot_integration and bot_integration.is_running():
        return {"success": True, "message": "Bot już działa"}
    
    your_bot = YourBot()
    bot_integration = LiveChatBotIntegration(your_bot)
    asyncio.create_task(bot_integration.start())
    await asyncio.sleep(1)
    return {"success": True, "message": "Bot uruchomiony", "details": {"bot_agent_id": bot_integration.get_bot_agent_id()}}

@app.post("/api/bot/stop")
async def stop_bot():
    global bot_integration
    if bot_integration:
        await bot_integration.stop()
        bot_integration = None
    return {"success": True, "message": "Bot zatrzymany"}

# ============================================
# ENDPOINT DO REJESTRACJI WEBHOOKA (opcjonalny)
# ============================================

@app.post("/register-webhook")
async def register_webhook():
    """Rejestruje webhook w LiveChat (użyj tylko raz)"""
    try:
        token = await auth_client.get_valid_token()
        ngrok_url = "https://crinklier-ruddily-leonore.ngrok-free.dev"  # Twój URL z ngrok
        
        payload = {
            "url": f"{ngrok_url}/webhook",
            "action": "incoming_event",
            "description": "Bot webhook",
            "type": "bot",
            "owner_client_id": auth_client.client_id
        }
        
        async with httpx.AsyncClient() as client:
            response = await client.post(
                "https://api.livechatinc.com/v3.6/configuration/action/register_webhook",
                headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
                json=payload
            )
            return response.json()
    except Exception as e:
        return {"error": str(e)}

@app.on_event("startup")
async def startup():
    print("\n🚀 Serwer na http://localhost:8000")
    print("🌍 Webhook URL: http://localhost:8000/webhook (użyj ngrok dla publicznego dostępu)")
    print("🔑 Endpoint do odbioru tokena: http://localhost:8000/token-page")
    print("📝 Aby zarejestrować webhook, wyślij POST na /register-webhook lub zrób to ręcznie w konsoli")

@app.on_event("shutdown")
async def shutdown():
    global bot_integration
    if bot_integration:
        await bot_integration.stop()
    await auth_client.close()
    await api_client.close()

if __name__ == "__main__":
    asyncio.run(main())