import os
import asyncio
import time
import json
import base64
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass
from datetime import datetime

import httpx
from loguru import logger
from dotenv import load_dotenv
from pydantic import BaseModel
import secrets
import traceback

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
                    return {"success": True, "event_id": event_id}
                except:
                    logger.info(f"✅ Wiadomość wysłana!")
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

class YourBot:
    
    def __init__(self):
        logger.info("Inicjalizacja bota...")
        self.conversation_state = {}
        logger.info("Bot gotowy")
    
    async def process_message(self, text: str, chat_id: str = None) -> tuple[str, bool]:
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
        
        if any(word in text_lower for word in ['cześć', 'witaj', 'hej', 'dzień dobry']):
            state["failed"] = 0
            return "Cześć! W czym mogę pomóc?", False
        elif any(word in text_lower for word in ['dziękuję', 'dzięki', 'thx']):
            state["failed"] = 0
            return "Proszę bardzo! Czy mogę pomóc w czymś jeszcze?", False
        elif any(word in text_lower for word in ['godziny', 'otwarcia', 'czynne']):
            state["failed"] = 0
            return "Jesteśmy czynni od poniedziałku do piątku w godzinach 9:00-17:00.", False
        elif any(word in text_lower for word in ['adres', 'gdzie', 'siedziba']):
            state["failed"] = 0
            return "Nasza siedziba znajduje się przy ul. Przykładowej 123 w Warszawie.", False
        else:
            state["failed"] = state.get("failed", 0) + 1
            if state["failed"] >= 3:
                return "Przepraszam, nie mogę pomóc. Łączę z konsultantem.", True
            return "Przepraszam, nie zrozumiałem. Czy możesz powiedzieć inaczej?", False

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
            
            print(f"\n🤖 Przetwarzanie wiadomości...")
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

@app.on_event("startup")
async def startup():
    print("\n" + "="*60)
    print("🚀 SERWER BOTA LIVECHAT")
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
    print("="*60)
    
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

if __name__ == "__main__":
    if not os.getenv("RUNNING_IN_UVICORN"):
        asyncio.run(main())