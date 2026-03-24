# livechat_integration/websocket_client.py
import asyncio
import json
from typing import Callable, Optional, Dict, Any
from dataclasses import dataclass

import websockets
from websockets import WebSocketClientProtocol
from loguru import logger

from .auth import auth_client
from .models import IncomingEvent, MessageEvent, TransferChat, TransferTarget

@dataclass
class MessageHandler:
    """Handler dla wiadomości"""
    chat_id: str
    message: str
    author_id: str

class LiveChatWebSocketClient:
    """Klient WebSocket dla LiveChat RTM API"""
    
    def __init__(self, bot_agent_id: str):
        self.bot_agent_id = bot_agent_id
        self.ws: Optional[WebSocketClientProtocol] = None
        self.running = False
        self._ping_task: Optional[asyncio.Task] = None
        self._receive_task: Optional[asyncio.Task] = None
        
        # Handlery
        self.message_handlers: list[Callable[[MessageHandler], Any]] = []
    
    async def connect(self):
        """Łączy z LiveChat RTM"""
        token = await auth_client.get_valid_token()
        
        # Wybierz URL w zależności od regionu
        region = auth_client.region
        if region == "eu":
            url = "wss://api-fra.livechatinc.com/v3.5/agent/rtm/ws"
        else:
            url = "wss://api.livechatinc.com/v3.5/agent/rtm/ws"
        
        logger.info(f"Łączenie z {url}")
        self.ws = await websockets.connect(url)
        
        # Logowanie
        await self.ws.send(json.dumps({
            "action": "login",
            "payload": {"token": token}
        }))
        
        # Poczekaj na potwierdzenie
        response = await self.ws.recv()
        logger.info(f"Połączono: {response}")
        
        self.running = True
        
        # Uruchom zadania
        self._ping_task = asyncio.create_task(self._ping_loop())
        self._receive_task = asyncio.create_task(self._receive_loop())
    
    async def _ping_loop(self):
        """Wysyła ping co 15 sekund"""
        while self.running and self.ws:
            try:
                await self.ws.send(json.dumps({"action": "ping"}))
                await asyncio.sleep(15)
            except Exception as e:
                logger.error(f"Błąd ping: {e}")
                break
    
    async def _receive_loop(self):
        """Odbiera wiadomości z WebSocket"""
        while self.running and self.ws:
            try:
                message = await self.ws.recv()
                await self._handle_message(message)
            except websockets.ConnectionClosed:
                logger.warning("Połączenie zamknięte")
                break
            except Exception as e:
                logger.error(f"Błąd odbioru: {e}")
                break
    
    async def _handle_message(self, message: str):
        """Przetwarza przychodzącą wiadomość"""
        try:
            data = json.loads(message)
            
            # Loguj wszystko (debug)
            logger.debug(f"Otrzymano: {data}")
            
            # Tylko incoming_event nas interesuje
            if data.get("action") != "incoming_event":
                return
            
            event = data["payload"]["event"]
            chat_id = data["payload"]["chat_id"]
            author_id = event.get("author_id")
            
            # Ignoruj własne wiadomości
            if author_id == self.bot_agent_id:
                return
            
            # Tylko wiadomości tekstowe
            if event.get("type") == "message":
                text = event.get("text", "")
                
                # Wywołaj wszystkie zarejestrowane handlery
                handler = MessageHandler(
                    chat_id=chat_id,
                    message=text,
                    author_id=author_id
                )
                
                for callback in self.message_handlers:
                    # Uruchom asynchronicznie (nie blokuj)
                    asyncio.create_task(self._safe_callback(callback, handler))
        
        except Exception as e:
            logger.error(f"Błąd parsowania: {e}")
    
    async def _safe_callback(self, callback, handler):
        """Bezpiecznie wywołuje callback"""
        try:
            if asyncio.iscoroutinefunction(callback):
                await callback(handler)
            else:
                callback(handler)
        except Exception as e:
            logger.error(f"Błąd w callback: {e}")
    
    async def send_message(self, chat_id: str, text: str):
        """Wysyła wiadomość jako bot"""
        if not self.ws or not self.running:
            raise Exception("WebSocket nie jest połączony")
        
        message = {
            "action": "send_event",
            "payload": {
                "chat_id": chat_id,
                "event": {
                    "type": "message",
                    "text": text
                }
            },
            "author_id": self.bot_agent_id
        }
        
        await self.ws.send(json.dumps(message))
        logger.info(f"📤 Wysłano do {chat_id}: {text[:50]}...")
    
    async def transfer_chat(self, chat_id: str, target_group_id: int = 0):
        """Przekazuje czat do grupy agentów"""
        if not self.ws or not self.running:
            raise Exception("WebSocket nie jest połączony")
        
        message = {
            "action": "transfer_chat",
            "payload": {
                "id": chat_id,
                "target": {
                    "type": "group",
                    "ids": [target_group_id]
                },
                "ignore_agents_availability": False
            },
            "author_id": self.bot_agent_id
        }
        
        await self.ws.send(json.dumps(message))
        logger.info(f"🔄 Przekazano czat {chat_id} do grupy {target_group_id}")
    
    def on_message(self, callback):
        """Dekorator do rejestracji handlera wiadomości"""
        self.message_handlers.append(callback)
        return callback
    
    async def close(self):
        """Zamyka połączenie"""
        self.running = False
        
        if self._ping_task:
            self._ping_task.cancel()
        if self._receive_task:
            self._receive_task.cancel()
        
        if self.ws:
            await self.ws.close()
        
        logger.info("WebSocket zamknięty")