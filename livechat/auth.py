# livechat_integration/auth.py
import os
import secrets
import time
from typing import Optional, Dict
from urllib.parse import urlencode

import httpx
from jose import jwt
from dotenv import load_dotenv
from loguru import logger

from .models import TokenResponse

load_dotenv()

class LiveChatAuth:
    """Klient autoryzacji OAuth 2.0 dla LiveChat"""
    
    def __init__(self):
        self.client_id = os.getenv("LIVECHAT_CLIENT_ID")
        self.client_secret = os.getenv("LIVECHAT_CLIENT_SECRET")
        self.redirect_uri = os.getenv("LIVECHAT_REDIRECT_URI")
        self.region = os.getenv("LIVECHAT_REGION", "us")
        
        # URL-e zależne od regionu
        self.accounts_url = "https://accounts.livechatinc.com"
        self.api_url = "https://api.livechatinc.com"
        
        if self.region == "eu":
            self.rtm_url = "wss://api-fra.livechatinc.com/v3.5/agent/rtm/ws"
        else:
            self.rtm_url = "wss://api.livechatinc.com/v3.5/agent/rtm/ws"
        
        # Scopes (uprawnienia)
        self.scopes = "chats--all:rw agents-bot--all:rw agents--all:r"
        
        # Cache tokena
        self._token: Optional[Dict] = None
        self._token_expires_at: Optional[float] = None
        
        # Klient HTTP
        self.http_client = httpx.AsyncClient(timeout=30.0)
    
    def get_authorization_url(self, state: str) -> str:
        """Generuje URL do autoryzacji LiveChat"""
        params = {
            "response_type": "code",
            "client_id": self.client_id,
            "redirect_uri": self.redirect_uri,
            "scope": self.scopes,
            "state": state,
            "access_type": "online"
        }
        return f"{self.accounts_url}/?{urlencode(params)}"
    
    def generate_state(self) -> str:
        """Generuje stan do ochrony CSRF"""
        return secrets.token_urlsafe(32)
    
    async def exchange_code(self, code: str) -> TokenResponse:
        """Wymienia kod autoryzacyjny na token"""
        data = {
            "grant_type": "authorization_code",
            "client_id": self.client_id,
            "client_secret": self.client_secret,
            "code": code,
            "redirect_uri": self.redirect_uri
        }
        
        response = await self.http_client.post(
            f"{self.accounts_url}/token",
            data=data
        )
        response.raise_for_status()
        
        token_data = response.json()
        self._save_token(token_data)
        
        return TokenResponse(**token_data)
    
    async def refresh_token(self, refresh_token: str) -> TokenResponse:
        """Odświeża token dostępu"""
        data = {
            "grant_type": "refresh_token",
            "client_id": self.client_id,
            "client_secret": self.client_secret,
            "refresh_token": refresh_token
        }
        
        response = await self.http_client.post(
            f"{self.accounts_url}/token",
            data=data
        )
        response.raise_for_status()
        
        token_data = response.json()
        self._save_token(token_data)
        
        return TokenResponse(**token_data)
    
    def _save_token(self, token_data: Dict):
        """Zapisuje token w cache"""
        self._token = token_data
        self._token_expires_at = time.time() + token_data["expires_in"]
        logger.info(f"Token zapisany, ważny do: {time.ctime(self._token_expires_at)}")
    
    async def get_valid_token(self) -> str:
        """Zwraca ważny token (odświeża jeśli potrzeba)"""
        # Jeśli token jest ważny (z 5-minutowym buforem)
        if self._token and self._token_expires_at and self._token_expires_at > time.time() + 300:
            return self._token["access_token"]
        
        # Próbuj odświeżyć
        if self._token and self._token.get("refresh_token"):
            try:
                await self.refresh_token(self._token["refresh_token"])
                return self._token["access_token"]
            except Exception as e:
                logger.error(f"Błąd odświeżania tokena: {e}")
                self._token = None
                self._token_expires_at = None
                raise Exception("Wymagana ponowna autoryzacja")
        
        raise Exception("Brak tokena - wykonaj autoryzację")
    
    def is_authenticated(self) -> bool:
        """Sprawdza czy mamy ważny token"""
        return bool(
            self._token and 
            self._token_expires_at and 
            self._token_expires_at > time.time()
        )
    
    async def close(self):
        """Zamyka klienta HTTP"""
        await self.http_client.aclose()

# Globalna instancja auth
auth_client = LiveChatAuth()