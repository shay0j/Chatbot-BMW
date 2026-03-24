# livechat_integration/api_client.py
import httpx
from loguru import logger
from typing import Optional, Dict, Any

from .auth import auth_client
from .models import BotAgentCreate, BotAgentResponse

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
        
        response = await self.http_client.request(
            method, url, headers=headers, **kwargs
        )
        response.raise_for_status()
        
        return response.json()
    
    async def create_bot_agent(self, name: str) -> str:
        """Tworzy agenta-bota"""
        data = BotAgentCreate(name=name).model_dump()
        
        result = await self._request(
            "POST",
            "/configuration/agents/create_bot_agent",
            json=data
        )
        
        bot_agent_id = result["bot_agent_id"]
        logger.info(f"✅ Bot agent utworzony: {bot_agent_id}")
        return bot_agent_id
    
    async def list_agents(self) -> list:
        """Lista agentów"""
        return await self._request("GET", "/configuration/agents/list_agents")
    
    async def check_agent_availability(self) -> bool:
        """Sprawdza czy jest dostępny agent ludzki"""
        agents = await self.list_agents()
        
        for agent in agents:
            if agent.get("type") == "agent" and agent.get("status") == "accepting chats":
                return True
        
        return False
    
    async def close(self):
        """Zamyka klienta"""
        await self.http_client.aclose()

# Globalna instancja
api_client = LiveChatAPIClient()