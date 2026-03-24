# livechat_integration/models.py
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any, Literal
from datetime import datetime

class TokenResponse(BaseModel):
    """Odpowiedź z tokenem OAuth"""
    access_token: str
    refresh_token: Optional[str] = None
    expires_in: int
    token_type: str = "bearer"

class BotAgentCreate(BaseModel):
    """Model tworzenia agenta-bota"""
    name: str = "Mój Własny Bot"
    status: Literal["accepting chats", "not accepting chats", "offline"] = "not accepting chats"
    avatar_path: Optional[str] = None
    default_group_priority: Optional[str] = None

class BotAgentResponse(BaseModel):
    """Odpowiedź po utworzeniu agenta-bota"""
    bot_agent_id: str

class IncomingEvent(BaseModel):
    """Zdarzenie przychodzące (wiadomość)"""
    chat_id: str
    event: Dict[str, Any]
    author_id: Optional[str] = None

class MessageEvent(BaseModel):
    """Wiadomość tekstowa"""
    type: Literal["message"] = "message"
    text: str
    author_id: Optional[str] = None
    created_at: Optional[datetime] = None

class TransferTarget(BaseModel):
    """Cel transferu czatu"""
    type: Literal["group", "agent", "department"]
    ids: List[int]

class TransferChat(BaseModel):
    """Akcja transferu czatu"""
    id: str
    target: TransferTarget
    ignore_agents_availability: bool = False