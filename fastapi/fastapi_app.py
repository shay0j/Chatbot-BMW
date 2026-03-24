# api/fastapi_app.py
import os
import secrets
from typing import Optional

from fastapi import FastAPI, Request, Response, HTTPException, Depends
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.sessions import SessionMiddleware
from pydantic import BaseModel
import httpx

from livechat_integration.auth import auth_client
from livechat_integration.api_client import api_client

# Import Twojego bota
from main import YourBot, LiveChatBotIntegration

# Inicjalizacja FastAPI
app = FastAPI(title="Bot LiveChat", version="1.0.0")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Session (do przechowywania stanu OAuth)
app.add_middleware(
    SessionMiddleware,
    secret_key=os.getenv("SECRET_KEY", secrets.token_urlsafe(32))
)

# Templates
templates = Jinja2Templates(directory="api/templates")

# Globalna instancja bota
bot_integration: Optional[LiveChatBotIntegration] = None


# ============================================
# Modele odpowiedzi API
# ============================================

class StatusResponse(BaseModel):
    running: bool
    authenticated: bool
    bot_agent_id: Optional[str] = None


class StartResponse(BaseModel):
    success: bool
    message: str
    details: Optional[dict] = None


# ============================================
# Middleware autoryzacji
# ============================================

async def require_auth(request: Request):
    """Sprawdza czy użytkownik jest zalogowany"""
    if not auth_client.is_authenticated():
        raise HTTPException(status_code=401, detail="Unauthorized")
    return True


# ============================================
# Endpointy widoków
# ============================================

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Strona główna"""
    if auth_client.is_authenticated():
        return RedirectResponse(url="/panel")
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/login")
async def login(request: Request):
    """Inicjuje logowanie przez LiveChat"""
    state = auth_client.generate_state()
    request.session["oauth_state"] = state
    
    auth_url = auth_client.get_authorization_url(state)
    return RedirectResponse(url=auth_url)


@app.get("/callback")
async def callback(request: Request, code: str, state: str):
    """Callback OAuth"""
    stored_state = request.session.get("oauth_state")
    
    if not state or not stored_state or state != stored_state:
        raise HTTPException(status_code=400, detail="Invalid state")
    
    try:
        await auth_client.exchange_code(code)
        request.session.pop("oauth_state", None)
        return RedirectResponse(url="/panel")
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/panel", response_class=HTMLResponse)
async def panel(request: Request, _: bool = Depends(require_auth)):
    """Panel sterowania"""
    return templates.TemplateResponse("panel.html", {"request": request})


@app.get("/logout")
async def logout(request: Request):
    """Wylogowanie"""
    request.session.clear()
    # Tutaj możesz dodać unieważnienie tokena
    return RedirectResponse(url="/")


# ============================================
# API dla panelu
# ============================================

@app.get("/api/status", response_model=StatusResponse)
async def get_status(_: bool = Depends(require_auth)):
    """Status bota"""
    global bot_integration
    
    return StatusResponse(
        running=bot_integration is not None and bot_integration.is_running(),
        authenticated=auth_client.is_authenticated(),
        bot_agent_id=bot_integration.get_bot_agent_id() if bot_integration else None
    )


@app.post("/api/start", response_model=StartResponse)
async def start_bot(_: bool = Depends(require_auth)):
    """Uruchamia bota"""
    global bot_integration
    
    if bot_integration and bot_integration.is_running():
        return StartResponse(
            success=True,
            message="Bot już uruchomiony",
            details={"bot_agent_id": bot_integration.get_bot_agent_id()}
        )
    
    try:
        # Utwórz Twojego bota
        your_bot = YourBot()
        
        # Utwórz integrację
        bot_integration = LiveChatBotIntegration(your_bot)
        
        # Uruchom asynchronicznie
        import asyncio
        asyncio.create_task(bot_integration.start())
        
        # Daj mu chwilę na start
        await asyncio.sleep(1)
        
        return StartResponse(
            success=True,
            message="Bot uruchomiony",
            details={"bot_agent_id": bot_integration.get_bot_agent_id()}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/stop")
async def stop_bot(_: bool = Depends(require_auth)):
    """Zatrzymuje bota"""
    global bot_integration
    
    if bot_integration:
        await bot_integration.stop()
        bot_integration = None
    
    return {"success": True, "message": "Bot zatrzymany"}


# ============================================
# Zdarzenia startu/zamknięcia
# ============================================

@app.on_event("startup")
async def startup():
    """Inicjalizacja przy starcie"""
    print("FastAPI server uruchomiony")


@app.on_event("shutdown")
async def shutdown():
    """Sprzątanie przy zamknięciu"""
    global bot_integration
    
    if bot_integration:
        await bot_integration.stop()
    
    await auth_client.close()
    await api_client.close()
    print("Serwer zamknięty")