"""
Główny plik aplikacji BMW Assistant - ZK Motors Edition.
Z pełną integracją RAG, PromptService i pamięcią konwersacji.
"""
import json
import time
from datetime import datetime
from typing import Optional, Dict, Any, List
from pathlib import Path

from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse, HTMLResponse
from fastapi.security import HTTPBearer
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, validator
import uvicorn

from app.core.config import settings, validate_configuration
from app.core.exceptions import BMWAssistantException
from app.utils.logger import setup_logger, log
from app.services.rag_service import RAGService, get_rag_service
from app.services.llm_service import LLMService, get_llm_service
from app.services.prompt_service import PromptService, get_prompt_service
from app.services.cache import init_cache

# ============================================
# 🎯 INITIALIZATION
# ============================================

logger = setup_logger(__name__)
security = HTTPBearer(auto_error=False)

# Ścieżki do plików
BASE_DIR = Path(__file__).parent.absolute()
TEMPLATES_DIR = BASE_DIR / "templates"
STATIC_DIR = BASE_DIR / "static"

# Prosta pamięć konwersacji (w pamięci RAM)
conversation_memory: Dict[str, List[Dict]] = {}
MAX_HISTORY = 10

# ============================================
# 📦 MODELS
# ============================================

class ChatRequest(BaseModel):
    """Request model dla endpointu chat"""
    message: str = Field(..., min_length=1, max_length=1000)
    session_id: Optional[str] = Field(default="default", description="ID sesji dla pamięci konwersacji")
    stream: bool = Field(default=False, description="Czy streamować odpowiedź")
    temperature: float = Field(
        default=0.7,
        ge=0.0, 
        le=1.0,
        description="Kreatywność odpowiedzi (0.0 - faktualna, 1.0 - kreatywna)"
    )
    language: str = Field(
        default="pl",
        pattern="^(pl|en|de)$",
        description="Język odpowiedzi (pl, en, de)"
    )
    
    @validator('message')
    def message_not_empty(cls, v):
        if not v.strip():
            raise ValueError('Wiadomość nie może być pusta')
        return v.strip()


class ChatResponse(BaseModel):
    """Response model dla endpointu chat"""
    answer: str
    success: bool = Field(default=True, description="Czy odpowiedź się udała")
    session_id: str = Field(..., description="ID sesji")
    history_length: int = Field(..., description="Długość historii konwersacji")
    processing_time: float = Field(..., description="Czas przetwarzania w sekundach")
    sources: list[Dict[str, Any]] = Field(
        default=[],
        description="Źródła użyte do wygenerowania odpowiedzi"
    )
    model_used: str = Field(default="", description="Model użyty do generacji")
    tokens_used: Optional[Dict[str, int]] = Field(default=None, description="Użyte tokeny")
    confidence: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Pewność odpowiedzi (średnie podobieństwo dokumentów)"
    )


class HealthResponse(BaseModel):
    """Model odpowiedzi health check"""
    status: str
    timestamp: str
    version: str
    environment: str
    services: Dict[str, str]
    memory: Dict[str, Any]


class ResetResponse(BaseModel):
    """Model odpowiedzi reset"""
    success: bool
    message: str
    session_id: str
    history_length: int


class HistoryResponse(BaseModel):
    """Model odpowiedzi historii"""
    session_id: str
    history: List[Dict[str, Any]]
    total_messages: int
    limit: int


# ============================================
# 🧠 FUNKCJE PAMIĘCI KONWERSACJI
# ============================================

def get_conversation_history(session_id: str) -> List[Dict[str, Any]]:
    """Pobiera historię konwersacji dla sesji"""
    if session_id not in conversation_memory:
        conversation_memory[session_id] = []
    return conversation_memory[session_id]


def add_to_history(session_id: str, role: str, message: str):
    """Dodaje wiadomość do historii"""
    history = get_conversation_history(session_id)
    history.append({
        "role": role,
        "message": message,
        "timestamp": datetime.now().isoformat()
    })
    
    # Ogranicz historię do MAX_HISTORY
    if len(history) > MAX_HISTORY:
        conversation_memory[session_id] = history[-MAX_HISTORY:]


def format_history_for_prompt(history: List[Dict]) -> List[Dict[str, str]]:
    """Formatuje historię na format dla PromptService"""
    formatted = []
    for msg in history[-6:]:  # Ostatnie 6 wiadomości
        formatted.append({
            "role": msg["role"],
            "content": msg["message"]
        })
    return formatted


# ============================================
# 🚀 FASTAPI APPLICATION
# ============================================

app = FastAPI(
    title=settings.APP_NAME,
    description="Asystent klienta ZK Motors z RAG i Cohere LLM. Oficjalny dealer BMW i MINI.",
    version=settings.APP_VERSION,
    docs_url="/docs" if not settings.IS_PRODUCTION else None,
    redoc_url="/redoc" if not settings.IS_PRODUCTION else None,
    openapi_url="/openapi.json" if not settings.IS_PRODUCTION else None,
)

# ============================================
# 📁 STATIC FILES & TEMPLATES
# ============================================

# Static files
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# ============================================
# 🔧 MIDDLEWARE
# ============================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS_STR,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response

# ============================================
# 🏠 BASIC ENDPOINTS
# ============================================

@app.get("/", response_class=HTMLResponse)
async def root():
    """Strona główna z czatem"""
    chat_html_path = TEMPLATES_DIR / "chat.html"
    
    if not chat_html_path.exists():
        # Fallback jeśli nie ma chat.html
        return HTMLResponse(f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>{settings.APP_NAME}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                h1 {{ color: #0066b3; }}
                .info {{ background: #f0f8ff; padding: 20px; border-radius: 8px; }}
            </style>
        </head>
        <body>
            <h1>🚗 {settings.APP_NAME} v{settings.APP_VERSION}</h1>
            
            <div class="info">
                <h3>Asystent klienta ZK Motors</h3>
                <p>Oficjalny asystent dealera BMW i MINI zintegrowany z RAG i Cohere LLM.</p>
                <p><strong>Pamięć konwersacji:</strong> Ostatnie {MAX_HISTORY} wiadomości</p>
            </div>
            
            <h2>🔧 Endpointy API</h2>
            <ul>
                <li><code>POST /chat</code> - Główny endpoint chat</li>
                <li><code>POST /chat/reset</code> - Reset historii</li>
                <li><code>GET /chat/history</code> - Pobierz historię</li>
                <li><code>GET /health</code> - Status serwisu</li>
                <li><code>GET /models</code> - Lista modeli</li>
            </ul>
            
            <p><a href="/docs">📚 Dokumentacja API (Swagger)</a></p>
        </body>
        </html>
        """)
    
    try:
        with open(chat_html_path, 'r', encoding='utf-8') as f:
            html_content = f.read()
        
        return HTMLResponse(content=html_content)
        
    except Exception as e:
        return HTMLResponse(f"""
        <!DOCTYPE html>
        <html>
        <head><title>Error</title></head>
        <body>
            <h1>Error loading chat interface</h1>
            <pre>{str(e)}</pre>
        </body>
        </html>
        """)


@app.get("/health", response_model=HealthResponse)
async def health_check(
    rag_service: RAGService = Depends(get_rag_service),
    llm_service: LLMService = Depends(get_llm_service)
):
    """Health check - sprawdza status wszystkich komponentów"""
    try:
        # Sprawdź RAG
        rag_health = await rag_service.health_check()
        
        # Sprawdź LLM
        llm_health = await llm_service.health_check()
        
        services_status = {
            "rag_system": rag_health.get("status", "unknown"),
            "llm_service": llm_health.get("status", "unknown"),
            "api": "healthy",
            "cache": "connected" if settings.REDIS_URL else "in_memory",
            "memory": "enabled"
        }
        
        overall_status = "healthy" if all(
            s == "healthy" or s == "connected" or s == "operational"
            for s in services_status.values()
        ) else "degraded"
        
        memory_stats = {
            "active_sessions": len(conversation_memory),
            "total_messages": sum(len(h) for h in conversation_memory.values()),
            "max_history": MAX_HISTORY
        }
        
        return HealthResponse(
            status=overall_status,
            timestamp=datetime.utcnow().isoformat(),
            version=settings.APP_VERSION,
            environment=settings.ENVIRONMENT,
            services=services_status,
            memory=memory_stats
        )
        
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return HealthResponse(
            status="degraded",
            timestamp=datetime.utcnow().isoformat(),
            version=settings.APP_VERSION,
            environment=settings.ENVIRONMENT,
            services={"error": str(e)},
            memory={"error": "memory check failed"}
        )


@app.get("/models")
async def list_models():
    """Lista dostępnych modeli LLM"""
    models = [
        {
            "id": "command-r",
            "name": "Command R",
            "provider": "Cohere",
            "max_tokens": 128000,
            "context_length": 128000,
            "description": "Flagaowy model do konwersacji"
        },
        {
            "id": "command-r-plus", 
            "name": "Command R+",
            "provider": "Cohere",
            "max_tokens": 128000,
            "context_length": 128000,
            "description": "Zaawansowany model z lepszym rozumieniem"
        },
        {
            "id": "command",
            "name": "Command",
            "provider": "Cohere",
            "max_tokens": 4096,
            "context_length": 4096,
            "description": "Model zoptymalizowany pod wykonywanie poleceń"
        }
    ]
    
    return {
        "models": models,
        "default_model": settings.COHERE_CHAT_MODEL,
        "embedding_model": settings.COHERE_EMBED_MODEL,
        "memory_enabled": True,
        "max_history": MAX_HISTORY
    }

# ============================================
# 💬 CHAT ENDPOINT (GŁÓWNY)
# ============================================

@app.post("/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    background_tasks: BackgroundTasks,
    rag_service: RAGService = Depends(get_rag_service),
    llm_service: LLMService = Depends(get_llm_service),
    prompt_service: PromptService = Depends(get_prompt_service)
):
    """
    Główny endpoint chat z pamięcią konwersacji.
    
    Proces:
    1. Pobierz historię konwersacji
    2. Pobierz kontekst z RAG na podstawie pytania
    3. Zbuduj prompt z kontekstem i historią
    4. Wygeneruj odpowiedź za pomocą Cohere LLM
    5. Zapisz w pamięci i zwróć odpowiedź
    """
    start_time = time.time()
    
    try:
        session_id = request.session_id
        logger.info(f"Chat request from session {session_id}: {request.message[:50]}...")
        
        # 1. Pobierz historię konwersacji
        history = get_conversation_history(session_id)
        conversation_history = format_history_for_prompt(history)
        
        # 2. Wyszukaj kontekst w RAG
        logger.debug("🔍 Retrieving context from RAG...")
        context_result = await rag_service.retrieve(
            query=request.message,
            top_k=settings.TOP_K_DOCUMENTS,
            similarity_threshold=settings.SIMILARITY_THRESHOLD
        )
        
        if not context_result.documents:
            logger.warning(f"No relevant documents found for: {request.message}")
        
        # 3. Przygotuj prompt
        logger.debug("📝 Building prompt with history...")
        prompt = prompt_service.build_chat_prompt(
            user_message=request.message,
            context_documents=context_result.documents,
            conversation_history=conversation_history,
            language=request.language,
            temperature=request.temperature,
            user_id=session_id
        )
        
        # 4. Generuj odpowiedź za pomocą LLM
        logger.debug(f"🧠 Generating response with {settings.COHERE_CHAT_MODEL}...")
        
        if request.stream:
            # Streaming not implemented
            raise HTTPException(
                status_code=status.HTTP_501_NOT_IMPLEMENTED,
                detail="Streaming not available in current version"
            )
        
        llm_response = await llm_service.generate(
            prompt=prompt,
            model=settings.COHERE_CHAT_MODEL,
            temperature=request.temperature,
            max_tokens=settings.MAX_TOKENS
        )
        
        # 5. Przygotuj odpowiedź
        processing_time = time.time() - start_time
        
        # 6. Dodaj do historii
        add_to_history(session_id, "user", request.message)
        add_to_history(session_id, "assistant", llm_response.text)
        
        response = ChatResponse(
            answer=llm_response.text,
            session_id=session_id,
            history_length=len(get_conversation_history(session_id)),
            processing_time=processing_time,
            sources=context_result.to_api_response().get("sources", []),
            model_used=settings.COHERE_CHAT_MODEL,
            tokens_used=llm_response.tokens_used,
            confidence=context_result.average_similarity
        )
        
        # 7. Logowanie w tle (opcjonalne)
        background_tasks.add_task(
            log_interaction,
            user_message=request.message,
            assistant_response=llm_response.text,
            session_id=session_id,
            sources_count=len(response.sources),
            tokens_used=response.tokens_used,
            processing_time=processing_time,
            confidence=response.confidence
        )
        
        logger.info(f"✅ Response generated in {processing_time:.2f}s for session {session_id}", extra={
            "tokens": response.tokens_used,
            "sources": len(response.sources),
            "history_length": response.history_length
        })
        
        return response
        
    except Exception as e:
        logger.error(f"❌ Chat endpoint error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error: {str(e)}"
        )


@app.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    """Placeholder dla streaming - nie zaimplementowane"""
    async def event_generator():
        yield f"data: {json.dumps({'error': 'Streaming not implemented in current version'})}\n\n"
        yield "data: [DONE]\n\n"
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache"}
    )


@app.post("/chat/reset", response_model=ResetResponse)
async def reset_chat(session_id: str = "default"):
    """Resetuje historię konwersacji dla danej sesji"""
    if session_id in conversation_memory:
        conversation_memory[session_id] = []
        logger.info(f"Reset conversation history for session: {session_id}")
    
    return ResetResponse(
        success=True,
        message=f"Session {session_id} reset successfully",
        session_id=session_id,
        history_length=len(get_conversation_history(session_id))
    )


@app.get("/chat/history", response_model=HistoryResponse)
async def get_history(session_id: str = "default", limit: int = 10):
    """Pobiera historię konwersacji"""
    history = get_conversation_history(session_id)
    
    return HistoryResponse(
        session_id=session_id,
        history=history[-limit:],
        total_messages=len(history),
        limit=limit
    )

# ============================================
# 🔍 DEBUG ENDPOINTS (tylko development)
# ============================================

@app.get("/debug/rag")
async def debug_rag(
    query: str,
    top_k: int = settings.TOP_K_DOCUMENTS,
    rag_service: RAGService = Depends(get_rag_service)
):
    """Debug endpoint do testowania RAG (tylko development)"""
    if settings.IS_PRODUCTION:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Debug endpoints disabled in production"
        )
    
    result = await rag_service.retrieve(query=query, top_k=top_k)
    
    return {
        "query": query,
        "result": result.to_api_response(),
        "documents_count": len(result.documents),
        "average_similarity": result.average_similarity
    }


@app.get("/debug/stats")
async def debug_stats(
    rag_service: RAGService = Depends(get_rag_service),
    llm_service: LLMService = Depends(get_llm_service)
):
    """Statystyki serwisów (tylko development)"""
    if settings.IS_PRODUCTION:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Debug endpoints disabled in production"
        )
    
    rag_stats = await rag_service.get_stats()
    llm_stats = await llm_service.get_stats()
    
    memory_stats = {
        "active_sessions": len(conversation_memory),
        "total_messages": sum(len(h) for h in conversation_memory.values()),
        "sessions": list(conversation_memory.keys())[:5]
    }
    
    return {
        "rag_service": rag_stats,
        "llm_service": llm_stats,
        "memory": memory_stats,
        "timestamp": datetime.utcnow().isoformat()
    }

# ============================================
# 🛠️ UTILITY FUNCTIONS
# ============================================

async def log_interaction(
    user_message: str,
    assistant_response: str,
    session_id: str,
    sources_count: int,
    tokens_used: Dict[str, int],
    processing_time: float,
    confidence: Optional[float]
):
    """Loguje interakcję w tle"""
    try:
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "session_id": session_id,
            "user_message_preview": user_message[:100],
            "assistant_response_preview": assistant_response[:100],
            "sources_count": sources_count,
            "tokens_used": tokens_used,
            "processing_time": processing_time,
            "confidence": confidence,
            "environment": settings.ENVIRONMENT
        }
        
        logger.info(f"Interaction logged", extra=log_entry)
        
    except Exception as e:
        logger.error(f"Failed to log interaction: {str(e)}")

# ============================================
# ⚡ APPLICATION EVENTS
# ============================================

@app.on_event("startup")
async def startup_event():
    """Uruchamiane przy starcie aplikacji"""
    try:
        # Walidacja konfiguracji
        validate_configuration()
        
        # Inicjalizacja cache
        await init_cache()
        
        # Informacje o starcie
        logger.info(f"🚀 {settings.APP_NAME} v{settings.APP_VERSION} starting up...")
        logger.info(f"🌍 Environment: {settings.ENVIRONMENT}")
        logger.info(f"🧠 LLM Model: {settings.COHERE_CHAT_MODEL}")
        logger.info(f"🔍 RAG: {settings.TOP_K_DOCUMENTS} docs, threshold: {settings.SIMILARITY_THRESHOLD}")
        logger.info(f"💾 Memory: Enabled (last {MAX_HISTORY} messages per session)")
        logger.info("✅ Application started successfully")
        
        if settings.IS_DEVELOPMENT:
            logger.warning("⚡ Running in DEVELOPMENT mode")
        
    except Exception as e:
        logger.critical(f"❌ Startup failed: {str(e)}", exc_info=True)
        raise


@app.on_event("shutdown")
async def shutdown_event():
    """Uruchamiane przy zamykaniu aplikacji"""
    logger.info("🛑 Shutting down application...")
    logger.info(f"📊 Memory stats: {len(conversation_memory)} sessions, {sum(len(h) for h in conversation_memory.values())} total messages")

# ============================================
# 🚨 ERROR HANDLERS
# ============================================

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handler dla HTTPException"""
    logger.warning(
        f"HTTP {exc.status_code}: {exc.detail}",
        extra={"path": request.url.path, "method": request.method}
    )
    
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail},
    )


@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    """Handler dla nieprzewidzianych wyjątków"""
    logger.critical(
        f"Unhandled exception: {str(exc)}",
        extra={"path": request.url.path, "method": request.method},
        exc_info=True
    )
    
    detail = "Internal server error"
    if settings.IS_DEVELOPMENT:
        detail = f"{exc.__class__.__name__}: {str(exc)}"
    
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": detail},
    )

# ============================================
# 🎯 MAIN ENTRY POINT
# ============================================

def main():
    """Główna funkcja uruchamiająca aplikację"""
    import sys
    
    try:
        validate_configuration()
        
        config = {
            "app": "app.main:app",
            "host": settings.HOST,
            "port": settings.PORT,
            "reload": settings.RELOAD,
            "log_level": settings.LOG_LEVEL.lower(),
        }
        
        if settings.IS_DEVELOPMENT:
            print(f"\n{'='*60}")
            print(f"🚗 {settings.APP_NAME} v{settings.APP_VERSION}")
            print(f"🌍 Environment: {settings.ENVIRONMENT}")
            print(f"🔗 Czat: http://{settings.HOST}:{settings.PORT}/")
            print(f"📚 Docs: http://{settings.HOST}:{settings.PORT}/docs")
            print(f"🧠 Model: {settings.COHERE_CHAT_MODEL}")
            print(f"💾 Memory: {MAX_HISTORY} messages per session")
            print(f"🔗 API: http://{settings.HOST}:{settings.PORT}/chat")
            print(f"{'='*60}\n")
        
        uvicorn.run(**config)
        
    except Exception as e:
        print(f"❌ Failed to start: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()