"""
BMW Assistant - Production Version with Improved RAG and Intent Detection
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
from app.utils.logger import setup_logger, log
from app.services.llm_service import LLMService, get_llm_service
from app.services.prompt_service import PromptService, get_prompt_service
from app.services.rag_service import get_rag_service, RAGService
from app.services.cache import init_cache

# ============================================
# INITIALIZATION
# ============================================

logger = setup_logger(__name__)
security = HTTPBearer(auto_error=False)

BASE_DIR = Path(__file__).parent.absolute()
TEMPLATES_DIR = BASE_DIR / "templates"
STATIC_DIR = BASE_DIR / "static"

conversation_memory: Dict[str, List[Dict]] = {}
MAX_HISTORY = 5

# ============================================
# MODELS
# ============================================

class ChatRequest(BaseModel):
    """Request model dla endpointu chat"""
    message: str = Field(..., min_length=1, max_length=1000)
    session_id: Optional[str] = Field(default="default", description="ID sesji")
    stream: bool = Field(default=False, description="Czy streamować odpowiedź")
    temperature: float = Field(
        default=0.7,
        ge=0.0, 
        le=1.0,
        description="Kreatywność odpowiedzi"
    )
    language: str = Field(
        default="pl",
        pattern="^(pl|en|de)$",
        description="Język odpowiedzi"
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
    sources: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Źródła użyte do wygenerowania odpowiedzi"
    )
    model_used: str = Field(default="", description="Model użyty do generacji")
    tokens_used: Optional[Dict[str, int]] = Field(default=None, description="Użyte tokeny")
    confidence: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Pewność odpowiedzi"
    )
    rag_info: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Informacje o RAG"
    )
    data_quality: Optional[str] = Field(
        default=None,
        description="Jakość danych: high/medium/low"
    )

class HealthResponse(BaseModel):
    """Model odpowiedzi health check"""
    status: str
    timestamp: str
    version: str
    environment: str
    services: Dict[str, str]
    memory: Dict[str, Any]
    rag_stats: Optional[Dict[str, Any]] = None

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
# HELPER FUNCTIONS
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
    
    if len(history) > MAX_HISTORY:
        conversation_memory[session_id] = history[-MAX_HISTORY:]

def format_history_for_prompt(history: List[Dict]) -> List[Dict[str, str]]:
    """Formatuje historię na format dla promptu"""
    formatted = []
    for msg in history[-4:]:
        formatted.append({
            "role": msg["role"],
            "content": msg["message"]
        })
    return formatted

async def log_interaction(
    user_message: str,
    assistant_response: str,
    session_id: str,
    sources_count: int,
    tokens_used: Dict[str, int],
    processing_time: float,
    confidence: Optional[float],
    rag_used: bool = False,
    rag_has_data: bool = False,
    rag_tech_data: bool = False,
    intent: str = "general"
):
    """Loguje interakcję w tle"""
    try:
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "session_id": session_id,
            "user_message_preview": user_message[:100],
            "assistant_response_preview": assistant_response[:100],
            "sources_count": sources_count,
            "processing_time": round(processing_time, 2),
            "confidence": round(confidence, 2) if confidence else None,
            "rag_used": rag_used,
            "rag_has_data": rag_has_data,
            "rag_tech_data": rag_tech_data,
            "intent": intent
        }
        
        logger.info(f"Interaction logged", extra=log_entry)
        
    except Exception as e:
        logger.error(f"Failed to log interaction: {str(e)}")

# ============================================
# FASTAPI APPLICATION
# ============================================

app = FastAPI(
    title=settings.APP_NAME,
    description="Asystent klienta ZK Motors z RAG i Cohere LLM. Oficjalny dealer BMW i MINI.",
    version=settings.APP_VERSION,
    docs_url="/docs" if not settings.IS_PRODUCTION else None,
    redoc_url="/redoc" if not settings.IS_PRODUCTION else None,
    openapi_url="/openapi.json" if not settings.IS_PRODUCTION else None,
)

if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["*"]
)

@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    response.headers["X-Process-Time"] = str(time.time() - start_time)
    return response

# ============================================
# BASIC ENDPOINTS
# ============================================

@app.get("/", response_class=HTMLResponse)
async def root():
    chat_html_path = TEMPLATES_DIR / "chat.html"
    
    if not chat_html_path.exists():
        return HTMLResponse(f"""
        <!DOCTYPE html>
        <html>
        <head><title>Error - File Not Found</title></head>
        <body>
            <h1>Error: chat.html not found</h1>
            <p>Expected path: {chat_html_path}</p>
        </body>
        </html>
        """)
    
    try:
        with open(chat_html_path, 'r', encoding='utf-8') as f:
            html_content = f.read()
        
        logger.info(f"Loaded chat.html from {chat_html_path}")
        
        js_config = f"""
        <script>
            window.API_BASE_URL = window.location.origin;
            window.API_ENDPOINTS = {{
                chat: '/chat',
                health: '/health',
                ping: '/ping',
                reset: '/chat/reset',
                status: '/api/status',
                rag_info: '/rag/info'
            }};
            console.log('API Base URL:', window.API_BASE_URL);
        </script>
        """
        
        html_content = html_content.replace('</head>', js_config + '</head>')
        return HTMLResponse(content=html_content)
        
    except Exception as e:
        logger.error(f"Error loading chat.html: {str(e)}")
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

@app.get("/ping")
async def ping():
    return {"status": "online", "timestamp": datetime.utcnow().isoformat()}

@app.get("/api/status")
async def api_status(rag_service: RAGService = Depends(get_rag_service)):
    rag_stats = await rag_service.get_stats()
    
    return {
        "online": True,
        "service": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "rag_status": rag_stats.get("status", "unknown"),
        "llm_ready": True,
        "timestamp": datetime.utcnow().isoformat(),
        "rag_stats": {
            "documents": rag_stats.get("documents_in_store", 0),
            "queries_processed": rag_stats.get("queries_processed", 0),
            "confidence_threshold": rag_stats.get("min_confidence_threshold", 0.6)
        }
    }

@app.get("/health/quick")
async def quick_health_check():
    try:
        memory_stats = {
            "active_sessions": len(conversation_memory),
            "total_messages": sum(len(h) for h in conversation_memory.values()),
            "max_history": MAX_HISTORY
        }
        
        return {
            "status": "healthy",
            "service": settings.APP_NAME,
            "version": settings.APP_VERSION,
            "timestamp": datetime.utcnow().isoformat(),
            "memory": memory_stats,
            "quick_check": True
        }
    except Exception as e:
        return {"status": "error", "error": str(e), "quick_check": True}

@app.get("/health", response_model=HealthResponse)
async def health_check(
    rag_service: RAGService = Depends(get_rag_service),
    llm_service: LLMService = Depends(get_llm_service)
):
    try:
        rag_health = await rag_service.health_check()
        rag_stats = await rag_service.get_stats()
        
        memory_stats = {
            "active_sessions": len(conversation_memory),
            "total_messages": sum(len(h) for h in conversation_memory.values()),
            "max_history": MAX_HISTORY
        }
        
        return HealthResponse(
            status=rag_health.get("status", "unknown"),
            timestamp=datetime.utcnow().isoformat(),
            version=settings.APP_VERSION,
            environment=settings.ENVIRONMENT,
            services={
                "rag": rag_health.get("status", "unknown"),
                "llm": "operational",
                "embedding": rag_health.get("embedding_service", "unknown"),
                "vector_store": rag_health.get("vector_store", "unknown")
            },
            memory=memory_stats,
            rag_stats=rag_stats
        )
        
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return HealthResponse(
            status="degraded",
            timestamp=datetime.utcnow().isoformat(),
            version=settings.APP_VERSION,
            environment=settings.ENVIRONMENT,
            services={"error": str(e)},
            memory={"error": "memory check failed"},
            rag_stats=None
        )

@app.get("/rag/info")
async def get_rag_info(rag_service: RAGService = Depends(get_rag_service)):
    try:
        health = await rag_service.health_check()
        stats = await rag_service.get_stats()
        
        return {
            "healthy": health.get("status") == "healthy",
            "vector_store": health.get("vector_store", "unknown"),
            "documents": stats.get("documents_in_store", 0),
            "queries_processed": stats.get("queries_processed", 0),
            "cache_hit_rate": stats.get("cache_hit_rate", 0),
            "intent_skipped": stats.get("intent_skipped", 0),
            "low_confidence_skipped": stats.get("low_confidence_skipped", 0),
            "confidence_threshold": stats.get("min_confidence_threshold", 0.6),
            "details": stats
        }
    except Exception as e:
        return {
            "healthy": False,
            "error": str(e),
            "details": "RAG service error"
        }

@app.get("/models")
async def list_models():
    models = [
        {
            "id": "command-r",
            "name": "Command R",
            "provider": "Cohere",
            "max_tokens": 128000,
            "context_length": 128000,
            "description": "Flagowy model do konwersacji"
        },
        {
            "id": "command-r-plus", 
            "name": "Command R+",
            "provider": "Cohere",
            "max_tokens": 128000,
            "context_length": 128000,
            "description": "Zaawansowany model z lepszym rozumieniem"
        }
    ]
    
    return {
        "models": models,
        "default_model": settings.COHERE_CHAT_MODEL,
        "memory_enabled": True,
        "max_history": MAX_HISTORY
    }

# ============================================
# CHAT ENDPOINT - UPDATED WITH NEW RAG & PROMPT SERVICE
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
    Główny endpoint chat z nowym RAG i PromptService
    """
    start_time = time.time()
    
    try:
        session_id = request.session_id
        user_message = request.message
        
        logger.info(f"Chat request: {user_message[:50]}... | {{'name': 'app.main'}}")
        
        # 1. Pobierz historię konwersacji
        history = get_conversation_history(session_id)
        conversation_history = format_history_for_prompt(history)
        is_first_message = len(history) == 0
        
        # 2. Użyj nowego RAGService z filtrowaniem intencji - POPRAWIONE: użyj retrieve zamiast retrieve_simple
        rag_results = await rag_service.retrieve_with_intent_check(
            query=user_message,
            top_k=3,
            confidence_threshold=0.6
        )
        
        logger.info(f"RAG results: has_data={rag_results.get('has_data')}, skip_rag={rag_results.get('skip_rag')}, below_threshold={rag_results.get('below_threshold')}, confidence={rag_results.get('confidence', 0.0):.2f}")
        
        # 3. Zbuduj prompt z nowym PromptService
        prompt_data = prompt_service.build_chat_prompt(
            user_message=user_message,
            rag_results=rag_results,
            conversation_history=conversation_history,
            session_id=session_id,
            language=request.language
        )
        
        # 4. Generuj odpowiedź
        response_text = ""
        tokens_used = None
        model_used = ""
        
        # Scenariusz A: Bezpośrednia odpowiedź (przywitania)
        if not prompt_data["use_llm"]:
            response_text = prompt_data["direct_response"]
            llm_success = True
            model_used = "greeting"
            logger.info(f"Using direct greeting response (skip_rag={rag_results.get('skip_rag')})")
        
        # Scenariusz B: Generuj przez LLM
        else:
            try:
                llm_result = await llm_service.generate(
                    prompt=prompt_data["prompt"],
                    model=settings.COHERE_CHAT_MODEL,
                    temperature=request.temperature,
                    max_tokens=400
                )
                
                # Wyodrębnij tekst odpowiedzi
                if hasattr(llm_result, 'text'):
                    response_text = llm_result.text
                    llm_success = True
                elif isinstance(llm_result, dict) and 'text' in llm_result:
                    response_text = llm_result['text']
                    llm_success = True
                elif isinstance(llm_result, dict) and 'generations' in llm_result:
                    response_text = llm_result['generations'][0]['text']
                    llm_success = True
                else:
                    llm_success = False
                
                # Pobierz użyte tokeny
                if hasattr(llm_result, 'tokens_used'):
                    tokens_used = llm_result.tokens_used
                elif isinstance(llm_result, dict) and 'tokens_used' in llm_result:
                    tokens_used = llm_result['tokens_used']
                    
                model_used = settings.COHERE_CHAT_MODEL
                    
            except Exception as llm_error:
                logger.error(f"LLM error: {str(llm_error)}")
                llm_success = False
        
        # 5. Jeśli LLM zawiódł, użyj fallback
        if not llm_success or not response_text.strip():
            logger.warning("LLM failed, using fallback")
            response_text = prompt_service.build_fallback_response(
                intent=rag_results.get('intent', 'general'),
                detected_models=rag_results.get('detected_models', []),
                confidence=rag_results.get('confidence', 0.0),
                is_technical=rag_results.get('tech', False)
            )
            model_used = "fallback"
        elif prompt_data["use_llm"]:
            # Czyść odpowiedź z formatowania LLM
            response_text = prompt_service.clean_response(
                response=response_text,
                session_id=session_id,
                rag_used=prompt_data.get("rag_used", False),
                rag_has_data=rag_results.get('has_data', False),
                confidence=rag_results.get('confidence', 0.0),
                intent=rag_results.get('intent', 'general')
            )
        
        # 6. Dodaj do historii
        add_to_history(session_id, "user", user_message)
        add_to_history(session_id, "assistant", response_text)
        
        # 7. Przygotuj odpowiedź API
        processing_time = time.time() - start_time
        
        # Przygotuj źródła
        sources = []
        if rag_results.get('has_data') and rag_results.get('sources'):
            for i, source in enumerate(rag_results['sources'][:2], 1):
                sources.append({
                    'id': i,
                    'title': source.get('title', 'Źródło')[:80],
                    'content_preview': source.get('content', '')[:150] + ('...' if len(source.get('content', '')) > 150 else ''),
                    'similarity': round(source.get('score', 0), 3),
                    'url': source.get('url', ''),
                    'source': source.get('source', 'unknown')
                })
        
        # Określ jakość danych
        if rag_results.get('tech', False):
            data_quality = "high"
        elif rag_results.get('has_data', False):
            data_quality = "medium"
        else:
            data_quality = "low"
        
        # Aktualizuj quality w zależności od confidence
        confidence = rag_results.get('confidence', 0.0)
        if confidence < 0.4:
            data_quality = "low"
        elif confidence < 0.7:
            data_quality = "medium"
        else:
            data_quality = "high"
        
        response = ChatResponse(
            answer=response_text,
            session_id=session_id,
            history_length=len(get_conversation_history(session_id)),
            processing_time=processing_time,
            sources=sources,
            model_used=model_used,
            tokens_used=tokens_used,
            confidence=confidence,
            rag_info={
                "has_data": rag_results.get('has_data', False),
                "has_technical_data": rag_results.get('tech', False),
                "confidence": confidence,
                "intent": rag_results.get('intent', 'general'),
                "skip_rag": rag_results.get('skip_rag', False),
                "below_threshold": rag_results.get('below_threshold', False)
            },
            data_quality=data_quality
        )
        
        # 8. Loguj w tle
        background_tasks.add_task(
            log_interaction,
            user_message=user_message,
            assistant_response=response_text,
            session_id=session_id,
            sources_count=len(sources),
            tokens_used=tokens_used,
            processing_time=processing_time,
            confidence=confidence,
            rag_used=prompt_data.get("rag_used", False),
            rag_has_data=rag_results.get('has_data', False),
            rag_tech_data=rag_results.get('tech', False),
            intent=rag_results.get('intent', 'general')
        )
        
        logger.info(f"Response in {processing_time:.2f}s, quality: {data_quality}", extra={
            'name': 'app.main',
            'extra': {
                'rag_data': rag_results.get('has_data', False),
                'sources': len(sources)
            }
        })
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Chat error: {str(e)}", exc_info=True)
        
        # Ostateczny fallback
        fallback_response = """Przepraszam, wystąpił błąd systemu. 
Jestem Leo, ekspertem BMW w ZK Motors. 
Proszę spróbować ponownie lub skontaktować się bezpośrednio z naszym salonem."""
        
        return ChatResponse(
            answer=fallback_response,
            success=False,
            session_id=request.session_id,
            history_length=len(get_conversation_history(request.session_id)),
            processing_time=time.time() - start_time,
            sources=[],
            model_used="fallback",
            confidence=0.0,
            rag_info={"error": str(e)},
            data_quality="error"
        )

@app.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    async def event_generator():
        yield f"data: {json.dumps({'error': 'Streaming not implemented'})}\n\n"
        yield "data: [DONE]\n\n"
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache"}
    )

@app.post("/chat/reset", response_model=ResetResponse)
async def reset_chat(session_id: str = "default"):
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
    history = get_conversation_history(session_id)
    
    return HistoryResponse(
        session_id=session_id,
        history=history[-limit:],
        total_messages=len(history),
        limit=limit
    )

# ============================================
# DEBUG ENDPOINTS
# ============================================

@app.get("/test/rag")
async def test_rag(
    query: str = "BMW X5 moc silnik",
    rag_service: RAGService = Depends(get_rag_service)
):
    """Testowy endpoint do sprawdzenia działania RAG"""
    if settings.IS_PRODUCTION:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Debug endpoints disabled in production"
        )
    
    result = await rag_service.retrieve_with_intent_check(query=query, top_k=3)
    
    return {
        "query": query,
        "has_data": result['has_data'],
        "skip_rag": result.get('skip_rag', False),
        "below_threshold": result.get('below_threshold', False),
        "confidence": result['confidence'],
        "intent": result.get('intent', 'general'),
        "tech": result.get('tech', False),
        "documents_count": len(result.get('documents', [])),
        "sources_count": len(result.get('sources', []))
    }

@app.get("/debug/rag")
async def debug_rag(
    query: str,
    top_k: int = 3,
    rag_service: RAGService = Depends(get_rag_service)
):
    if settings.IS_PRODUCTION:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Debug endpoints disabled in production"
        )
    
    result = await rag_service.retrieve_with_intent_check(query=query, top_k=top_k)
    
    return {
        "query": query,
        "result": result
    }

@app.get("/debug/stats")
async def debug_stats(
    rag_service: RAGService = Depends(get_rag_service),
    llm_service: LLMService = Depends(get_llm_service),
    prompt_service: PromptService = Depends(get_prompt_service)
):
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
        "prompt_service": {
            "response_history_size": len(prompt_service.response_history),
            "max_history": prompt_service.max_history
        },
        "timestamp": datetime.utcnow().isoformat()
    }

# ============================================
# APPLICATION EVENTS
# ============================================

@app.on_event("startup")
async def startup_event():
    try:
        validate_configuration()
        await init_cache()
        
        # Inicjalizuj serwisy
        rag_service = await get_rag_service()
        rag_stats = await rag_service.get_stats()
        
        logger.info(f"{settings.APP_NAME} v{settings.APP_VERSION} starting up...")
        logger.info(f"Environment: {settings.ENVIRONMENT}")
        logger.info(f"LLM Model: {settings.COHERE_CHAT_MODEL}")
        logger.info(f"RAG Documents: {rag_stats.get('documents_in_store', 0)}")
        logger.info(f"RAG Confidence Threshold: {rag_stats.get('min_confidence_threshold', 0.6)}")
        logger.info(f"Memory: {MAX_HISTORY} messages per session")
        logger.info(f"API: http://{settings.HOST}:{settings.PORT}")
        logger.info(f"Chat: http://{settings.HOST}:{settings.PORT}/")
        
        chat_html_path = TEMPLATES_DIR / "chat.html"
        if chat_html_path.exists():
            logger.info(f"HTML Interface: chat.html found")
        else:
            logger.warning(f"HTML Interface: chat.html NOT FOUND")
        
        logger.info("Application started successfully")
        
        if settings.IS_DEVELOPMENT:
            logger.warning("Running in DEVELOPMENT mode")
        
    except Exception as e:
        logger.critical(f"Startup failed: {str(e)}", exc_info=True)
        raise

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Shutting down application...")
    logger.info(f"Memory stats: {len(conversation_memory)} sessions, {sum(len(h) for h in conversation_memory.values())} total messages")

# ============================================
# ERROR HANDLERS
# ============================================

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
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
# MAIN ENTRY POINT
# ============================================

def main():
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
            print(f"{settings.APP_NAME} v{settings.APP_VERSION}")
            print(f"Environment: {settings.ENVIRONMENT}")
            print(f"Chat: http://{settings.HOST}:{settings.PORT}/")
            print(f"Model: {settings.COHERE_CHAT_MODEL}")
            print(f"Memory: {MAX_HISTORY} messages per session")
            print(f"Test RAG: http://{settings.HOST}:{settings.PORT}/test/rag?query=BMW+X5")
            print(f"Health: http://{settings.HOST}:{settings.PORT}/health")
            print(f"{'='*60}\n")
        
        uvicorn.run(**config)
        
    except Exception as e:
        print(f"Failed to start: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()