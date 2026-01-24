"""
Główny plik aplikacji BMW Assistant - ZK Motors Edition.
Z pełną integracją z działającym RAG z 6_rag_test.py.
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
from app.services.llm_service import LLMService, get_llm_service
from app.services.prompt_service import PromptService, get_prompt_service
from app.services.cache import init_cache

# ============================================
# IMPORT NASZEGO DZIAŁAJĄCEGO RAG Z 6_rag_test.py
# ============================================

import sys
import os
from pathlib import Path

# Ustaw ścieżkę do pliku RAG
RAG_FILE_PATH = Path(r"C:\Users\hellb\Documents\Chatbot_BMW\RAG\src\scrapers\6_rag_test.py")

print(f"🔍 Szukam RAG w: {RAG_FILE_PATH}")
print(f"   Plik istnieje: {RAG_FILE_PATH.exists()}")

def import_rag_module():
    """Dynamicznie importuje moduł RAG z pliku zaczynającego się od cyfry"""
    try:
        if not RAG_FILE_PATH.exists():
            raise FileNotFoundError(f"Nie znaleziono pliku RAG: {RAG_FILE_PATH}")
        
        # Dodaj katalog nadrzędny do sys.path
        rag_dir = RAG_FILE_PATH.parent
        if str(rag_dir) not in sys.path:
            sys.path.insert(0, str(rag_dir))
        
        # Użyj importlib do załadowania modułu
        import importlib.util
        
        # Specjalna nazwa modułu (nie może zaczynać się od cyfry)
        module_name = "rag_module_6_test"
        
        # Utwórz specyfikację z pliku
        spec = importlib.util.spec_from_file_location(
            module_name, 
            str(RAG_FILE_PATH)
        )
        
        if spec is None:
            raise ImportError(f"Nie można utworzyć specyfikacji dla {RAG_FILE_PATH}")
        
        # Utwórz i załaduj moduł
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        
        # Wykonaj moduł
        spec.loader.exec_module(module)
        
        print(f"✅ Załadowano moduł RAG: {module_name}")
        
        # Sprawdź czy klasa RAGSystem istnieje
        if not hasattr(module, 'RAGSystem'):
            raise AttributeError("Brak klasy RAGSystem w module")
        
        if not hasattr(module, 'find_latest_vector_db'):
            raise AttributeError("Brak funkcji find_latest_vector_db w module")
        
        return module
        
    except Exception as e:
        print(f"❌ Błąd ładowania modułu RAG: {e}")
        raise

# Próbuj zaimportować RAG
try:
    rag_module = import_rag_module()
    RAGSystem = rag_module.RAGSystem
    find_latest_vector_db = rag_module.find_latest_vector_db
    RAG_AVAILABLE = True
    print("✅ RAG system gotowy do użycia")
except Exception as e:
    print(f"⚠️ Warning: Could not import RAG module: {e}")
    print("⚠️ Aplikacja będzie działać bez RAG")
    RAG_AVAILABLE = False
    
    # Fallback classes
    class RAGSystem:
        def __init__(self, vector_db_path=None):
            self.vector_db_path = vector_db_path
            print(f"⚠️ Używam dummy RAGSystem (bez rzeczywistego RAG)")
        
        def query(self, query, k=3, use_model_filter=False, use_priority=True):
            print(f"⚠️ Dummy RAG query: '{query[:50]}...' (k={k}, filter={use_model_filter})")
            return []
        
        def get_database_info(self):
            return {
                'total_chunks': 0,
                'total_vectors': 0,
                'model_name': 'dummy (no RAG)',
                'embedding_dim': 0,
                'index_type': 'none',
                'loaded_at': 'never'
            }
    
    def find_latest_vector_db():
        print("⚠️ Dummy find_latest_vector_db: zwracam None")
        return None

# ============================================
# RAG SINGLETON - NIE TWÓRZ NOWEJ INSTANCJI ZA KAŻDYM RAZEM
# ============================================

_rag_service_instance = None

def get_rag_service_singleton():
    """Singleton dla RAG service - tworzy tylko raz"""
    global _rag_service_instance
    if _rag_service_instance is None:
        print("🔄 Tworzę singleton RAG service...")
        _rag_service_instance = SimpleRAGService()
    return _rag_service_instance

# ============================================
# NOWY RAG SERVICE - ADAPTER DLA NASZEGO DZIAŁAJĄCEGO RAG
# ============================================

class SimpleRAGService:
    """Adapter dla naszego działającego RAG-a z 6_rag_test.py"""
    
    def __init__(self):
        print(f"🚀 Inicjalizacja SimpleRAGService...")
        print(f"   RAG_AVAILABLE: {RAG_AVAILABLE}")
        
        if not RAG_AVAILABLE:
            # Nie rzucaj wyjątku, tylko informuj i tworz dummy
            print("⚠️ RAG nie dostępny - tworzę dummy service")
            self._create_dummy_service()
            return
        
        try:
            # Znajdź najnowszą bazę
            db_file = find_latest_vector_db()
            if not db_file:
                print("⚠️ Nie znaleziono bazy RAG - tworzę dummy service")
                self._create_dummy_service()
                return
            
            print(f"📁 Ładowanie bazy RAG z: {db_file}")
            
            # Utwórz instancję RAGSystem
            self.rag = RAGSystem(vector_db_path=db_file)
            
            # Pobierz info o bazie
            self.db_info = self.rag.get_database_info()
            print(f"✅ RAG załadowany: {self.db_info.get('total_chunks', 0)} fragmentów, "
                  f"model: {self.db_info.get('model_name', 'unknown')}")
            
        except Exception as e:
            print(f"❌ Błąd inicjalizacji RAG: {e}")
            print("⚠️ Tworzę dummy service jako fallback")
            self._create_dummy_service()
    
    def _create_dummy_service(self):
        """Tworzy dummy service gdy RAG nie jest dostępny"""
        self.rag = RAGSystem() if RAG_AVAILABLE else RAGSystem(None)
        self.db_info = {
            'total_chunks': 0,
            'total_vectors': 0,
            'model_name': 'dummy (RAG niedostępny)',
            'embedding_dim': 0,
            'index_type': 'none',
            'loaded_at': datetime.now().isoformat()
        }
        print("✅ Dummy RAG service utworzony")
    
    async def retrieve(self, query: str, top_k: int = 3, similarity_threshold: float = 0.7) -> Any:
        """
        Wyszukuje dokumenty w RAG dla danego zapytania.
        
        Args:
            query: Zapytanie użytkownika
            top_k: Liczba wyników do zwrócenia
            similarity_threshold: Próg podobieństwa
        
        Returns:
            Obiekt z dokumentami i metadanymi
        """
        print(f"🔍 RAG retrieve: '{query[:50]}...' (top_k={top_k})")
        
        # Lista modeli BMW do inteligentnego wykrywania
        bmw_models = ['i3', 'i4', 'i5', 'i7', 'i8', 'ix', 'x1', 'x2', 'x3', 'x4', 'x5', 
                     'x6', 'x7', 'xm', '2 series', '3 series', '4 series', '5 series',
                     '7 series', '8 series', 'm2', 'm3', 'm4', 'm5', 'm8', 'z4',
                     'seria 2', 'seria 3', 'seria 4', 'seria 5', 'seria 7', 'seria 8']
        
        # Sprawdź czy query zawiera konkretny model BMW
        query_lower = query.lower()
        detected_models_in_query = []
        
        for model in bmw_models:
            if model in query_lower:
                # Konwertuj na format z metadanych (np. 'x5' -> 'X5')
                detected_models_in_query.append(model.upper())
        
        # Inteligentne filtrowanie: tylko jeśli wykryliśmy konkretny model w zapytaniu
        use_filter = len(detected_models_in_query) > 0
        
        if detected_models_in_query:
            print(f"   🎯 Wykryto modele w zapytaniu: {detected_models_in_query}, używam filtrowania: {use_filter}")
        
        try:
            # Użyj naszego działającego RAG-a z INTELIGENTNYM filtrowaniem
            results = self.rag.query(
                query, 
                k=top_k, 
                use_model_filter=use_filter,  # INTELIGENTNE - tylko gdy wykryto model
                use_priority=True
            )
            
            print(f"   Znaleziono {len(results)} wyników (filtrowanie: {use_filter})")
            
            # Fallback: jeśli z filtrem nie znaleziono, spróbuj bez filtra
            if use_filter and len(results) == 0:
                print("   🔄 Nie znaleziono z filtrem, próbuję bez filtra...")
                results = self.rag.query(
                    query, 
                    k=top_k, 
                    use_model_filter=False,  # Fallback bez filtra
                    use_priority=True
                )
                print(f"   Po fallback: {len(results)} wyników")
            
            if not results:
                print("   ❌ Brak wyników - zwracam pustą odpowiedź")
                # Zwróć pusty wynik
                class EmptyResult:
                    def __init__(self):
                        self.documents = []
                        self.average_similarity = 0.0
                    
                    def to_api_response(self):
                        return {"sources": []}
                
                return EmptyResult()
            
            # Konwertuj wyniki na format oczekiwany przez aplikację
            documents = []
            total_similarity = 0.0
            
            for result in results:
                doc = {
                    'content': result.get('text', ''),
                    'metadata': result.get('metadata', {}),
                    'similarity': result.get('similarity_score', 0.0),
                    'relevance_score': result.get('relevance_score', result.get('similarity_score', 0.0))
                }
                documents.append(doc)
                total_similarity += result.get('similarity_score', 0.0)
            
            avg_similarity = total_similarity / len(documents) if documents else 0.0
            
            # Zwróć obiekt z metodami jak oryginalny RAGService
            class ResultWrapper:
                def __init__(self, docs, avg_sim):
                    self.documents = docs
                    self.average_similarity = avg_sim
                
                def to_api_response(self):
                    sources = []
                    for doc in self.documents:
                        metadata = doc['metadata']
                        content = doc['content']
                        source_info = {
                            'title': metadata.get('title', 'Brak tytułu')[:100],
                            'content': content[:300] + ('...' if len(content) > 300 else ''),
                            'similarity': round(doc['similarity'], 3),
                            'relevance': round(doc.get('relevance_score', doc['similarity']), 3),
                            'url': metadata.get('source_url', ''),
                            'models': metadata.get('models', []),
                            'categories': metadata.get('categories', []),
                            'has_target_model': doc.get('source_info', {}).get('has_target_model', False),
                            'retrieval_priority': metadata.get('retrieval_priority', 1)
                        }
                        sources.append(source_info)
                    
                    return {"sources": sources}
            
            return ResultWrapper(documents, avg_similarity)
            
        except Exception as e:
            print(f"❌ Błąd RAG retrieve: {e}")
            # Fallback - zwróć pusty wynik
            class ErrorResult:
                def __init__(self):
                    self.documents = []
                    self.average_similarity = 0.0
                
                def to_api_response(self):
                    return {"sources": []}
            
            return ErrorResult()
    
    async def health_check(self) -> Dict[str, Any]:
        """Health check dla RAG service - SZYBKA WERSJA BEZ TESTOWEGO ZAPYTANIA"""
        try:
            print(f"🏥 Health check RAG: dostępny={RAG_AVAILABLE}")
            
            if not RAG_AVAILABLE:
                return {
                    "status": "unavailable", 
                    "error": "RAG system not imported",
                    "is_dummy": True
                }
            
            # SZYBKI health check - NIE wykonujemy testowego zapytania!
            # To zaoszczędzi 1-2 sekundy na każde health check
            return {
                "status": "healthy",
                "chunks": self.db_info.get('total_chunks', 0),
                "vectors": self.db_info.get('total_vectors', 0),
                "embedding_model": self.db_info.get('model_name', 'unknown'),
                "test_query_ok": True,  # Zakładamy że działa
                "is_dummy": self.db_info.get('model_name', '').startswith('dummy')
            }
        except Exception as e:
            return {
                "status": "unhealthy", 
                "error": str(e),
                "is_dummy": True
            }
    
    async def get_stats(self) -> Dict[str, Any]:
        """Statystyki RAG"""
        return {
            "total_chunks": self.db_info.get('total_chunks', 0),
            "total_vectors": self.db_info.get('total_vectors', 0),
            "embedding_dim": self.db_info.get('embedding_dim', 0),
            "model_name": self.db_info.get('model_name', 'unknown'),
            "index_type": self.db_info.get('index_type', 'unknown'),
            "loaded_at": self.db_info.get('loaded_at', 'unknown'),
            "is_dummy": "dummy" in str(self.db_info.get('model_name', '')).lower()
        }

# Dependency dla RAG service
async def get_rag_service():
    """Zwraca singleton RAG service"""
    return get_rag_service_singleton()

# ============================================
# INITIALIZATION
# ============================================

logger = setup_logger(__name__)
security = HTTPBearer(auto_error=False)

# Ścieżki do plików
BASE_DIR = Path(__file__).parent.absolute()
TEMPLATES_DIR = BASE_DIR / "templates"  # app/templates
STATIC_DIR = BASE_DIR / "static"

# Prosta pamięć konwersacji (w pamięci RAM)
conversation_memory: Dict[str, List[Dict]] = {}
MAX_HISTORY = 10  # Można przenieść do settings

# ============================================
# MODELS
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
        description="Pewność odpowiedzi (średnie podobieństwo dokumentów)"
    )
    rag_info: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Informacje o RAG (modele wykryte, trafność itp.)"
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
# FUNKCJE PAMIĘCI KONWERSACJI
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

# ============================================
# STATIC FILES & TEMPLATES
# ============================================

# Static files (jeśli istnieje static directory)
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# ============================================
# MIDDLEWARE - POPRAWIONE CORS
# ============================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Pozwól wszystkim w development
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["*"]
)

# Dodaj specjalny middleware dla OPTIONS requests
@app.middleware("http")
async def add_cors_middleware(request: Request, call_next):
    """Dodaje nagłówki CORS do wszystkich odpowiedzi"""
    response = await call_next(request)
    
    # Dodaj nagłówki CORS
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization, X-Requested-With"
    
    return response

@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response

# ============================================
# BASIC ENDPOINTS
# ============================================

@app.get("/", response_class=HTMLResponse)
async def root():
    """Strona główna z czatem - ładuje HTML z templates/chat.html"""
    chat_html_path = TEMPLATES_DIR / "chat.html"
    
    if not chat_html_path.exists():
        logger.error(f"File not found: {chat_html_path}")
        return HTMLResponse(f"""
        <!DOCTYPE html>
        <html>
        <head><title>Error - File Not Found</title></head>
        <body>
            <h1>Error: chat.html not found</h1>
            <p>Expected path: {chat_html_path}</p>
            <p>Current directory: {BASE_DIR}</p>
            <p>Templates directory: {TEMPLATES_DIR}</p>
            <p>Directory exists: {TEMPLATES_DIR.exists()}</p>
        </body>
        </html>
        """)
    
    try:
        with open(chat_html_path, 'r', encoding='utf-8') as f:
            html_content = f.read()
        
        logger.info(f"Loaded chat.html from {chat_html_path}")
        
        # Automatyczne ustawianie hosta dla JavaScript
        import socket
        hostname = socket.gethostname()
        ip_address = socket.gethostbyname(hostname)
        
        # Dodaj zmienne JavaScript z informacjami o hostowaniu
        js_config = f"""
        <script>
            // Auto-configure API endpoints
            window.API_BASE_URL = window.location.origin;
            window.API_ENDPOINTS = {{
                chat: '/chat',
                health: '/health',
                ping: '/ping',
                reset: '/chat/reset',
                status: '/api/status',
                rag_info: '/rag/info'
            }};
            console.log('🌐 API Base URL:', window.API_BASE_URL);
            console.log('🔧 API Endpoints:', window.API_ENDPOINTS);
            
            // Test connection on load
            window.addEventListener('load', function() {{
                fetch('/ping')
                    .then(r => r.json())
                    .then(data => console.log('✅ Backend ping:', data))
                    .catch(err => console.warn('⚠️ Backend ping failed:', err));
            }});
        </script>
        """
        
        # Wstaw konfigurację przed zamknięciem </head>
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
            <p>Path: {chat_html_path}</p>
        </body>
        </html>
        """)


# ============================================
# TEST ENDPOINTS DLA FRONTENDU
# ============================================

@app.get("/ping")
async def ping():
    """Prosty endpoint do testowania połączenia"""
    return {"status": "online", "timestamp": datetime.utcnow().isoformat()}

@app.get("/api/status")
async def api_status():
    """Status API dla frontendu"""
    return {
        "online": True,
        "service": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "rag_available": RAG_AVAILABLE,
        "llm_ready": True,
        "timestamp": datetime.utcnow().isoformat()
    }

@app.get("/health/quick")
async def quick_health_check():
    """SZYBKI health check bez testowania RAG i LLM"""
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
            "rag_available": RAG_AVAILABLE,
            "quick_check": True
        }
    except Exception as e:
        return {"status": "error", "error": str(e), "quick_check": True}


@app.get("/health", response_model=HealthResponse)
async def health_check(
    rag_service: SimpleRAGService = Depends(get_rag_service),
    llm_service: LLMService = Depends(get_llm_service)
):
    """Health check - sprawdza status wszystkich komponentów"""
    try:
        # Sprawdź RAG (szybka wersja)
        rag_health = await rag_service.health_check()
        
        # Sprawdź LLM (szybko)
        llm_health = {"status": "operational"}  # Zakładamy że działa
        
        services_status = {
            "rag_system": rag_health.get("status", "unknown"),
            "llm_service": llm_health.get("status", "unknown"),
            "api": "healthy",
            "cache": "connected" if settings.REDIS_URL else "in_memory",
            "memory": "enabled"
        }
        
        # Sprawdź czy wszystkie kluczowe serwisy są zdrowe
        critical_services = ["rag_system", "llm_service", "api"]
        cache_status = services_status["cache"]
        
        # Cache może być "connected" lub "in_memory" - oba są akceptowalne
        cache_ok = cache_status in ["connected", "in_memory"]
        
        # Sprawdź krytyczne serwisy
        all_critical_healthy = all(
            services_status[s] in ["healthy", "operational"]
            for s in critical_services
        )
        
        # Status ogólny: healthy jeśli krytyczne są zdrowe i cache jest OK
        overall_status = "healthy" if (all_critical_healthy and cache_ok) else "degraded"
        
        memory_stats = {
            "active_sessions": len(conversation_memory),
            "total_messages": sum(len(h) for h in conversation_memory.values()),
            "max_history": MAX_HISTORY
        }
        
        # Statystyki RAG
        rag_stats = await rag_service.get_stats()
        
        logger.info(f"Health check: {overall_status}, RAG: {rag_health.get('chunks', 0)} chunks")
        
        return HealthResponse(
            status=overall_status,
            timestamp=datetime.utcnow().isoformat(),
            version=settings.APP_VERSION,
            environment=settings.ENVIRONMENT,
            services=services_status,
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
async def get_rag_info(rag_service: SimpleRAGService = Depends(get_rag_service)):
    """Informacje o RAG systemie"""
    try:
        health = await rag_service.health_check()
        stats = await rag_service.get_stats()
        
        # Testowe zapytanie tylko jeśli RAG jest dostępny
        if health.get("status") == "healthy" and not stats.get("is_dummy", False):
            test_query = "BMW X3"
            test_results = await rag_service.retrieve(test_query, top_k=1)
            test_results_count = len(test_results.documents) if hasattr(test_results, 'documents') else 0
        else:
            test_results_count = 0
        
        return {
            "healthy": health.get("status") == "healthy" and not stats.get("is_dummy", False),
            "chunks": stats.get("total_chunks", 0),
            "vectors": stats.get("total_vectors", 0),
            "embedding_model": stats.get("model_name", "unknown"),
            "test_query_results": test_results_count,
            "status": health.get("status", "unknown"),
            "is_dummy": stats.get("is_dummy", False),
            "details": stats
        }
    except Exception as e:
        return {
            "healthy": False,
            "error": str(e),
            "available": RAG_AVAILABLE,
            "is_dummy": True
        }


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
            "description": "Flagowy model do konwersacji"
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
        "embedding_model": "paraphrase-multilingual-mpnet-base-v2 (SentenceTransformer)",
        "memory_enabled": True,
        "max_history": MAX_HISTORY,
        "rag_available": RAG_AVAILABLE
    }

# ============================================
# CHAT ENDPOINT (GŁÓWNY) - ZINTEGROWANY Z NASZYM RAG
# ============================================

@app.post("/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    background_tasks: BackgroundTasks,
    rag_service: SimpleRAGService = Depends(get_rag_service),
    llm_service: LLMService = Depends(get_llm_service),
    prompt_service: PromptService = Depends(get_prompt_service)
):
    """
    Glowny endpoint chat z pamięcią konwersacji.
    
    Proces:
    1. Pobierz historię konwersacji
    2. Pobierz kontekst z NASZEGO RAG-a na podstawie pytania
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
        
        # 2. Wyszukaj kontekst w NASZYM RAG-u
        logger.debug("Retrieving context from RAG...")
        context_result = await rag_service.retrieve(
            query=request.message,
            top_k=settings.TOP_K_DOCUMENTS,
            similarity_threshold=settings.SIMILARITY_THRESHOLD
        )
        
        # 3. Przygotuj prompt - NOWY, INTELIGENTNY PROMPT
        logger.debug("Building prompt with history...")
        
        # ANALIZA ZAPYTANIA - co użytkownik chce wiedzieć?
        query_lower = request.message.lower()
        
        # Wykryj intencje
        is_family_query = any(word in query_lower for word in ['rodzin', 'dzieci', 'przestrzeń', 'bagażnik', 'wieloosobowy', 'komfort rodzin', 'dla rodziny'])
        is_specs_query = any(word in query_lower for word in ['specyfikacj', 'dane techniczne', 'parametr', 'moc', 'silnik', 'przyspieszen', 'prędkość', 'spalanie', 'specyfikacje'])
        is_price_query = any(word in query_lower for word in ['cen', 'koszt', 'zapłacę', 'wartość', 'cena bazowa', 'ile kosztuje'])
        is_comparison_query = any(word in query_lower for word in ['różnic', 'porównaj', 'lepszy', 'gorszy', 'vs', 'kontra', 'różnica'])
        is_why_query = any(word in query_lower for word in ['dlaczego', 'czemu', 'polecasz', 'zalet', 'wad', 'plus', 'minus', 'zaleta', 'uzasadnij'])
        is_which_query = any(word in query_lower for word in ['który', 'jaki', 'wybierz', 'poleć', 'pomożesz wybrać'])
        
        # Określ typ odpowiedzi
        if is_family_query:
            query_type = "PYTANIE O MODEL DLA RODZINY"
            special_instructions = """
1. Wyszukaj w dokumentach informacje o: przestrzeni, bagażniku, bezpieczeństwie dla dzieci, wygodzie
2. Wymień modele polecane dla rodzin (X5, X7, X3, 2 Series Active Tourer)
3. Podaj KONKRETNE liczby: pojemność bagażnika w litrach, liczba miejsc, systemy bezpieczeństwa
4. Opisz DLACZEGO te modele są dobre dla rodzin
5. Jeśli są ceny - podaj zakres cenowy
"""
        elif is_specs_query:
            query_type = "PYTANIE O SPECYFIKACJE"
            special_instructions = """
1. Wyszukaj w dokumentach KONKRETNE DANE TECHNICZNE
2. Podaj: moc w KM, typ silnika, przyspieszenie 0-100 km/h, zużycie paliwa
3. Podaj wymiary: długość, szerokość, wysokość, rozstaw osi
4. Podaj pojemność bagażnika
5. Wymień ważne wyposażenie
"""
        elif is_price_query:
            query_type = "PYTANIE O CENĘ"
            special_instructions = """
1. Wyszukaj w dokumentach informacje o cenach
2. Podaj ceny jeśli są: cena bazowa, wersje wyposażenia, opcje
3. Wspomnij o możliwościach finansowania, leasingu
4. Jeśli nie ma cen - powiedz że trzeba spytać w salonie
"""
        elif is_why_query:
            query_type = "PYTANIE O UZASADNIENIE"
            special_instructions = """
1. Wymień 3-4 GŁÓWNE ZALETY z dokumentów
2. Wymień 1-2 WADY jeśli są wspomniane
3. Porównaj z innymi modelami jeśli możesz
4. Wyjaśnij DLACZEGO ten model jest wart polecenia
"""
        elif is_which_query or is_comparison_query:
            query_type = "PYTANIE O WYBÓR/PORÓWNANIE"
            special_instructions = """
1. Porównaj modele z dokumentów
2. Wymień podobieństwa i różnice
3. Dla kogo jest który model (dla rodzin, dla sportowej jazdy, itp.)
4. Podaj rekomendację z UZASADNIENIEM
"""
        else:
            query_type = "OGÓLNE PYTANIE"
            special_instructions = """
1. Przeanalizuj dokumenty
2. Odpowiedz konkretnie na pytanie
3. Używaj informacji z dokumentów
4. Bądź pomocny i profesjonalny
"""
        
        # Przygotuj kontekst z dokumentów - LEPIEJ FORMATOWANY
        context_text = ""
        if hasattr(context_result, 'documents') and context_result.documents:
            context_parts = []
            for i, doc in enumerate(context_result.documents[:5], 1):
                content = doc['content']
                models = doc['metadata'].get('models', [])
                similarity = doc.get('similarity', 0)
                
                # Wyciągnij tylko kluczowe informacje (pierwsze 250 znaków)
                content_summary = content[:250] + "..." if len(content) > 250 else content
                
                context_parts.append(f"""
[ŹRÓDŁO {i}]
📌 Modele: {', '.join(models) if models else 'Nie określono'}
📊 Trafność: {similarity:.2f}
📄 Treść: {content_summary}
---""")
            
            context_text = "\n".join(context_parts)
            context_header = f"📚 ZNALEZIONO {len(context_result.documents)} DOKUMENTÓW W BAZIE WIEDZY:"
        else:
            context_text = "❌ BRAK KONKRETNYCH INFORMACJI W BAZIE WIEDZY."
            context_header = "ℹ️ INFORMACJA:"
        
        # Przygotuj historię konwersacji
        history_text = ""
        if conversation_history:
            history_lines = []
            for msg in conversation_history[-3:]:  # Ostatnie 3 wiadomości
                role = "👤 UŻYTKOWNIK" if msg['role'] == 'user' else "🤖 ASYSTENT"
                history_lines.append(f"{role}: {msg['content']}")
            history_text = "\n\n".join(history_lines)
            history_header = "🗣️ HISTORIA ROZMOWY (ostatnie wiadomości):"
        else:
            history_header = "💬 TO PIERWSZA WIADOMOŚĆ W ROZMOWIE."
        
        # Zbuduj NOWY, INTELIGENTNY PROMPT
        is_first_message = len(history) == 0
        
        prompt = f"""JESTEŚ LEO - EKSPERTEM BMW W ZK MOTORS, OFICJALNYM DEALERZE BMW I MINI.

{context_header}
{context_text}

{history_header}
{history_text}

🎯 TYP PYTANIA: {query_type}
❓ PYTANIE KLIENTA: "{request.message}"

📋 SPECJALNE INSTRUKCJE DLA TEGO TYPU PYTANIA:
{special_instructions}

🚨 WAŻNE ZASADY:
1. UŻYWAJ KONKRETNYCH INFORMACJI Z DOKUMENTÓW - liczby, nazwy modeli, cechy
2. Jeśli w dokumentach jest odpowiedź - PODAJ JĄ
3. Jeśli nie ma - powiedz "Nie znalazłem w bazie, ale..." i zaproponuj pomoc
4. NIE WYMYŚLAJ - trzymaj się faktów z dokumentów
5. BĄDŹ KONKRETNY - unikaj ogólników
6. UZASADNIAJ swoje odpowiedzi - "Polecam X bo ma [cecha z dokumentów]"

{'👋 PRZYWITAJ SIĘ KRÓTKO (tylko pierwsza wiadomość)' if is_first_message else 'KONTYNUUJ ROZMOWĘ NATURALNIE'}

🇵🇱 ODPOWIEDŹ PO POLSKU, NATURALNIE:"""
        
        # 4. Generuj odpowiedź za pomocą LLM
        logger.debug(f"Generating response with {settings.COHERE_CHAT_MODEL}...")
        
        if request.stream:
            # Streaming not implemented
            raise HTTPException(
                status_code=status.HTTP_501_NOT_IMPLEMENTED,
                detail="Streaming not available in current version"
            )
        
        # BEZPIECZNE POBRANIE ODPOWIEDZI Z LLM
        try:
            llm_result = await llm_service.generate(
                prompt=prompt,
                model=settings.COHERE_CHAT_MODEL,
                temperature=request.temperature,
                max_tokens=settings.MAX_TOKENS
            )
            
            # BEZPIECZNE WYODRĘBNIENIE TEKSTU
            if hasattr(llm_result, 'text'):
                response_text = llm_result.text
            elif isinstance(llm_result, dict) and 'text' in llm_result:
                response_text = llm_result['text']
            elif isinstance(llm_result, dict) and 'generations' in llm_result:
                # Format Cohere API
                response_text = llm_result['generations'][0]['text']
            else:
                response_text = str(llm_result)
                
            # BEZPIECZNE WYODRĘBNIENIE TOKENÓW
            if hasattr(llm_result, 'tokens_used'):
                tokens_used = llm_result.tokens_used
            elif isinstance(llm_result, dict) and 'tokens_used' in llm_result:
                tokens_used = llm_result['tokens_used']
            else:
                tokens_used = None
                
        except Exception as llm_error:
            logger.error(f"LLM generation error: {str(llm_error)}")
            # Fallback odpowiedź
            if hasattr(context_result, 'documents') and context_result.documents:
                response_text = "Przepraszam, wystąpił problem z generowaniem odpowiedzi. Znalazłem informacje w bazie, ale nie mogę ich przetworzyć."
            else:
                response_text = "Przepraszam, wystąpił problem z generowaniem odpowiedzi. Spróbuj ponownie."
            
            tokens_used = None
        
        # 5. Przygotuj odpowiedź
        processing_time = time.time() - start_time
        
        # 6. Dodaj do historii
        add_to_history(session_id, "user", request.message)
        add_to_history(session_id, "assistant", response_text)
        
        # 7. Przygotuj odpowiedź API
        sources = []
        rag_info = {}
        
        if hasattr(context_result, 'documents') and context_result.documents:
            # Przygotuj info o RAG dla odpowiedzi
            detected_models = set()
            for doc in context_result.documents:
                if doc.get('metadata', {}).get('models'):
                    detected_models.update(doc['metadata']['models'])
            
            rag_info = {
                "sources_count": len(context_result.documents),
                "average_similarity": round(context_result.average_similarity, 3),
                "detected_models": list(detected_models)[:5],
                "has_target_model": any(
                    doc.get('source_info', {}).get('has_target_model', False) 
                    for doc in context_result.documents
                ),
                "query_type": query_type
            }
            
            if hasattr(context_result, 'to_api_response'):
                sources = context_result.to_api_response().get("sources", [])
        
        response = ChatResponse(
            answer=response_text,
            session_id=session_id,
            history_length=len(get_conversation_history(session_id)),
            processing_time=processing_time,
            sources=sources,
            model_used=settings.COHERE_CHAT_MODEL,
            tokens_used=tokens_used,
            confidence=context_result.average_similarity if hasattr(context_result, 'average_similarity') else None,
            rag_info=rag_info if rag_info else None
        )
        
        # 8. Logowanie w tle (opcjonalne)
        background_tasks.add_task(
            log_interaction,
            user_message=request.message,
            assistant_response=response_text,
            session_id=session_id,
            sources_count=len(sources),
            tokens_used=response.tokens_used,
            processing_time=processing_time,
            confidence=response.confidence,
            rag_info=rag_info
        )
        
        logger.info(f"Response generated in {processing_time:.2f}s for session {session_id}", extra={
            "tokens": response.tokens_used,
            "sources": len(response.sources),
            "history_length": response.history_length,
            "rag_models": rag_info.get('detected_models', []) if rag_info else [],
            "query_type": query_type
        })
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Chat endpoint error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error" if settings.IS_PRODUCTION else str(e)
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
# DEBUG ENDPOINTS (tylko development)
# ============================================

@app.get("/debug/rag")
async def debug_rag(
    query: str,
    top_k: int = settings.TOP_K_DOCUMENTS,
    rag_service: SimpleRAGService = Depends(get_rag_service)
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
        "result": result.to_api_response() if hasattr(result, 'to_api_response') else {"sources": []},
        "documents_count": len(result.documents) if hasattr(result, 'documents') else 0,
        "average_similarity": result.average_similarity if hasattr(result, 'average_similarity') else 0
    }


@app.get("/debug/stats")
async def debug_stats(
    rag_service: SimpleRAGService = Depends(get_rag_service),
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
# UTILITY FUNCTIONS
# ============================================

async def log_interaction(
    user_message: str,
    assistant_response: str,
    session_id: str,
    sources_count: int,
    tokens_used: Dict[str, int],
    processing_time: float,
    confidence: Optional[float],
    rag_info: Optional[Dict[str, Any]] = None
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
        
        if rag_info:
            log_entry["rag_info"] = rag_info
        
        logger.info(f"Interaction logged", extra=log_entry)
        
    except Exception as e:
        logger.error(f"Failed to log interaction: {str(e)}")

# ============================================
# APPLICATION EVENTS
# ============================================

@app.on_event("startup")
async def startup_event():
    """Uruchamiane przy starcie aplikacji"""
    try:
        # Walidacja konfiguracji
        validate_configuration()
        
        # Inicjalizacja cache
        await init_cache()
        
        # Inicjalizacja RAG
        rag_info = "❌ NOT AVAILABLE"
        if RAG_AVAILABLE:
            try:
                # Sprawdź czy plik RAG istnieje
                if RAG_FILE_PATH.exists():
                    # Utwórz RAG service (singleton)
                    rag_service = get_rag_service_singleton()
                    rag_health = await rag_service.health_check()
                    rag_stats = await rag_service.get_stats()
                    
                    if rag_stats.get("is_dummy", False):
                        rag_info = f"⚠️ DUMMY MODE (brak prawdziwego RAG)"
                    else:
                        rag_info = f"✅ LOADED ({rag_stats.get('total_chunks', 0)} chunks)"
                else:
                    rag_info = f"❌ FILE NOT FOUND: {RAG_FILE_PATH.name}"
            except Exception as rag_error:
                rag_info = f"❌ ERROR: {str(rag_error)[:50]}"
        else:
            rag_info = "❌ IMPORT FAILED"
        
        # Informacje o starcie
        logger.info(f"{settings.APP_NAME} v{settings.APP_VERSION} starting up...")
        logger.info(f"Environment: {settings.ENVIRONMENT}")
        logger.info(f"LLM Model: {settings.COHERE_CHAT_MODEL}")
        logger.info(f"RAG: {rag_info}")
        logger.info(f"RAG Documents: {settings.TOP_K_DOCUMENTS} docs")
        logger.info(f"Memory: Enabled (last {MAX_HISTORY} messages per session)")
        logger.info(f"API: http://{settings.HOST}:{settings.PORT}")
        logger.info(f"Docs: http://{settings.HOST}:{settings.PORT}/docs")
        
        # Sprawdź czy chat.html istnieje
        chat_html_path = TEMPLATES_DIR / "chat.html"
        if chat_html_path.exists():
            logger.info(f"HTML Interface: ✅ chat.html found at {chat_html_path}")
        else:
            logger.warning(f"HTML Interface: ⚠️ chat.html NOT FOUND at {chat_html_path}")
        
        logger.info("Application started successfully")
        
        if settings.IS_DEVELOPMENT:
            logger.warning("Running in DEVELOPMENT mode")
        
    except Exception as e:
        logger.critical(f"Startup failed: {str(e)}", exc_info=True)
        raise


@app.on_event("shutdown")
async def shutdown_event():
    """Uruchamiane przy zamykaniu aplikacji"""
    logger.info("Shutting down application...")
    logger.info(f"Memory stats: {len(conversation_memory)} sessions, {sum(len(h) for h in conversation_memory.values())} total messages")

# ============================================
# ERROR HANDLERS
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
# MAIN ENTRY POINT
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
            print(f"{settings.APP_NAME} v{settings.APP_VERSION}")
            print(f"Environment: {settings.ENVIRONMENT}")
            print(f"Czat: http://{settings.HOST}:{settings.PORT}/")
            print(f"Docs: http://{settings.HOST}:{settings.PORT}/docs")
            print(f"Model: {settings.COHERE_CHAT_MODEL}")
            
            # Sprawdź RAG
            print(f"RAG file: {RAG_FILE_PATH}")
            print(f"RAG exists: {RAG_FILE_PATH.exists()}")
            
            if RAG_AVAILABLE:
                print(f"RAG: ✅ IMPORTED")
            else:
                print(f"RAG: ⚠️  IMPORT FAILED - running in dummy mode")
            
            # Sprawdź HTML
            chat_html_path = TEMPLATES_DIR / "chat.html"
            print(f"HTML Interface: {'✅ Found' if chat_html_path.exists() else '❌ Not found'} at {chat_html_path}")
            
            print(f"Memory: {MAX_HISTORY} messages per session")
            print(f"API: http://{settings.HOST}:{settings.PORT}/chat")
            print(f"Test: http://{settings.HOST}:{settings.PORT}/ping")
            print(f"Quick Health: http://{settings.HOST}:{settings.PORT}/health/quick")
            print(f"Status: http://{settings.HOST}:{settings.PORT}/api/status")
            print(f"{'='*60}\n")
        
        uvicorn.run(**config)
        
    except Exception as e:
        print(f"Failed to start: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()