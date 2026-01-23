"""
Konfiguracja BMW Assistant - prosta i działająca wersja.
"""
import os
from pathlib import Path
from functools import lru_cache


class Settings:
    """Uproszczona konfiguracja - działa na 100%"""
    
    # ============================================
    # 🔐 API KEYS (HARDCODED - zmień później na .env)
    # ============================================
    
    COHERE_API_KEY = "hftDskvdWUY58HQvXLUtcqYtxI0WcVAU7t6NC2mp"
    FIRECRAWL_API_KEY = "fc-6391e2adfb514c0098495a167319437c"
    
    # ============================================
    # 🚀 APPLICATION SETTINGS
    # ============================================
    
    APP_NAME = "BMW Assistant"
    APP_VERSION = "1.0.0"
    ENVIRONMENT = "development"  # development, production
    
    # FastAPI
    HOST = "0.0.0.0"
    PORT = 8000
    DEBUG = True
    RELOAD = True
    
    # Security
    SECRET_KEY = "change-this-to-secure-random-key-in-production"
    JWT_SECRET_KEY = "another-secure-key-for-jwt-tokens"
    ACCESS_TOKEN_EXPIRE_MINUTES = 30
    
    # CORS
    CORS_ORIGINS = ["http://localhost:3000", "http://localhost:8000", "http://localhost:8080"]
    
    # ============================================
    # 🧠 AI & RAG SETTINGS
    # ============================================
    
    # LLM
    COHERE_CHAT_MODEL = "command-r"
    LLM_TEMPERATURE = 0.7
    MAX_TOKENS = 2000
    
    # RAG
    TOP_K_DOCUMENTS = 5
    SIMILARITY_THRESHOLD = 0.7
    CHROMA_DB_PATH = "./data/chroma_db"
    CHROMA_COLLECTION_NAME = "bmw_knowledge_base"
    
    # Embeddings
    COHERE_EMBED_MODEL = "embed-multilingual-v3.0"
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 200
    
    # ============================================
    # 📁 PATHS & DIRECTORIES
    # ============================================
    
    OUTPUT_DIR = "./data"
    LOG_DIR = "./logs"
    TEMP_DIR = "./tmp"
    CACHE_DIR = "./data/cache"
    MODELS_DIR = "./data/models"
    EMBEDDINGS_DIR = "./data/embeddings"
    RAW_CRAWL_DIR = "./data/raw_pages"
    
    # File paths
    FIRECRAWL_OUTPUT = "./data/firecrawl_all.json"
    FAILED_URLS = "./data/failed_urls.json"
    
    # ============================================
    # 🌐 WEB SCRAPING
    # ============================================
    
    SOURCES = [
        "https://www.bmw.pl",
        "https://www.bmw-zkmotors.pl/",
        "https://www.zkmotors.pl/",
        "https://www.mini.com.pl/"
    ]
    
    REQUEST_DELAY = 1.0
    MAX_DEPTH = 3
    MAX_PAGES = 1000
    USER_AGENT = "Mozilla/5.0 (compatible; BMW-Assistant-Bot/1.0)"
    REQUEST_TIMEOUT = 30
    
    # ============================================
    # 💾 DATABASE & CACHE
    # ============================================
    
    REDIS_URL = "redis://localhost:6379/0"
    REDIS_CACHE_TTL = 3600
    DATABASE_URL = "sqlite:///./data/bmw_assistant.db"
    
    # ============================================
    # 📊 MONITORING & LOGGING
    # ============================================
    
    LOG_LEVEL = "INFO"
    SENTRY_DSN = ""
    METRICS_PORT = 9090
    ENABLE_METRICS = True
    
    # ============================================
    # 🔧 PERFORMANCE & RATE LIMITING
    # ============================================
    
    RATE_LIMIT_PER_MINUTE = 60
    API_TIMEOUT = 60
    MAX_WORKERS = 4
    EMBEDDING_BATCH_SIZE = 32
    
    # ============================================
    # 🎯 APPLICATION SPECIFIC
    # ============================================
    
    DEFAULT_LANGUAGE = "pl"
    SUPPORTED_LANGUAGES = ["pl", "en", "de"]
    MAX_CONVERSATION_HISTORY = 20
    ENABLE_STREAMING = True
    
    # ============================================
    # 🧪 TESTING & DEVELOPMENT
    # ============================================
    
    TEST_MODE = False
    MOCK_EXTERNAL_APIS = False
    TEST_DATABASE_URL = "sqlite:///./data/test.db"
    
    # ============================================
    # 🔐 SECURITY (production only)
    # ============================================
    
    ENABLE_HTTPS = False
    SSL_CERT_PATH = ""
    SSL_KEY_PATH = ""
    ADMIN_IPS = ["127.0.0.1"]
    ENABLE_ADMIN_PANEL = False
    
    # ============================================
    # 🎯 COMPUTED PROPERTIES
    # ============================================
    
    @property
    def IS_DEVELOPMENT(self):
        return self.ENVIRONMENT == "development"
    
    @property
    def IS_PRODUCTION(self):
        return self.ENVIRONMENT == "production"
    
    @property
    def API_URL(self):
        protocol = "https" if self.ENABLE_HTTPS else "http"
        return f"{protocol}://{self.HOST}:{self.PORT}"
    
    @property
    def CORS_ORIGINS_STR(self):
        return self.CORS_ORIGINS
    
    @property
    def SUPPORTED_LANGUAGES_STR(self):
        return ",".join(self.SUPPORTED_LANGUAGES)


@lru_cache()
def get_settings() -> Settings:
    """Zwraca cached instancję Settings."""
    return Settings()


# Globalna instancja dla łatwego importu
settings = get_settings()


def validate_configuration():
    """
    Waliduje konfigurację i tworzy potrzebne katalogi.
    
    Returns:
        bool: True jeśli wszystko OK
        
    Raises:
        ValueError: Jeśli są błędy konfiguracji
    """
    print("🔧 Validating configuration...")
    
    errors = []
    
    # 1. Sprawdź kluczowe API keys
    if not settings.COHERE_API_KEY:
        errors.append("❌ COHERE_API_KEY is not set")
    
    if not settings.FIRECRAWL_API_KEY:
        errors.append("❌ FIRECRAWL_API_KEY is not set")
    
    # 2. Sprawdź ważne ustawienia
    if settings.SECRET_KEY == "change-this-to-secure-random-key-in-production":
        print("⚠️  WARNING: Using default SECRET_KEY - change in production!")
    
    if settings.JWT_SECRET_KEY == "another-secure-key-for-jwt-tokens":
        print("⚠️  WARNING: Using default JWT_SECRET_KEY - change in production!")
    
    # 3. Stwórz katalogi
    directories = [
        settings.OUTPUT_DIR,
        settings.CHROMA_DB_PATH,
        settings.LOG_DIR,
        settings.TEMP_DIR,
        settings.CACHE_DIR,
        settings.MODELS_DIR,
        settings.EMBEDDINGS_DIR,
        settings.RAW_CRAWL_DIR,
    ]
    
    for directory in directories:
        try:
            Path(directory).mkdir(parents=True, exist_ok=True)
            print(f"📁 Created/verified: {directory}")
        except Exception as e:
            errors.append(f"Cannot create directory {directory}: {str(e)}")
    
    # 4. Sprawdź port
    if not (1 <= settings.PORT <= 65535):
        errors.append(f"Invalid port: {settings.PORT}")
    
    # 5. Sprawdź temperature
    if not (0.0 <= settings.LLM_TEMPERATURE <= 1.0):
        errors.append(f"Invalid LLM_TEMPERATURE: {settings.LLM_TEMPERATURE} (must be 0.0-1.0)")
    
    # 6. Sprawdź similarity threshold
    if not (0.0 <= settings.SIMILARITY_THRESHOLD <= 1.0):
        errors.append(f"Invalid SIMILARITY_THRESHOLD: {settings.SIMILARITY_THRESHOLD} (must be 0.0-1.0)")
    
    # 7. Jeśli są błędy, rzuć wyjątek
    if errors:
        error_msg = "Configuration errors:\n" + "\n".join(errors)
        raise ValueError(error_msg)
    
    print("✅ Configuration validated successfully!")
    print(f"   App: {settings.APP_NAME} v{settings.APP_VERSION}")
    print(f"   Environment: {settings.ENVIRONMENT}")
    print(f"   API: {settings.API_URL}")
    print(f"   LLM Model: {settings.COHERE_CHAT_MODEL}")
    print(f"   RAG: {settings.TOP_K_DOCUMENTS} docs, threshold: {settings.SIMILARITY_THRESHOLD}")
    print(f"   Default language: {settings.DEFAULT_LANGUAGE}")
    
    return True


def setup_environment():
    """Ustawia zmienne środowiskowe z konfiguracji."""
    # Ustaw kluczowe zmienne środowiskowe
    os.environ["COHERE_API_KEY"] = settings.COHERE_API_KEY
    os.environ["FIRECRAWL_API_KEY"] = settings.FIRECRAWL_API_KEY
    os.environ["APP_ENV"] = settings.ENVIRONMENT
    
    # Ustaw dla FastAPI
    os.environ["HOST"] = settings.HOST
    os.environ["PORT"] = str(settings.PORT)
    
    print("🌍 Environment variables set from configuration")


def get_config_summary() -> dict:
    """Zwraca podsumowanie konfiguracji (bez wrażliwych danych)."""
    return {
        "app": {
            "name": settings.APP_NAME,
            "version": settings.APP_VERSION,
            "environment": settings.ENVIRONMENT,
        },
        "api": {
            "host": settings.HOST,
            "port": settings.PORT,
            "url": settings.API_URL,
        },
        "ai": {
            "llm_model": settings.COHERE_CHAT_MODEL,
            "embedding_model": settings.COHERE_EMBED_MODEL,
            "temperature": settings.LLM_TEMPERATURE,
        },
        "rag": {
            "top_k_documents": settings.TOP_K_DOCUMENTS,
            "similarity_threshold": settings.SIMILARITY_THRESHOLD,
            "chroma_db_path": settings.CHROMA_DB_PATH,
        },
        "paths": {
            "output_dir": settings.OUTPUT_DIR,
            "log_dir": settings.LOG_DIR,
            "chroma_db": settings.CHROMA_DB_PATH,
        }
    }


# ============================================
# 🧪 TEST - uruchom bezpośrednio aby przetestować
# ============================================

if __name__ == "__main__":
    print("🧪 Testing configuration module...")
    try:
        validate_configuration()
        print("✅ All tests passed!")
        
        # Pokaż podsumowanie konfiguracji
        print("\n📋 Configuration Summary:")
        summary = get_config_summary()
        
        for category, values in summary.items():
            print(f"\n{category.upper()}:")
            for key, value in values.items():
                print(f"  {key}: {value}")
        
        # Pokaż ważne uwagi
        print("\n⚠️  Important notes:")
        if settings.IS_PRODUCTION:
            print("  - Running in PRODUCTION mode")
            if settings.DEBUG:
                print("  - WARNING: DEBUG is True in production!")
        else:
            print("  - Running in DEVELOPMENT mode")
        
        if settings.SECRET_KEY.startswith("change-this"):
            print("  - WARNING: Using default SECRET_KEY")
        
        print(f"\n🎯 Ready to run at: {settings.API_URL}")
        
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        exit(1)