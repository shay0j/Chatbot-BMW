"""
Konfiguracja BMW Assistant z .env i dobrymi praktykami.
"""
import os
from pathlib import Path
from typing import List, Optional
from functools import lru_cache
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
import secrets


class Settings(BaseSettings):
    """Konfiguracja z .env i walidacją Pydantic."""
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
        env_prefix="",
    )
    
    # API KEYS (z .env)
    COHERE_API_KEY: Optional[str] = Field(default=None, description="Cohere API Key")
    FIRECRAWL_API_KEY: Optional[str] = Field(default=None, description="Firecrawl API Key")
    
    # APPLICATION SETTINGS
    APP_NAME: str = "BMW Assistant"
    APP_VERSION: str = "1.0.0"
    ENVIRONMENT: str = "development"
    
    # FastAPI
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    DEBUG: bool = True
    RELOAD: bool = True
    
    # Security
    SECRET_KEY: str = Field(
        default_factory=lambda: secrets.token_urlsafe(32) 
        if os.getenv("ENVIRONMENT") == "production" 
        else "dev-secret-key-change-in-production"
    )
    JWT_SECRET_KEY: str = Field(
        default_factory=lambda: secrets.token_urlsafe(32)
        if os.getenv("ENVIRONMENT") == "production"
        else "dev-jwt-secret-change-in-production"
    )
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    
    # CORS - jako string, potem parsujemy
    CORS_ORIGINS_STR: Optional[str] = Field(
        default="http://localhost:3000,http://localhost:8000,http://localhost:8080,http://127.0.0.1:3000,http://127.0.0.1:8000",
        description="CORS origins separated by commas"
    )
    
    # AI & RAG SETTINGS
    # LLM
    COHERE_CHAT_MODEL: str = Field(
        default="command-r7b-12-2024",
        description="Cohere chat model (command-r7b-12-2024, command-a-001, etc)"
    )
    LLM_TEMPERATURE: float = Field(default=0.7, ge=0.0, le=1.0)
    MAX_TOKENS: int = Field(default=2000, ge=100, le=4096)
    
    # RAG
    TOP_K_DOCUMENTS: int = Field(default=5, ge=1, le=20)
    SIMILARITY_THRESHOLD: float = Field(default=0.7, ge=0.0, le=1.0)
    CHROMA_DB_PATH: str = "./data/chroma_db"
    CHROMA_COLLECTION_NAME: str = "bmw_knowledge_base"
    
    # Embeddings
    COHERE_EMBED_MODEL: str = Field(
        default="embed-multilingual-v3.0",
        description="Cohere embedding model"
    )
    CHUNK_SIZE: int = Field(default=1000, ge=100, le=2000)
    CHUNK_OVERLAP: int = Field(default=200, ge=0, le=500)
    
    # PATHS & DIRECTORIES
    # Base directories
    BASE_DIR: Path = Path(__file__).parent.parent.parent
    OUTPUT_DIR: Path = Field(default_factory=lambda: Path("./data"))
    LOG_DIR: Path = Field(default_factory=lambda: Path("./logs"))
    TEMP_DIR: Path = Field(default_factory=lambda: Path("./tmp"))
    
    # WEB SCRAPING
    SOURCES: List[str] = [
        "https://www.bmw.pl",
        "https://www.bmw-zkmotors.pl/",
        "https://www.zkmotors.pl/",
        "https://www.mini.com.pl/"
    ]
    
    REQUEST_DELAY: float = Field(default=1.0, ge=0.1, le=10.0)
    MAX_DEPTH: int = Field(default=3, ge=1, le=10)
    MAX_PAGES: int = Field(default=1000, ge=1, le=10000)
    USER_AGENT: str = "Mozilla/5.0 (compatible; BMW-Assistant-Bot/1.0)"
    REQUEST_TIMEOUT: int = Field(default=30, ge=5, le=120)
    
    # DATABASE & CACHE
    REDIS_URL: str = "redis://localhost:6379/0"
    REDIS_CACHE_TTL: int = Field(default=3600, ge=60, le=86400)
    DATABASE_URL: str = "sqlite:///./data/bmw_assistant.db"
    
    # MONITORING & LOGGING
    LOG_LEVEL: str = Field(default="INFO")
    SENTRY_DSN: str = ""
    METRICS_PORT: int = Field(default=9090, ge=1024, le=65535)
    ENABLE_METRICS: bool = False
    
    # PERFORMANCE & RATE LIMITING
    RATE_LIMIT_PER_MINUTE: int = Field(default=60, ge=1, le=1000)
    API_TIMEOUT: int = Field(default=60, ge=10, le=300)
    MAX_WORKERS: int = Field(default=4, ge=1, le=os.cpu_count() or 4)
    EMBEDDING_BATCH_SIZE: int = Field(default=32, ge=1, le=256)
    
    # APPLICATION SPECIFIC
    DEFAULT_LANGUAGE: str = Field(default="pl")
    SUPPORTED_LANGUAGES: List[str] = ["pl", "en", "de"]
    MAX_CONVERSATION_HISTORY: int = Field(default=20, ge=1, le=100)
    ENABLE_STREAMING: bool = True
    
    # TESTING & DEVELOPMENT
    TEST_MODE: bool = False
    MOCK_EXTERNAL_APIS: bool = Field(
        default_factory=lambda: os.getenv("ENVIRONMENT") == "development"
    )
    TEST_DATABASE_URL: str = "sqlite:///./data/test.db"
    
    # SECURITY (production only)
    ENABLE_HTTPS: bool = False
    SSL_CERT_PATH: str = ""
    SSL_KEY_PATH: str = ""
    ADMIN_IPS: List[str] = ["127.0.0.1", "::1"]
    ENABLE_ADMIN_PANEL: bool = False
    
    # VALIDATORS
    @field_validator('LOG_LEVEL')
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if v.upper() not in valid_levels:
            raise ValueError(f"LOG_LEVEL must be one of: {', '.join(valid_levels)}")
        return v.upper()
    
    @field_validator('DEFAULT_LANGUAGE')
    @classmethod
    def validate_default_language(cls, v: str) -> str:
        valid_languages = ["pl", "en", "de"]
        if v.lower() not in valid_languages:
            raise ValueError(f"DEFAULT_LANGUAGE must be one of: {', '.join(valid_languages)}")
        return v.lower()
    
    @field_validator('COHERE_CHAT_MODEL')
    @classmethod
    def validate_chat_model(cls, v: str) -> str:
        # Aktualne modele Cohere (grudzień 2024 / styczeń 2026)
        valid_models = [
            "command-r7b-12-2024",  # Najnowszy model
            "command-a-001",        # Główny model płatny
            "command-a",            # Alternatywa
            "command-r-001",        # Nowy R model
            "command-light",        # Lekki (może deprecated)
            "command",              # Stary (deprecated)
            "command-r",            # Stary (deprecated)
            "command-r-plus",       # Stary (deprecated)
        ]
        
        # Log ostrzeżenie dla starych modeli, ale pozwól spróbować
        deprecated_models = ["command", "command-r", "command-r-plus", "command-light"]
        if v in deprecated_models:
            print(f"Warning: Model {v} might be deprecated (check Cohere docs)")
        
        # Sprawdź czy model nie jest pusty
        if not v or len(v) < 3:
            raise ValueError("COHERE_CHAT_MODEL cannot be empty")
        
        # Nie blokuj jeśli model nie jest na liście - pozwól Cohere API zweryfikować
        if v not in valid_models:
            print(f"Note: Model {v} not in predefined list, letting Cohere API validate it")
        
        return v
    
    @field_validator('COHERE_EMBED_MODEL')
    @classmethod
    def validate_embed_model(cls, v: str) -> str:
        if not v.startswith("embed-"):
            raise ValueError("COHERE_EMBED_MODEL must start with 'embed-'")
        return v
    
    @field_validator('FIRECRAWL_API_KEY')
    @classmethod
    def validate_firecrawl_key(cls, v: Optional[str]) -> Optional[str]:
        if v and not v.startswith("fc-"):
            raise ValueError("FIRECRAWL_API_KEY must start with 'fc-'")
        return v
    
    @field_validator('PORT')
    @classmethod
    def validate_port(cls, v: int) -> int:
        if not (1024 <= v <= 65535):
            raise ValueError("PORT must be between 1024 and 65535")
        return v
    
    # COMPUTED PROPERTIES
    @property
    def CORS_ORIGINS(self) -> List[str]:
        """Parse CORS origins from string."""
        if self.CORS_ORIGINS_STR:
            return [origin.strip() for origin in self.CORS_ORIGINS_STR.split(",") if origin.strip()]
        return [
            "http://localhost:3000",
            "http://localhost:8000", 
            "http://localhost:8080",
            "http://127.0.0.1:3000",
            "http://127.0.0.1:8000",
        ]
    
    @property
    def IS_DEVELOPMENT(self) -> bool:
        return self.ENVIRONMENT.lower() == "development"
    
    @property
    def IS_PRODUCTION(self) -> bool:
        return self.ENVIRONMENT.lower() == "production"
    
    @property
    def IS_TESTING(self) -> bool:
        return self.ENVIRONMENT.lower() == "testing"
    
    @property
    def API_URL(self) -> str:
        protocol = "https" if self.ENABLE_HTTPS else "http"
        return f"{protocol}://{self.HOST}:{self.PORT}"
    
    @property
    def ALLOW_ANONYMOUS_ACCESS(self) -> bool:
        return self.IS_DEVELOPMENT or self.TEST_MODE
    
    @property
    def USE_MOCK_LLM(self) -> bool:
        """Czy używać mock LLM zamiast prawdziwego API"""
        return (
            self.MOCK_EXTERNAL_APIS or 
            not self.COHERE_API_KEY or 
            self.IS_DEVELOPMENT and "test" in os.getenv("PYTEST_CURRENT_TEST", "")
        )
    
    # Subdirectories jako properties
    @property
    def CACHE_DIR(self) -> Path:
        return self.OUTPUT_DIR / "cache"
    
    @property
    def MODELS_DIR(self) -> Path:
        return self.OUTPUT_DIR / "models"
    
    @property
    def EMBEDDINGS_DIR(self) -> Path:
        return self.OUTPUT_DIR / "embeddings"
    
    @property
    def RAW_CRAWL_DIR(self) -> Path:
        return self.OUTPUT_DIR / "raw_pages"
    
    # File paths
    @property
    def FIRECRAWL_OUTPUT(self) -> Path:
        return self.OUTPUT_DIR / "firecrawl_all.json"
    
    @property
    def FAILED_URLS(self) -> Path:
        return self.OUTPUT_DIR / "failed_urls.json"
    
    # UTILITY METHODS
    def ensure_dirs(self) -> List[Path]:
        """Tworzy wszystkie potrzebne katalogi."""
        dirs_to_create = [
            self.OUTPUT_DIR,
            self.LOG_DIR,
            self.TEMP_DIR,
            self.CACHE_DIR,
            self.MODELS_DIR,
            self.EMBEDDINGS_DIR,
            self.RAW_CRAWL_DIR,
            Path(self.CHROMA_DB_PATH),
        ]
        
        created_dirs = []
        for directory in dirs_to_create:
            if isinstance(directory, str):
                directory = Path(directory)
            directory.mkdir(parents=True, exist_ok=True)
            created_dirs.append(directory)
        
        return created_dirs
    
    def validate_config(self) -> List[str]:
        """Waliduje konfigurację, zwraca listę błędów."""
        errors = []
        
        # Sprawdź klucze API tylko jeśli nie używamy mocka
        if not self.USE_MOCK_LLM:
            if not self.COHERE_API_KEY or len(self.COHERE_API_KEY) < 10:
                errors.append("COHERE_API_KEY jest wymagany lub nieprawidłowy")
            
            if not self.FIRECRAWL_API_KEY:
                errors.append("FIRECRAWL_API_KEY jest wymagany")
        
        return errors
    
    def get_safe_summary(self) -> dict:
        """Zwraca bezpieczne podsumowanie konfiguracji (bez kluczy API)."""
        return {
            "app": {
                "name": self.APP_NAME,
                "version": self.APP_VERSION,
                "environment": self.ENVIRONMENT,
                "debug": self.DEBUG,
            },
            "api": {
                "host": self.HOST,
                "port": self.PORT,
                "url": self.API_URL,
                "cors_origins": self.CORS_ORIGINS[:3],
            },
            "ai": {
                "llm_model": self.COHERE_CHAT_MODEL,
                "embedding_model": self.COHERE_EMBED_MODEL,
                "temperature": self.LLM_TEMPERATURE,
                "use_mock": self.USE_MOCK_LLM,
            },
            "rag": {
                "top_k_documents": self.TOP_K_DOCUMENTS,
                "similarity_threshold": self.SIMILARITY_THRESHOLD,
                "collection_name": self.CHROMA_COLLECTION_NAME,
            },
            "security": {
                "https_enabled": self.ENABLE_HTTPS,
                "anonymous_access": self.ALLOW_ANONYMOUS_ACCESS,
            }
        }
    
    def __str__(self) -> str:
        summary = self.get_safe_summary()
        lines = [f"{self.APP_NAME} v{self.APP_VERSION}"]
        lines.append(f"Environment: {self.ENVIRONMENT}")
        lines.append(f"API: {self.API_URL}")
        lines.append(f"LLM: {self.COHERE_CHAT_MODEL} (mock: {self.USE_MOCK_LLM})")
        lines.append(f"Embeddings: {self.COHERE_EMBED_MODEL}")
        return "\n".join(lines)


@lru_cache()
def get_settings() -> Settings:
    """Zwraca cached instancję Settings."""
    settings = Settings()
    
    # Automatycznie tworzymy katalogi
    if settings.IS_DEVELOPMENT:
        settings.ensure_dirs()
    
    # Walidacja
    errors = settings.validate_config()
    if errors:
        print("Configuration errors:")
        for error in errors:
            print(f"   - {error}")
        
        if settings.IS_PRODUCTION:
            raise ValueError(f"Configuration errors: {', '.join(errors)}")
        else:
            print("Continuing in development mode despite errors...")
    
    return settings


# Globalna instancja dla łatwego importu
settings = get_settings()


def validate_configuration():
    """
    Kompatybilność wsteczna z oryginalną funkcją.
    Teraz walidacja jest wbudowana w Settings.
    """
    print("Validating configuration...")
    
    # Stwórz katalogi
    created_dirs = settings.ensure_dirs()
    for directory in created_dirs:
        print(f"Created/verified: {directory}")
    
    # Sprawdź ważne ustawienia
    if settings.SECRET_KEY == "dev-secret-key-change-in-production":
        print("WARNING: Using default SECRET_KEY - change in production!")
    
    if settings.JWT_SECRET_KEY == "dev-jwt-secret-change-in-production":
        print("WARNING: Using default JWT_SECRET_KEY - change in production!")
    
    print("Configuration validated successfully!")
    print(f"   App: {settings.APP_NAME} v{settings.APP_VERSION}")
    print(f"   Environment: {settings.ENVIRONMENT}")
    print(f"   API: {settings.API_URL}")
    print(f"   LLM Model: {settings.COHERE_CHAT_MODEL}")
    print(f"   RAG: {settings.TOP_K_DOCUMENTS} docs, threshold: {settings.SIMILARITY_THRESHOLD}")
    print(f"   Default language: {settings.DEFAULT_LANGUAGE}")
    
    return True


def setup_environment():
    """Konfiguruje środowisko na podstawie settings."""
    # Ustaw zmienne środowiskowe
    os.environ["COHERE_API_KEY"] = settings.COHERE_API_KEY or ""
    os.environ["FIRECRAWL_API_KEY"] = settings.FIRECRAWL_API_KEY or ""
    
    # Dla FastAPI
    os.environ["HOST"] = settings.HOST
    os.environ["PORT"] = str(settings.PORT)


def print_config_summary():
    """Wyświetla podsumowanie konfiguracji."""
    print(str(settings))
    
    if settings.USE_MOCK_LLM:
        print("Using MOCK LLM (no API calls)")
    if settings.IS_DEVELOPMENT:
        print("Development mode: DEBUG=True, RELOAD=True")
    if settings.IS_PRODUCTION and settings.DEBUG:
        print("WARNING: DEBUG=True in production!")


# TEST - uruchom bezpośrednio aby przetestować
if __name__ == "__main__":
    print("Testing configuration...")
    
    try:
        # Pobierz i wyświetl konfigurację
        config = get_settings()
        print_config_summary()
        
        # Pokaż bezpieczne podsumowanie
        print("\nSafe config summary:")
        import json
        print(json.dumps(config.get_safe_summary(), indent=2, ensure_ascii=False))
        
        print("\nConfiguration OK!")
        
    except Exception as e:
        print(f"Configuration failed: {e}")
        import traceback
        traceback.print_exc()
        exit(1)