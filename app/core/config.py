"""
Konfiguracja BMW Assistant z .env - DOSTOSOWANA DO RAG I CHROMADB
"""
import os
from pathlib import Path
from typing import List, Optional
from functools import lru_cache
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
import secrets


class Settings(BaseSettings):
    """Konfiguracja z .env - dostosowana do twojego projektu"""
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
        env_prefix="",
    )
    
    # ========== API KEYS (z .env) ==========
    COHERE_API_KEY: str = Field(default="", description="Cohere API Key")
    FIRECRAWL_API_KEY: Optional[str] = Field(default=None, description="Firecrawl API Key")
    
    # ========== APPLICATION SETTINGS ==========
    APP_NAME: str = "BMW Assistant"
    APP_VERSION: str = "1.0.0"
    ENVIRONMENT: str = "development"
    
    # FastAPI
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    DEBUG: bool = True
    RELOAD: bool = Field(default=True)
    
    # CORS - JAKO STRING, parsujemy w validatorze
    CORS_ORIGINS_STR: str = Field(
        default="http://localhost:3000,http://localhost:8000,http://localhost:8080",
        description="CORS origins as comma-separated string"
    )
    
    # ========== AI & RAG SETTINGS ==========
    # LLM - dopasowane do twojego .env
    COHERE_CHAT_MODEL: str = Field(
        default="command-r7b-12-2024",
        description="Cohere chat model"
    )
    LLM_TEMPERATURE: float = Field(default=0.7, ge=0.0, le=1.0)
    MAX_TOKENS: int = Field(default=1000, ge=100, le=2000)
    
    # RAG - KLUCZOWE ZMIANY DLA CHROMADB!
    TOP_K_DOCUMENTS: int = Field(default=3, ge=1, le=10)
    SIMILARITY_THRESHOLD: float = Field(default=0.6, ge=0.0, le=1.0)
    
    # WAŻNE: ta sama ścieżka co w twoim embedderze!
    CHROMA_DB_PATH: str = Field(
        default=r"C:\Users\hellb\Documents\Chatbot_BMW\RAG\chroma_db_working",
        description="Path to ChromaDB database"
    )
    
    # WAŻNE: ta sama nazwa kolekcji co w twoim embedderze!
    CHROMA_COLLECTION_NAME: str = Field(
        default="bmw_docs",
        description="ChromaDB collection name"
    )
    
    # Embeddings - dopasowane do twojego .env
    COHERE_EMBED_MODEL: str = Field(
        default="embed-multilingual-v3.0",
        description="Cohere embedding model"
    )
    
    # ========== PATHS & DIRECTORIES ==========
    # Base directories
    BASE_DIR: Path = Path(__file__).parent.parent.parent
    
    # WAŻNE: Zmieniamy OUTPUT_DIR na RAG folder zamiast ./data
    OUTPUT_DIR: Path = Field(
        default_factory=lambda: Path(r"C:\Users\hellb\Documents\Chatbot_BMW\RAG\output"),
        description="RAG output directory"
    )
    
    LOG_DIR: Path = Field(default_factory=lambda: Path("./logs"))
    TEMP_DIR: Path = Field(default_factory=lambda: Path("./tmp"))
    
    # ========== LOGGING ==========
    LOG_LEVEL: str = Field(default="INFO")
    
    # ========== APPLICATION SPECIFIC ==========
    DEFAULT_LANGUAGE: str = Field(default="pl")
    SUPPORTED_LANGUAGES: List[str] = ["pl", "en", "de"]
    MAX_CONVERSATION_HISTORY: int = Field(default=10, ge=1, le=100)
    ENABLE_STREAMING: bool = True
    
    # ========== SECURITY ==========
    SECRET_KEY: str = Field(
        default_factory=lambda: secrets.token_urlsafe(32)
    )
    
    # ========== TESTING - DODAJEMY ==========
    TEST_MODE: bool = Field(default=False, description="Czy aplikacja działa w trybie testowym")
    
    # ========== VALIDATORS ==========
    @field_validator('COHERE_API_KEY')
    @classmethod
    def validate_cohere_api_key(cls, v: str) -> str:
        if not v or len(v) < 10:
            raise ValueError("COHERE_API_KEY is required")
        return v
    
    @field_validator('LOG_LEVEL')
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if v.upper() not in valid_levels:
            raise ValueError(f"LOG_LEVEL must be one of: {', '.join(valid_levels)}")
        return v.upper()
    
    @field_validator('CHROMA_DB_PATH')
    @classmethod
    def validate_chroma_path(cls, v: str) -> str:
        # Sprawdź czy ścieżka istnieje
        path = Path(v)
        if not path.exists():
            print(f"⚠️  Warning: ChromaDB path does not exist: {v}")
            # W development tworzymy katalog
            if os.getenv("ENVIRONMENT") == "development":
                path.mkdir(parents=True, exist_ok=True)
                print(f"Created ChromaDB directory: {path}")
        return v
    
    @field_validator('CHROMA_COLLECTION_NAME')
    @classmethod
    def validate_collection_name(cls, v: str) -> str:
        if not v or len(v.strip()) == 0:
            raise ValueError("CHROMA_COLLECTION_NAME cannot be empty")
        return v.strip()
    
    # ========== COMPUTED PROPERTIES ==========
    @property
    def IS_DEVELOPMENT(self) -> bool:
        return self.ENVIRONMENT.lower() == "development"
    
    @property
    def IS_PRODUCTION(self) -> bool:
        return self.ENVIRONMENT.lower() == "production"
    
    @property
    def API_URL(self) -> str:
        return f"http://{self.HOST}:{self.PORT}"
    
    @property
    def ALLOW_ANONYMOUS_ACCESS(self) -> bool:
        return self.IS_DEVELOPMENT
    
    @property
    def USE_MOCK_LLM(self) -> bool:
        """Czy używać mock LLM (tylko dla testów)"""
        return self.IS_DEVELOPMENT and (self.TEST_MODE or "test" in os.getenv("PYTEST_CURRENT_TEST", ""))
    
    @property
    def CORS_ORIGINS(self) -> List[str]:
        """Parsuje CORS_ORIGINS_STR do listy"""
        if not self.CORS_ORIGINS_STR:
            return ["http://localhost:3000", "http://localhost:8000", "http://localhost:8080"]
        
        origins = []
        for origin in self.CORS_ORIGINS_STR.split(","):
            origin = origin.strip()
            if origin:
                origins.append(origin)
        
        return origins
    
    # ========== PATHS RAG ==========
    @property
    def CHROMA_DB_PATH_OBJ(self) -> Path:
        """Path jako Path object"""
        return Path(self.CHROMA_DB_PATH)
    
    @property
    def RAG_OUTPUT_DIR(self) -> Path:
        """Folder z danymi RAG"""
        return self.OUTPUT_DIR
    
    @property
    def CHROMA_COLLECTION_PATH(self) -> Path:
        """Pełna ścieżka do kolekcji ChromaDB"""
        return self.CHROMA_DB_PATH_OBJ / self.CHROMA_COLLECTION_NAME
    
    # ========== UTILITY METHODS ==========
    def ensure_dirs(self) -> List[Path]:
        """Tworzy wszystkie potrzebne katalogi."""
        dirs_to_create = [
            self.LOG_DIR,
            self.TEMP_DIR,
            self.CHROMA_DB_PATH_OBJ,
            self.RAG_OUTPUT_DIR,
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
        
        # Sprawdź klucze API
        if not self.COHERE_API_KEY:
            errors.append("COHERE_API_KEY jest wymagany")
        
        # Sprawdź czy ChromaDB path istnieje
        chroma_path = Path(self.CHROMA_DB_PATH)
        if not chroma_path.exists():
            errors.append(f"ChromaDB path nie istnieje: {self.CHROMA_DB_PATH}")
            print(f"Hint: Sprawdź czy folder istnieje lub uruchom embedder")
        
        # Sprawdź port
        if not (1024 <= self.PORT <= 65535):
            errors.append(f"Nieprawidłowy port: {self.PORT}")
        
        return errors
    
    def get_safe_summary(self) -> dict:
        """Zwraca bezpieczne podsumowanie konfiguracji."""
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
                "cors_origins": self.CORS_ORIGINS,
            },
            "ai": {
                "llm_model": self.COHERE_CHAT_MODEL,
                "embedding_model": self.COHERE_EMBED_MODEL,
                "temperature": self.LLM_TEMPERATURE,
            },
            "rag": {
                "top_k_documents": self.TOP_K_DOCUMENTS,
                "similarity_threshold": self.SIMILARITY_THRESHOLD,
                "chroma_path": str(self.CHROMA_DB_PATH),
                "collection_name": self.CHROMA_COLLECTION_NAME,
            }
        }
    
    def __str__(self) -> str:
        return f"""
{self.APP_NAME} v{self.APP_VERSION}
  Environment: {self.ENVIRONMENT}
  API: {self.API_URL}
  LLM: {self.COHERE_CHAT_MODEL}
  Embeddings: {self.COHERE_EMBED_MODEL}
  RAG: {self.TOP_K_DOCUMENTS} docs, threshold: {self.SIMILARITY_THRESHOLD}
  ChromaDB: {self.CHROMA_DB_PATH}
  Collection: {self.CHROMA_COLLECTION_NAME}
        """.strip()


@lru_cache()
def get_settings() -> Settings:
    """Zwraca cached instancję Settings."""
    settings = Settings()
    
    # Automatycznie tworzymy katalogi
    if settings.IS_DEVELOPMENT:
        try:
            created_dirs = settings.ensure_dirs()
            print(f"Created/verified directories: {[str(d) for d in created_dirs]}")
        except Exception as e:
            print(f"Warning: Could not create directories: {e}")
    
    # Walidacja
    errors = settings.validate_config()
    if errors:
        print("❌ Configuration errors:")
        for error in errors:
            print(f"   - {error}")
        
        if settings.IS_PRODUCTION:
            raise ValueError(f"Configuration errors: {', '.join(errors)}")
        else:
            print("⚠️  Continuing in development mode despite errors...")
    
    return settings


# Globalna instancja dla łatwego importu
settings = get_settings()


def validate_configuration():
    """
    Kompatybilność wsteczna z oryginalną funkcją.
    """
    print("Validating configuration...")
    
    # Stwórz katalogi
    try:
        created_dirs = settings.ensure_dirs()
        for directory in created_dirs:
            print(f"Created/verified: {directory}")
    except Exception as e:
        print(f"Warning: Could not create directories: {e}")
    
    # Wyświetl podsumowanie
    print("\n" + "="*60)
    print(str(settings))
    print("="*60 + "\n")
    
    # Sprawdź ChromaDB
    chroma_path = Path(settings.CHROMA_DB_PATH)
    if chroma_path.exists():
        print(f"✅ ChromaDB path exists: {chroma_path}")
        
        # Sprawdź czy są pliki ChromaDB
        chroma_files = list(chroma_path.glob("*"))
        if chroma_files:
            print(f"   Found {len(chroma_files)} files in ChromaDB")
        else:
            print(f"   Warning: ChromaDB directory is empty")
            print(f"   Hint: Run your embedder to populate ChromaDB")
    else:
        print(f"❌ ChromaDB path does not exist: {chroma_path}")
        print(f"   Hint: Run your embedder first")
    
    print("\nConfiguration validated successfully!")
    return True


def setup_environment():
    """Konfiguruje środowisko."""
    # Ustaw zmienne środowiskowe dla kompatybilności
    os.environ["COHERE_API_KEY"] = settings.COHERE_API_KEY
    if settings.FIRECRAWL_API_KEY:
        os.environ["FIRECRAWL_API_KEY"] = settings.FIRECRAWL_API_KEY


def print_config_summary():
    """Wyświetla podsumowanie konfiguracji."""
    print(str(settings))
    
    if settings.IS_DEVELOPMENT:
        print("Development mode: DEBUG=True, RELOAD=True")


# TEST
if __name__ == "__main__":
    print("Testing configuration...\n")
    
    try:
        # Pobierz konfigurację
        config = get_settings()
        print_config_summary()
        
        # Wyświetl CORS origins
        print(f"\nCORS Origins: {config.CORS_ORIGINS}")
        
        # Walidacja
        validate_configuration()
        
        print("\n✅ Configuration OK!")
        
    except Exception as e:
        print(f"\n❌ Configuration failed: {e}")
        import traceback
        traceback.print_exc()
        exit(1)