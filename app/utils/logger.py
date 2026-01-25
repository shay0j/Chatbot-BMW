"""
Zaawansowany system logowania dla BMW Assistant.
Wykorzystuje loguru dla strukturalnego logowania.
"""
import sys
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional

from loguru import logger

# ============================================
# 🎯 CONFIGURATION
# ============================================

# Proste enums bez pydantic
LOG_LEVELS = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
LOG_FORMATS = ["simple", "json", "detailed"]


# ============================================
# 🛠️ LOGGER SETUP
# ============================================

def setup_logger(
    name: str = "bmw_assistant",
    level: str = "INFO",
    log_format: str = "detailed",
    log_dir: Path = None,
    is_production: bool = False,
    is_development: bool = True
) -> logger:
    """
    Konfiguruje i zwraca logger.
    
    Args:
        name: Nazwa loggera
        level: Poziom logowania
        log_format: Format logów
        log_dir: Katalog dla plików logów
        is_production: Czy produkcja
        is_development: Czy development
    
    Returns:
        Skonfigurowany logger
    """
    # Walidacja poziomu logowania
    if level not in LOG_LEVELS:
        level = "INFO"
        print(f"⚠️  Invalid log level, using INFO instead")
    
    # Walidacja formatu
    if log_format not in LOG_FORMATS:
        log_format = "json" if is_production else "detailed"
        print(f"⚠️  Invalid log format, using {log_format} instead")
    
    # Upewnij się że katalog istnieje
    if log_dir:
        try:
            log_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            print(f"⚠️  Could not create log directory: {e}")
            log_dir = None
    
    # Usuń domyślnego handlera
    logger.remove()
    
    # Konfiguracja formatu
    if log_format == "json":
        format_string = (
            '{{"time": "{time:YYYY-MM-DD HH:mm:ss.SSS}", '
            '"level": "{level}", '
            '"name": "{name}", '
            '"function": "{function}", '
            '"line": {line}, '
            '"message": "{message}"}}'
        )
    elif log_format == "detailed":
        format_string = (
            "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
            "<level>{message}</level>"
        )
    else:  # simple
        format_string = (
            "{time:YYYY-MM-DD HH:mm:ss} | "
            "{level: <8} | "
            "{name}.{function}:{line} | "
            "{message}"
        )
    
    # Handler dla konsoli
    logger.add(
        sys.stdout,
        format=format_string,
        level=level,
        colorize=True,
        backtrace=True,
        diagnose=is_development
    )
    
    # Handler dla pliku tylko jeśli mamy katalog
    if log_dir:
        try:
            log_file = log_dir / f"{name}.log"
            logger.add(
                str(log_file),
                format=format_string,
                level=level,
                rotation="10 MB",  # Mniejszy limit dla testów
                retention="7 days",
                compression="zip",
                backtrace=True,
                diagnose=is_development
            )
            
            # Handler dla błędów
            error_log_file = log_dir / f"{name}_errors.log"
            logger.add(
                str(error_log_file),
                format=format_string,
                level="ERROR",
                rotation="5 MB",
                retention="30 days",
                compression="zip",
                backtrace=True,
                diagnose=True
            )
        except Exception as e:
            print(f"⚠️  Could not setup file logging: {e}")
    
    return logger.bind(name=name)


# ============================================
# 🎯 STRUCTURED LOGGING
# ============================================

class StructuredLogger:
    """Logger do strukturalnego logowania"""
    
    def __init__(self, logger_instance=None):
        self.logger = logger_instance or logger
    
    def debug(self, message: str, **kwargs):
        """Log debug z dodatkowymi polami"""
        self.logger.debug(message, **kwargs)
    
    def info(self, message: str, **kwargs):
        """Log info z dodatkowymi polami"""
        self.logger.info(message, **kwargs)
    
    def warning(self, message: str, **kwargs):
        """Log warning z dodatkowymi polami"""
        self.logger.warning(message, **kwargs)
    
    def error(self, message: str, **kwargs):
        """Log error z dodatkowymi polami"""
        self.logger.error(message, **kwargs)
    
    def critical(self, message: str, **kwargs):
        """Log critical z dodatkowymi polami"""
        self.logger.critical(message, **kwargs)
    
    def log_chat_interaction(
        self,
        conversation_id: str,
        user_message: str,
        assistant_response: str,
        processing_time: float,
        sources_count: int = 0,
        **kwargs
    ):
        """Loguje interakcję chat"""
        self.info(
            f"Chat: {conversation_id}",
            conversation_id=conversation_id,
            user_message=user_message[:100],
            response=assistant_response[:100],
            processing_time=processing_time,
            sources_count=sources_count,
            **kwargs
        )
    
    def log_llm_call(
        self,
        model: str,
        processing_time: float,
        **kwargs
    ):
        """Loguje wywołanie LLM"""
        self.debug(
            f"LLM: {model}",
            model=model,
            processing_time=processing_time,
            **kwargs
        )
    
    def log_rag_retrieval(
        self,
        query: str,
        documents_count: int,
        confidence: float,
        processing_time: float,
        **kwargs
    ):
        """Loguje wyszukiwanie w RAG"""
        self.debug(
            f"RAG: {query[:50]}...",
            query=query[:100],
            documents_count=documents_count,
            confidence=confidence,
            processing_time=processing_time,
            **kwargs
        )


# ============================================
# 🔧 LOGGER MIDDLEWARE
# ============================================

def log_exceptions(func):
    """
    Dekorator do logowania wyjątków.
    """
    import functools
    
    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        try:
            return await func(*args, **kwargs)
        except Exception as e:
            logger.error(
                f"Exception in {func.__name__}: {str(e)}",
                function=func.__name__,
                exception_type=e.__class__.__name__
            )
            raise
    
    return wrapper


class PerformanceLogger:
    """Logger wydajności - uproszczony"""
    
    @staticmethod
    def measure(name: str):
        """Context manager do mierzenia czasu"""
        import time
        from contextlib import contextmanager
        
        @contextmanager
        def timer():
            start_time = time.time()
            try:
                yield
            finally:
                elapsed = time.time() - start_time
                logger.debug(f"⏱️  {name}: {elapsed:.3f}s")
        
        return timer()


# ============================================
# 🎯 GLOBAL LOGGER INSTANCE
# ============================================

# Tworzymy tymczasową instancję
log = StructuredLogger()

# ============================================
# 🚀 INITIALIZATION
# ============================================

def init_logging(
    level: str = "INFO",
    log_dir: str = "./logs",
    is_production: bool = False
):
    """Inicjalizuje system logowania"""
    global log
    
    try:
        # Ustaw ścieżkę
        log_dir_path = Path(log_dir) if log_dir else None
        
        # Skonfiguruj logger
        logger_instance = setup_logger(
            name="bmw_assistant",
            level=level,
            log_format="json" if is_production else "detailed",
            log_dir=log_dir_path,
            is_production=is_production,
            is_development=not is_production
        )
        
        # Ustaw globalny logger
        log = StructuredLogger(logger_instance)
        
        # Loguj informacje o starcie
        log.info("Logging system initialized", 
                 log_level=level,
                 log_dir=str(log_dir) if log_dir else "console_only")
        
        return logger_instance
        
    except Exception as e:
        # Minimalna inicjalizacja w przypadku błędu
        print(f"⚠️  Failed to initialize logging: {e}")
        logger.remove()
        logger.add(sys.stdout, level="INFO", format="{time} | {level} | {message}")
        log = StructuredLogger()
        return logger


# Prosta auto-inicjalizacja (możesz to potem wywołać ręcznie)
try:
    # Spróbuj zaimportować settings
    from app.core.config import settings
    
    # Inicjalizuj z settings
    init_logging(
        level=getattr(settings, 'LOG_LEVEL', 'INFO'),
        log_dir=getattr(settings, 'LOG_DIR', './logs'),
        is_production=getattr(settings, 'IS_PRODUCTION', False)
    )
    
except ImportError:
    # Jeśli nie ma settings, użyj domyślnych
    print("⚠️  Could not import settings, using default logging")
    init_logging(level="INFO", log_dir="./logs", is_production=False)
except Exception as e:
    print(f"⚠️  Logging initialization error: {e}")
    init_logging(level="INFO", log_dir="./logs", is_production=False)