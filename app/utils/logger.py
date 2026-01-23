"""
Zaawansowany system logowania dla BMW Assistant.
Wykorzystuje loguru dla strukturalnego logowania.
"""
import sys
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional
from enum import Enum

from loguru import logger
from app.core.config import settings

# ============================================
# 🎯 CONFIGURATION
# ============================================

class LogLevel(str, Enum):
    """Poziomy logowania"""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class LogFormat(str, Enum):
    """Formaty logowania"""
    SIMPLE = "simple"
    JSON = "json"
    DETAILED = "detailed"


# ============================================
# 🛠️ LOGGER SETUP
# ============================================

def setup_logger(
    name: str = "bmw_assistant",
    level: str = None,
    log_format: str = None,
    log_dir: Path = None
) -> logger:
    """
    Konfiguruje i zwraca logger.
    
    Args:
        name: Nazwa loggera
        level: Poziom logowania (string)
        log_format: Format logów (string)
        log_dir: Katalog dla plików logów
    
    Returns:
        Skonfigurowany logger
    """
    # Użyj ustawień z configa jeśli nie podano
    if level is None:
        level = settings.LOG_LEVEL
    if log_format is None:
        log_format = LogFormat.JSON if settings.IS_PRODUCTION else LogFormat.DETAILED
    if log_dir is None:
        log_dir = Path(settings.LOG_DIR)
    
    # Walidacja poziomu logowania
    try:
        level_enum = LogLevel(level)
        level = level_enum.value
    except ValueError:
        # Jeśli poziom nie jest poprawnym Enum, użyj domyślnego
        level = LogLevel.INFO.value
        logger.warning(f"Invalid log level '{level}', using INFO instead")
    
    # Walidacja formatu
    try:
        format_enum = LogFormat(log_format)
        log_format = format_enum.value
    except ValueError:
        # Jeśli format nie jest poprawnym Enum, użyj domyślnego
        log_format = LogFormat.DETAILED.value if settings.IS_DEVELOPMENT else LogFormat.JSON.value
        logger.warning(f"Invalid log format '{log_format}', using default instead")
    
    # Upewnij się że katalog istnieje
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Usuń domyślnego handlera
    logger.remove()
    
    # Konfiguracja formatu
    if log_format == LogFormat.JSON:
        format_string = (
            '{"time": "{time:YYYY-MM-DD HH:mm:ss.SSS}", '
            '"level": "{level}", '
            '"name": "{name}", '
            '"function": "{function}", '
            '"line": {line}, '
            '"message": "{message}", '
            '"extra": {extra}}'
        )
    elif log_format == LogFormat.DETAILED:
        format_string = (
            "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
            "<level>{message}</level> | "
            "<yellow>{extra}</yellow>"
        )
    else:  # SIMPLE
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
        diagnose=settings.IS_DEVELOPMENT
    )
    
    # Handler dla pliku (wszystkie logi)
    log_file = log_dir / f"{name}.log"
    logger.add(
        str(log_file),
        format=format_string,
        level=level,
        rotation="500 MB",
        retention="30 days",
        compression="zip",
        backtrace=True,
        diagnose=settings.IS_DEVELOPMENT
    )
    
    # Handler dla błędów (osobny plik)
    error_log_file = log_dir / f"{name}_errors.log"
    logger.add(
        str(error_log_file),
        format=format_string,
        level="ERROR",
        rotation="100 MB",
        retention="90 days",
        compression="zip",
        backtrace=True,
        diagnose=True
    )
    
    # Handler dla requestów (jeśli potrzebny)
    request_log_file = log_dir / f"{name}_requests.log"
    logger.add(
        str(request_log_file),
        format=format_string,
        level="INFO",
        filter=lambda record: "request_id" in record["extra"],
        rotation="200 MB",
        retention="7 days",
        compression="zip"
    )
    
    return logger.bind(name=name)


# ============================================
# 🎯 STRUCTURED LOGGING
# ============================================

class StructuredLogger:
    """Logger do strukturalnego logowania"""
    
    def __init__(self, logger_instance=logger):
        self.logger = logger_instance
    
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
    
    def log_request(
        self,
        method: str,
        path: str,
        status_code: int,
        processing_time: float,
        request_id: str,
        user_id: Optional[str] = None,
        **kwargs
    ):
        """Loguje request HTTP"""
        self.info(
            f"{method} {path} {status_code}",
            request_method=method,
            request_path=path,
            status_code=status_code,
            processing_time=processing_time,
            request_id=request_id,
            user_id=user_id,
            **kwargs
        )
    
    def log_chat_interaction(
        self,
        conversation_id: str,
        user_message: str,
        assistant_response: str,
        tokens_used: Dict[str, int],
        processing_time: float,
        sources_count: int = 0,
        **kwargs
    ):
        """Loguje interakcję chat"""
        self.info(
            f"Chat interaction: {conversation_id}",
            conversation_id=conversation_id,
            user_message_preview=user_message[:100],
            assistant_response_preview=assistant_response[:100],
            tokens_used=tokens_used,
            processing_time=processing_time,
            sources_count=sources_count,
            **kwargs
        )
    
    def log_llm_call(
        self,
        model: str,
        prompt_length: int,
        response_length: int,
        tokens_used: Dict[str, int],
        processing_time: float,
        **kwargs
    ):
        """Loguje wywołanie LLM"""
        self.debug(
            f"LLM call: {model}",
            model=model,
            prompt_length=prompt_length,
            response_length=response_length,
            tokens_used=tokens_used,
            processing_time=processing_time,
            **kwargs
        )
    
    def log_rag_retrieval(
        self,
        query: str,
        documents_count: int,
        average_similarity: float,
        processing_time: float,
        **kwargs
    ):
        """Loguje wyszukiwanie w RAG"""
        self.debug(
            f"RAG retrieval: {query[:50]}...",
            query_preview=query[:100],
            documents_count=documents_count,
            average_similarity=average_similarity,
            processing_time=processing_time,
            **kwargs
        )


# ============================================
# 🔧 LOGGER MIDDLEWARE
# ============================================

def log_exceptions(func):
    """
    Dekorator do logowania wyjątków.
    
    Usage:
        @log_exceptions
        async def some_function():
            ...
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
                exception_type=e.__class__.__name__,
                exc_info=True
            )
            raise
    
    return wrapper


class PerformanceLogger:
    """Logger wydajności"""
    
    def __init__(self, operation_name: str, logger_instance=logger):
        self.operation_name = operation_name
        self.logger = logger_instance
        self.start_time = None
    
    def __enter__(self):
        self.start_time = datetime.now()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed = (datetime.now() - self.start_time).total_seconds()
        
        if exc_type is None:
            self.logger.debug(
                f"{self.operation_name} completed in {elapsed:.3f}s",
                operation=self.operation_name,
                duration=elapsed
            )
        else:
            self.logger.error(
                f"{self.operation_name} failed after {elapsed:.3f}s",
                operation=self.operation_name,
                duration=elapsed,
                exception=str(exc_val)
            )
    
    @classmethod
    def measure(cls, operation_name: str):
        """Kontekst manager do mierzenia czasu operacji"""
        return cls(operation_name)


# ============================================
# 📊 LOG ANALYTICS
# ============================================

class LogAnalyzer:
    """Analiza logów"""
    
    @staticmethod
    def count_logs_by_level(log_file: Path, level: str) -> int:
        """Liczy logi danego poziomu w pliku"""
        try:
            count = 0
            with open(log_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if f'"level": "{level}"' in line or f'| {level} ' in line:
                        count += 1
            return count
        except:
            return 0
    
    @staticmethod
    def get_recent_errors(log_file: Path, limit: int = 10) -> list:
        """Pobiera ostatnie błędy z pliku logów"""
        errors = []
        try:
            with open(log_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if '"level": "ERROR"' in line or '| ERROR ' in line:
                        try:
                            if line.strip().startswith('{'):
                                errors.append(json.loads(line.strip()))
                            else:
                                errors.append({"raw": line.strip()})
                        except:
                            errors.append({"raw": line.strip()})
            
            return errors[-limit:]
        except:
            return []
    
    @staticmethod
    def calculate_error_rate(total_logs: int, error_logs: int) -> float:
        """Oblicza wskaźnik błędów"""
        if total_logs == 0:
            return 0.0
        return (error_logs / total_logs) * 100


# ============================================
# 🎯 GLOBAL LOGGER INSTANCE
# ============================================

# Tworzymy globalną instancję loggera
log = StructuredLogger()

# Aliasy dla łatwego dostępu
debug = log.debug
info = log.info
warning = log.warning
error = log.error
critical = log.critical

# ============================================
# 🚀 INITIALIZATION
# ============================================

def init_logging():
    """Inicjalizuje system logowania"""
    # Ustaw globalny logger
    global log
    logger_instance = setup_logger()
    log = StructuredLogger(logger_instance)
    
    # Loguj informacje o starcie
    log.info("Logging system initialized", 
             environment=settings.ENVIRONMENT,  # Użyj bez .value
             log_level=settings.LOG_LEVEL,     # Użyj bez .value
             log_dir=str(settings.LOG_DIR))
    
    return logger_instance


# Auto-inicjalizacja przy imporcie
if not settings.TEST_MODE:
    try:
        init_logging()
    except Exception as e:
        # Minimalna inicjalizacja w przypadku błędu
        logger.remove()
        logger.add(sys.stdout, level="INFO", format="{time} | {level} | {message}")
        logger.warning(f"Failed to initialize logging system: {e}")