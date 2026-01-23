"""
Customowe wyjątki dla aplikacji BMW Assistant.
Wszystkie wyjątki dziedziczą po BMWAssistantException.
"""

class BMWAssistantException(Exception):
    """Bazowy wyjątek dla całej aplikacji"""
    def __init__(self, message: str, detail: str = None):
        self.message = message
        self.detail = detail
        super().__init__(self.message)
    
    def __str__(self):
        if self.detail:
            return f"{self.message}: {self.detail}"
        return self.message


# ============================================
# 🔐 AUTHENTICATION & AUTHORIZATION
# ============================================

class AuthenticationError(BMWAssistantException):
    """Błąd autentykacji"""
    def __init__(self, message: str = "Authentication failed", detail: str = None):
        super().__init__(message, detail)


class AuthorizationError(BMWAssistantException):
    """Błąd autoryzacji (brak uprawnień)"""
    def __init__(self, message: str = "Not authorized", detail: str = None):
        super().__init__(message, detail)


class TokenExpiredError(AuthenticationError):
    """Token wygasł"""
    def __init__(self, message: str = "Token expired", detail: str = None):
        super().__init__(message, detail)


class InvalidTokenError(AuthenticationError):
    """Nieprawidłowy token"""
    def __init__(self, message: str = "Invalid token", detail: str = None):
        super().__init__(message, detail)


# ============================================
# 📊 DATA & VALIDATION
# ============================================

class ValidationError(BMWAssistantException):
    """Błąd walidacji danych"""
    def __init__(self, message: str = "Validation error", detail: str = None):
        super().__init__(message, detail)


class NotFoundError(BMWAssistantException):
    """Zasób nie znaleziony"""
    def __init__(self, resource: str = "Resource", detail: str = None):
        message = f"{resource} not found"
        super().__init__(message, detail)


class AlreadyExistsError(BMWAssistantException):
    """Zasób już istnieje"""
    def __init__(self, resource: str = "Resource", detail: str = None):
        message = f"{resource} already exists"
        super().__init__(message, detail)


# ============================================
# 🔌 EXTERNAL SERVICES
# ============================================

class APIError(BMWAssistantException):
    """Błąd zewnętrznego API"""
    def __init__(self, service: str = "External API", detail: str = None):
        message = f"{service} error"
        super().__init__(message, detail)


class RateLimitExceeded(APIError):
    """Przekroczony limit requestów"""
    def __init__(self, service: str = "API", detail: str = None):
        message = f"{service} rate limit exceeded"
        super().__init__(message, detail)


class ServiceUnavailableError(APIError):
    """Serwis niedostępny"""
    def __init__(self, service: str = "Service", detail: str = None):
        message = f"{service} unavailable"
        super().__init__(message, detail)


class ConfigurationError(BMWAssistantException):
    """Błąd konfiguracji"""
    def __init__(self, message: str = "Configuration error", detail: str = None):
        super().__init__(message, detail)


# ============================================
# 🧠 AI & LLM SPECIFIC
# ============================================

class LLMError(BMWAssistantException):
    """Błąd modelu językowego"""
    def __init__(self, message: str = "LLM error", detail: str = None):
        super().__init__(message, detail)


class PromptError(BMWAssistantException):
    """Błąd w prompt engineering"""
    def __init__(self, message: str = "Prompt error", detail: str = None):
        super().__init__(message, detail)


class EmbeddingError(BMWAssistantException):
    """Błąd podczas tworzenia embeddingów"""
    def __init__(self, message: str = "Embedding error", detail: str = None):
        super().__init__(message, detail)


class RAGError(BMWAssistantException):
    """Błąd systemu RAG"""
    def __init__(self, message: str = "RAG error", detail: str = None):
        super().__init__(message, detail)


# ============================================
# 💾 DATABASE & CACHE
# ============================================

class DatabaseError(BMWAssistantException):
    """Błąd bazy danych"""
    def __init__(self, message: str = "Database error", detail: str = None):
        super().__init__(message, detail)


class CacheError(BMWAssistantException):
    """Błąd cache"""
    def __init__(self, message: str = "Cache error", detail: str = None):
        super().__init__(message, detail)


# ============================================
# 🌐 NETWORK & CONNECTIVITY
# ============================================

class NetworkError(BMWAssistantException):
    """Błąd sieci"""
    def __init__(self, message: str = "Network error", detail: str = None):
        super().__init__(message, detail)


class TimeoutError(BMWAssistantException):
    """Timeout operacji"""
    def __init__(self, operation: str = "Operation", detail: str = None):
        message = f"{operation} timeout"
        super().__init__(message, detail)


# ============================================
# 🎯 BUSINESS LOGIC
# ============================================

class ConversationError(BMWAssistantException):
    """Błąd zarządzania konwersacją"""
    def __init__(self, message: str = "Conversation error", detail: str = None):
        super().__init__(message, detail)


class BMWSpecsError(BMWAssistantException):
    """Błąd specyfikacji BMW"""
    def __init__(self, message: str = "BMW specifications error", detail: str = None):
        super().__init__(message, detail)


# ============================================
# 🛠️ UTILITY FUNCTIONS
# ============================================

def wrap_exception(exception_class):
    """
    Dekorator do opakowywania wyjątków w nasze customowe.
    
    Usage:
        @wrap_exception(APIError)
        def some_function():
            ...
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except BMWAssistantException:
                raise
            except Exception as e:
                raise exception_class(detail=str(e))
        return wrapper
    return decorator


def error_to_dict(exception: BMWAssistantException) -> dict:
    """Konwertuje wyjątek na słownik dla odpowiedzi API"""
    return {
        "error": exception.__class__.__name__,
        "message": str(exception),
        "detail": exception.detail if hasattr(exception, 'detail') else None
    }