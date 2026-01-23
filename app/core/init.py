"""
Core module for BMW Assistant application.
"""

from app.core.config import settings, get_settings, validate_configuration
from app.core.exceptions import (
    BMWAssistantException,
    ConfigurationError,
    APIError,
    NotFoundError,
    ValidationError,
    RateLimitExceeded,
    AuthenticationError
)
from app.core.security import (
    SecurityManager,
    JWTManager,
    PasswordManager,
    get_current_user,
    verify_token
)

__all__ = [
    # Config
    "settings",
    "get_settings",
    "validate_configuration",
    
    # Exceptions
    "BMWAssistantException",
    "ConfigurationError",
    "APIError",
    "NotFoundError",
    "ValidationError",
    "RateLimitExceeded",
    "AuthenticationError",
    
    # Security
    "SecurityManager",
    "JWTManager",
    "PasswordManager",
    "get_current_user",
    "verify_token",
]