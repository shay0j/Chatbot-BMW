"""
Moduł bezpieczeństwa i autentykacji dla BMW Assistant.
Obsługa JWT, hashowania haseł i autoryzacji.
"""
import secrets
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, Union

from jose import JWTError, jwt
from passlib.context import CryptContext
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi import Depends, HTTPException, status

from app.core.config import settings
from app.core.exceptions import (
    AuthenticationError,
    InvalidTokenError,
    TokenExpiredError
)

# ============================================
# 🔧 INITIALIZATION
# ============================================

# Kontekst do hashowania haseł
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Security scheme dla FastAPI
security = HTTPBearer(auto_error=False)

# ============================================
# 🎯 PASSWORD MANAGEMENT
# ============================================

class PasswordManager:
    """Zarządzanie hasłami"""
    
    @staticmethod
    def hash_password(password: str) -> str:
        """Hashuje hasło"""
        return pwd_context.hash(password)
    
    @staticmethod
    def verify_password(plain_password: str, hashed_password: str) -> bool:
        """Weryfikuje hasło"""
        return pwd_context.verify(plain_password, hashed_password)
    
    @staticmethod
    def generate_random_password(length: int = 16) -> str:
        """Generuje losowe hasło"""
        alphabet = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789!@#$%^&*"
        return ''.join(secrets.choice(alphabet) for _ in range(length))


# ============================================
# 🎫 JWT TOKEN MANAGEMENT
# ============================================

class JWTManager:
    """Zarządzanie tokenami JWT"""
    
    @staticmethod
    def create_access_token(
        data: Dict[str, Any],
        expires_delta: Optional[timedelta] = None
    ) -> str:
        """Tworzy JWT access token"""
        to_encode = data.copy()
        
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(
                minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES
            )
        
        to_encode.update({
            "exp": expire,
            "iat": datetime.utcnow(),
            "type": "access",
            "jti": secrets.token_hex(16)  # Unique JWT ID
        })
        
        encoded_jwt = jwt.encode(
            to_encode,
            settings.JWT_SECRET_KEY,
            algorithm="HS256"
        )
        
        return encoded_jwt
    
    @staticmethod
    def create_refresh_token(user_id: str) -> str:
        """Tworzy JWT refresh token"""
        expire = datetime.utcnow() + timedelta(days=30)
        
        to_encode = {
            "sub": user_id,
            "exp": expire,
            "iat": datetime.utcnow(),
            "type": "refresh",
            "jti": secrets.token_hex(16)
        }
        
        encoded_jwt = jwt.encode(
            to_encode,
            settings.JWT_SECRET_KEY,
            algorithm="HS256"
        )
        
        return encoded_jwt
    
    @staticmethod
    def verify_token(token: str) -> Dict[str, Any]:
        """Weryfikuje i dekoduje JWT token"""
        try:
            payload = jwt.decode(
                token,
                settings.JWT_SECRET_KEY,
                algorithms=["HS256"]
            )
            
            # Sprawdź czy token nie wygasł
            exp = payload.get("exp")
            if exp is None:
                raise InvalidTokenError("Token has no expiration")
            
            if datetime.utcnow() > datetime.fromtimestamp(exp):
                raise TokenExpiredError()
            
            # Sprawdź typ tokena
            token_type = payload.get("type")
            if token_type not in ["access", "refresh"]:
                raise InvalidTokenError("Invalid token type")
            
            return payload
            
        except JWTError as e:
            raise InvalidTokenError(detail=str(e))
    
    @staticmethod
    def refresh_access_token(refresh_token: str) -> str:
        """Odświeża access token używając refresh token"""
        try:
            payload = JWTManager.verify_token(refresh_token)
            
            # Upewnij się że to refresh token
            if payload.get("type") != "refresh":
                raise InvalidTokenError("Not a refresh token")
            
            # Stwórz nowy access token
            user_id = payload.get("sub")
            if not user_id:
                raise InvalidTokenError("No user ID in token")
            
            return JWTManager.create_access_token({"sub": user_id})
            
        except JWTError as e:
            raise InvalidTokenError(detail=str(e))


# ============================================
# 🛡️ SECURITY MANAGER (FACADE)
# ============================================

class SecurityManager:
    """Główny manager bezpieczeństwa - fasada dla wszystkich funkcji"""
    
    def __init__(self):
        self.password_manager = PasswordManager()
        self.jwt_manager = JWTManager()
    
    def authenticate_user(
        self, 
        username: str, 
        password: str, 
        hashed_password: str
    ) -> bool:
        """Autentykuje użytkownika"""
        if not username or not password:
            return False
        
        return self.password_manager.verify_password(password, hashed_password)
    
    def create_user_tokens(self, user_id: str) -> Dict[str, str]:
        """Tworzy access i refresh token dla użytkownika"""
        access_token = self.jwt_manager.create_access_token(
            {"sub": user_id}
        )
        refresh_token = self.jwt_manager.create_refresh_token(user_id)
        
        return {
            "access_token": access_token,
            "refresh_token": refresh_token,
            "token_type": "bearer"
        }
    
    def validate_and_decode(self, token: str) -> Dict[str, Any]:
        """Waliduje i dekoduje token"""
        return self.jwt_manager.verify_token(token)
    
    def hash_password(self, password: str) -> str:
        """Hashuje hasło"""
        return self.password_manager.hash_password(password)


# ============================================
# 🔌 FASTAPI DEPENDENCIES
# ============================================

def get_security_manager() -> SecurityManager:
    """Dependency dla SecurityManager"""
    return SecurityManager()


async def verify_token(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)
) -> str:
    """
    Weryfikuje token JWT z nagłówka Authorization.
    
    Returns:
        user_id (str): ID użytkownika z tokena
    
    Raises:
        AuthenticationError: Jeśli token jest nieprawidłowy
    """
    if credentials is None:
        raise AuthenticationError("No credentials provided")
    
    token = credentials.credentials
    
    try:
        security_manager = get_security_manager()
        payload = security_manager.validate_and_decode(token)
        
        user_id = payload.get("sub")
        if user_id is None:
            raise AuthenticationError("Invalid token payload")
        
        return user_id
        
    except (InvalidTokenError, TokenExpiredError) as e:
        raise AuthenticationError(detail=str(e))


async def optional_verify_token(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)
) -> Optional[str]:
    """
    Opcjonalna weryfikacja tokenu.
    Zwraca user_id jeśli token jest prawidłowy, w przeciwnym razie None.
    """
    if credentials is None:
        return None
    
    token = credentials.credentials
    
    try:
        security_manager = get_security_manager()
        payload = security_manager.validate_and_decode(token)
        return payload.get("sub")
    except:
        return None


def get_current_user(
    token: str = Depends(verify_token),
    security_manager: SecurityManager = Depends(get_security_manager)
) -> Dict[str, Any]:
    """
    Pobiera obecnego użytkownika na podstawie tokena.
    W rzeczywistej aplikacji pobierałby dane z bazy.
    """
    # W rzeczywistej aplikacji pobierasz użytkownika z bazy
    # user = await user_repository.get_by_id(token)
    
    # Na razie zwracamy podstawowe dane
    return {
        "id": token,
        "username": f"user_{token[:8]}",
        "role": "user"
    }


# ============================================
# 🎯 PERMISSION CHECKERS
# ============================================

class PermissionChecker:
    """Sprawdza uprawnienia użytkownika"""
    
    @staticmethod
    def is_admin(user: Dict[str, Any]) -> bool:
        """Czy użytkownik jest administratorem?"""
        return user.get("role") == "admin"
    
    @staticmethod
    def can_access_conversation(
        user: Dict[str, Any], 
        conversation_user_id: str
    ) -> bool:
        """Czy użytkownik może uzyskać dostęp do konwersacji?"""
        if PermissionChecker.is_admin(user):
            return True
        return user.get("id") == conversation_user_id
    
    @staticmethod
    def can_modify_resource(
        user: Dict[str, Any], 
        resource_owner_id: str
    ) -> bool:
        """Czy użytkownik może modyfikować zasób?"""
        if PermissionChecker.is_admin(user):
            return True
        return user.get("id") == resource_owner_id


# ============================================
# 🔒 RATE LIMITING
# ============================================

class RateLimiter:
    """Prosty rate limiter (w produkcji użyj Redis)"""
    
    def __init__(self, requests_per_minute: int = 60):
        self.requests_per_minute = requests_per_minute
        self.requests = {}  # user_id -> [timestamps]
    
    async def is_rate_limited(self, user_id: str) -> bool:
        """Sprawdza czy użytkownik przekroczył limit"""
        from datetime import datetime, timedelta
        
        now = datetime.utcnow()
        minute_ago = now - timedelta(minutes=1)
        
        # Pobierz requesty użytkownika
        user_requests = self.requests.get(user_id, [])
        
        # Oczyść stare requesty
        user_requests = [req for req in user_requests if req > minute_ago]
        
        # Sprawdź limit
        if len(user_requests) >= self.requests_per_minute:
            return True
        
        # Dodaj nowy request
        user_requests.append(now)
        self.requests[user_id] = user_requests[-self.requests_per_minute:]
        
        return False
    
    async def get_remaining_requests(self, user_id: str) -> int:
        """Zwraca pozostałą liczbę requestów w oknie czasowym"""
        from datetime import datetime, timedelta
        
        now = datetime.utcnow()
        minute_ago = now - timedelta(minutes=1)
        
        user_requests = self.requests.get(user_id, [])
        user_requests = [req for req in user_requests if req > minute_ago]
        
        return max(0, self.requests_per_minute - len(user_requests))


# ============================================
# 🚨 SECURITY UTILITIES
# ============================================

def sanitize_input(text: str) -> str:
    """
    Podstawowe czyszczenie inputu użytkownika.
    W produkcji użyj dedykowanej biblioteki jak bleach.
    """
    import html
    
    # Escape HTML
    text = html.escape(text)
    
    # Remove excessive whitespace
    text = ' '.join(text.split())
    
    # Truncate if too long
    max_length = 10000
    if len(text) > max_length:
        text = text[:max_length] + "... [TRUNCATED]"
    
    return text


def validate_email(email: str) -> bool:
    """Waliduje email (podstawowa wersja)"""
    import re
    
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, email))


def generate_api_key() -> str:
    """Generuje losowy klucz API"""
    return secrets.token_urlsafe(32)


# ============================================
# 🛡️ SECURITY MIDDLEWARE (przykład)
# ============================================

async def security_headers_middleware(request, call_next):
    """
    Middleware dodające nagłówki bezpieczeństwa.
    Możesz dodać to do FastAPI.
    """
    response = await call_next(request)
    
    # Dodaj nagłówki bezpieczeństwa
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    
    # CSP - Content Security Policy
    # W produkcji dostosuj do swoich potrzeb
    response.headers["Content-Security-Policy"] = (
        "default-src 'self'; "
        "script-src 'self' 'unsafe-inline'; "
        "style-src 'self' 'unsafe-inline'; "
        "img-src 'self' data: https:;"
    )
    
    return response