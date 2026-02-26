import os
import asyncio
import time
import json
import base64
import re
import sys
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import httpx
from loguru import logger
from dotenv import load_dotenv
from pydantic import BaseModel
import secrets
import traceback

# ============================================
# IMPORT RAG Z TWOJEGO PLIKU
# ============================================

RAG_FILE_PATH = Path(r"C:\Users\hellb\Documents\Chatbot_BMW\RAG\src\scrapers\6_rag_test.py")

def import_rag_module():
    """Dynamicznie importuje moduł RAG"""
    try:
        if not RAG_FILE_PATH.exists():
            raise FileNotFoundError(f"Nie znaleziono pliku RAG: {RAG_FILE_PATH}")
        
        rag_dir = RAG_FILE_PATH.parent
        if str(rag_dir) not in sys.path:
            sys.path.insert(0, str(rag_dir))
        
        import importlib.util
        
        module_name = "rag_module_6_test"
        spec = importlib.util.spec_from_file_location(
            module_name, 
            str(RAG_FILE_PATH)
        )
        
        if spec is None:
            raise ImportError(f"Nie można utworzyć specyfikacji dla {RAG_FILE_PATH}")
        
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        
        print(f"✅ Załadowano moduł RAG: {module_name}")
        
        if not hasattr(module, 'RAGSystem'):
            raise AttributeError("Brak klasy RAGSystem w module")
        
        if not hasattr(module, 'find_latest_vector_db'):
            raise AttributeError("Brak funkcji find_latest_vector_db w module")
        
        return module
        
    except Exception as e:
        print(f"❌ Błąd ładowania modułu RAG: {e}")
        raise

try:
    rag_module = import_rag_module()
    RAGSystem = rag_module.RAGSystem
    find_latest_vector_db = rag_module.find_latest_vector_db
    RAG_AVAILABLE = True
    print("✅ RAG system gotowy do użycia")
except Exception as e:
    print(f"⚠️ Ostrzeżenie: Could not import RAG module: {e}")
    print("Aplikacja będzie działać bez RAG")
    RAG_AVAILABLE = False
    
    class RAGSystem:
        def __init__(self, vector_db_path=None):
            self.vector_db_path = vector_db_path
            print(f"Używam dummy RAGSystem")
        
        def query(self, query, k=3, use_model_filter=False, use_priority=True):
            print(f"Dummy RAG query: '{query[:50]}...'")
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
        print("Dummy find_latest_vector_db: zwracam None")
        return None

# ============================================
# RAG SERVICE SINGLETON
# ============================================

_rag_service_instance = None

def get_rag_service():
    """Zwraca singleton RAG service"""
    global _rag_service_instance
    if _rag_service_instance is None:
        print("Tworzę singleton RAG service...")
        _rag_service_instance = SimpleRAGService()
    return _rag_service_instance

# ============================================
# POPRAWIONY RAG SERVICE Z TWOJEGO PLIKU
# ============================================

class SimpleRAGService:
    """Adapter dla RAG-a z zaawansowaną walidacją danych"""
    
    def __init__(self):
        print(f"Inicjalizacja SimpleRAGService...")
        
        if not RAG_AVAILABLE:
            print("RAG nie dostępny - tworzę dummy service")
            self._create_dummy_service()
            return
        
        try:
            db_file = find_latest_vector_db()
            if not db_file:
                print("Nie znaleziono bazy RAG - tworzę dummy service")
                self._create_dummy_service()
                return
            
            print(f"Ładowanie bazy RAG z: {db_file}")
            self.rag = RAGSystem(vector_db_path=db_file)
            self.db_info = self.rag.get_database_info()
            print(f"RAG załadowany: {self.db_info.get('total_chunks', 0)} fragmentów")
            
        except Exception as e:
            print(f"Błąd inicjalizacji RAG: {e}")
            print("Tworzę dummy service jako fallback")
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
        print("Dummy RAG service utworzony")
    
    async def retrieve(self, query: str, top_k: int = 3, similarity_threshold: float = 0.7) -> Any:
        """
        Wyszukuje dokumenty w RAG z zaawansowaną walidacją
        """
        print(f"RAG retrieve: '{query[:50]}...' (top_k={top_k})")
        
        # Wykryj modele BMW w zapytaniu
        bmw_models = ['i3', 'i4', 'i5', 'i7', 'i8', 'ix', 'x1', 'x2', 'x3', 'x4', 'x5', 
                     'x6', 'x7', 'xm', '2 series', '3 series', '4 series', '5 series',
                     '7 series', '8 series', 'm2', 'm3', 'm4', 'm5', 'm8', 'z4',
                     'seria 2', 'seria 3', 'seria 4', 'seria 5', 'seria 7', 'seria 8']
        
        query_lower = query.lower()
        detected_models_in_query = []
        
        for model in bmw_models:
            if model in query_lower:
                detected_models_in_query.append(model.upper())
        
        use_filter = len(detected_models_in_query) > 0
        
        if detected_models_in_query:
            print(f"   Wykryto modele: {detected_models_in_query}, filtrowanie: {use_filter}")
        
        try:
            # Pierwsze wyszukiwanie z filtrem jeśli wykryto modele
            if use_filter:
                results = self.rag.query(
                    query, 
                    k=top_k * 2,  # Więcej wyników do filtrowania
                    use_model_filter=True,
                    use_priority=True
                )
                print(f"   Znaleziono {len(results)} wyników z filtrem modelu")
            else:
                results = self.rag.query(
                    query, 
                    k=top_k,
                    use_model_filter=False,
                    use_priority=True
                )
                print(f"   Znaleziono {len(results)} wyników bez filtra")
            
            # Fallback: jeśli z filtrem nie ma wyników, spróbuj bez filtra
            if use_filter and len(results) < 2:
                print("   Mało wyników z filtrem, próbuję bez filtra...")
                fallback_results = self.rag.query(
                    query, 
                    k=top_k,
                    use_model_filter=False,
                    use_priority=True
                )
                # Dodaj fallback wyniki, ale zachowaj priorytet
                for result in fallback_results:
                    if result not in results:
                        results.append(result)
                results = results[:top_k * 2]
                print(f"   Po fallback: {len(results)} wyników")
            
            if not results:
                print("   Brak wyników - zwracam pustą odpowiedź")
                return self._create_empty_result()
            
            # WALIDUJ i sortuj wyniki
            validated_docs = []
            for result in results:
                doc_text = result.get('text', '')
                metadata = result.get('metadata', {})
                similarity = result.get('similarity_score', 0.0)
                
                # Walidacja jakości danych
                validation_result = self._validate_document_advanced(doc_text, metadata, query_lower)
                
                if validation_result['is_valid'] or similarity > 0.6:
                    doc = {
                        'content': doc_text,
                        'metadata': metadata,
                        'similarity': similarity,
                        'relevance_score': self._calculate_relevance_score(
                            result.get('relevance_score', similarity),
                            validation_result,
                            detected_models_in_query,
                            metadata.get('models', [])
                        ),
                        'validated': validation_result['is_valid'],
                        'warnings': validation_result['warnings'],
                        'quality_score': validation_result['quality_score']
                    }
                    validated_docs.append(doc)
            
            # Sortuj po relevance_score
            validated_docs.sort(key=lambda x: x['relevance_score'], reverse=True)
            validated_docs = validated_docs[:top_k]  # Weź najlepsze top_k
            
            # Oblicz średnie podobieństwo tylko dla zwalidowanych
            valid_similarities = [d['similarity'] for d in validated_docs if d.get('validated', False)]
            avg_similarity = sum(valid_similarities) / len(valid_similarities) if valid_similarities else 0.0
            
            class ResultWrapper:
                def __init__(self, docs, avg_sim, detected_models):
                    self.documents = docs
                    self.average_similarity = avg_sim
                    self.detected_models = detected_models
                    self.has_valid_data = len([d for d in docs if d.get('validated', False)]) > 0
                    self.total_warnings = sum(len(d.get('warnings', [])) for d in docs)
                    self.quality_scores = [d.get('quality_score', 0.0) for d in docs]
                
                def to_api_response(self):
                    sources = []
                    for i, doc in enumerate(self.documents):
                        metadata = doc['metadata']
                        content = doc['content']
                        source_info = {
                            'id': i + 1,
                            'title': metadata.get('title', 'Brak tytułu')[:100],
                            'content': content[:300] + ('...' if len(content) > 300 else ''),
                            'similarity': round(doc['similarity'], 3),
                            'relevance': round(doc.get('relevance_score', doc['similarity']), 3),
                            'quality': round(doc.get('quality_score', 0.0), 3),
                            'url': metadata.get('source_url', ''),
                            'models': metadata.get('models', []),
                            'validated': doc.get('validated', False),
                            'warnings': doc.get('warnings', []),
                            'has_technical_data': self._has_technical_data(content)
                        }
                        sources.append(source_info)
                    
                    return {"sources": sources}
                
                def _has_technical_data(self, text):
                    """Sprawdza czy tekst zawiera dane techniczne"""
                    tech_keywords = ['km', 'km/h', '0-100', 'silnik', 'moc', 'skrzynia', 'bieg', 'napęd', 'pojemność']
                    text_lower = text.lower()
                    return any(keyword in text_lower for keyword in tech_keywords)
            
            return ResultWrapper(validated_docs, avg_similarity, detected_models_in_query)
            
        except Exception as e:
            print(f"Błąd RAG retrieve: {e}")
            return self._create_empty_result()
    
    def _validate_document_advanced(self, text: str, metadata: Dict, query: str) -> Dict[str, Any]:
        """
        Zaawansowana walidacja dokumentu
        """
        text_lower = text.lower()
        query_lower = query.lower()
        
        # Sprawdź czy dokument jest aktualny
        year = metadata.get('year', '')
        is_recent = False
        try:
            if year and year.isdigit():
                year_int = int(year)
                is_recent = year_int >= 2020
        except:
            pass
        
        # Sprawdź czy zawiera dane techniczne
        has_technical_data = any(keyword in text_lower for keyword in [
            'km', 'km/h', '0-100', 'silnik', 'moc', 'skrzynia', 'bieg', 
            'napęd', 'pojemność', 'przyspieszenie', 'moment', 'v-max'
        ])
        
        # Sprawdź czy pasuje do zapytania
        query_words = set(query_lower.split())
        text_words = set(text_lower.split())
        matching_words = len(query_words.intersection(text_words))
        query_match_ratio = matching_words / max(len(query_words), 1)
        
        # Sprawdź ostrzeżenia
        warnings = []
        warning_patterns = [
            (r'6[\s\-]*biegow', 'Przestarzała skrzynia biegów'),
            (r'190\s*km', 'Nierealistyczna moc'),
            (r'stary', 'Stare dane'),
            (r'201[0-5]', 'Przestarzały rok'),
        ]
        
        for pattern, warning_msg in warning_patterns:
            if re.search(pattern, text_lower):
                warnings.append(warning_msg)
        
        # Oblicz jakość dokumentu (0.0 - 1.0)
        quality_score = 0.5  # Bazowy
        
        if is_recent:
            quality_score += 0.2
        if has_technical_data:
            quality_score += 0.2
        if query_match_ratio > 0.3:
            quality_score += 0.1
        if len(warnings) == 0:
            quality_score += 0.1
        if 'specyfikacja' in text_lower or 'dane techniczne' in text_lower:
            quality_score += 0.1
        
        # Normalizuj do 0.0-1.0
        quality_score = min(1.0, max(0.0, quality_score))
        
        is_valid = quality_score >= 0.6 and has_technical_data
        
        return {
            'is_valid': is_valid,
            'quality_score': quality_score,
            'is_recent': is_recent,
            'has_technical_data': has_technical_data,
            'query_match_ratio': query_match_ratio,
            'warnings': warnings
        }
    
    def _calculate_relevance_score(self, base_score: float, validation_result: Dict, 
                                 query_models: List[str], doc_models: List[str]) -> float:
        """Oblicza wynik relewancji z uwzględnieniem wielu czynników"""
        relevance = base_score
        
        # Bonus za zgodność modeli
        if query_models and doc_models:
            matching_models = set(m.upper() for m in query_models) & set(m.upper() for m in doc_models)
            if matching_models:
                relevance += 0.2
        
        # Bonus za jakość
        relevance += validation_result['quality_score'] * 0.1
        
        # Bonus za aktualność
        if validation_result['is_recent']:
            relevance += 0.1
        
        # Bonus za dane techniczne
        if validation_result['has_technical_data']:
            relevance += 0.15
        
        # Kara za ostrzeżenia
        relevance -= len(validation_result['warnings']) * 0.05
        
        return max(0.0, min(1.0, relevance))
    
    def _create_empty_result(self):
        """Tworzy pusty wynik"""
        class EmptyResult:
            def __init__(self):
                self.documents = []
                self.average_similarity = 0.0
                self.detected_models = []
                self.has_valid_data = False
                self.total_warnings = 0
            
            def to_api_response(self):
                return {"sources": []}
        
        return EmptyResult()
    
    async def health_check(self) -> Dict[str, Any]:
        """Health check dla RAG"""
        try:
            if not RAG_AVAILABLE:
                return {
                    "status": "unavailable", 
                    "error": "RAG system not imported",
                    "is_dummy": True
                }
            
            # Sprawdź czy możemy wykonać testowe zapytanie
            test_result = self.rag.query("BMW X5", k=1, use_model_filter=True)
            test_ok = len(test_result) > 0
            
            return {
                "status": "healthy" if test_ok else "degraded",
                "chunks": self.db_info.get('total_chunks', 0),
                "vectors": self.db_info.get('total_vectors', 0),
                "embedding_model": self.db_info.get('model_name', 'unknown'),
                "test_query_ok": test_ok,
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

# ============================================
# KONFIGURACJA I ZALADOWANIE ZMIENNYCH ŚRODOWISKOWYCH
# ============================================
load_dotenv()

print("\n" + "="*60)
print("🔍 DEBUG - Environment variables for LiveChat:")
print(f"LIVECHAT_CLIENT_ID: {os.getenv('LIVECHAT_CLIENT_ID')}")
print(f"LIVECHAT_CLIENT_SECRET: {os.getenv('LIVECHAT_CLIENT_SECRET')[:5] if os.getenv('LIVECHAT_CLIENT_SECRET') else 'None'}... (częściowo ukryty)")
print(f"LIVECHAT_REDIRECT_URI: {os.getenv('LIVECHAT_REDIRECT_URI')}")
print(f"LIVECHAT_REGION: {os.getenv('LIVECHAT_REGION')}")
print(f"LIVECHAT_TARGET_GROUP_ID: {os.getenv('LIVECHAT_TARGET_GROUP_ID')}")
print(f"LIVECHAT_ORGANIZATION_ID: {os.getenv('LIVECHAT_ORGANIZATION_ID')}")
print(f"LIVECHAT_ACCOUNT_ID: {os.getenv('LIVECHAT_ACCOUNT_ID')}")
print(f"LIVECHAT_EMAIL: {os.getenv('LIVECHAT_EMAIL')}")
print("="*60 + "\n")

NGROK_URL = "https://crinklier-ruddily-leonore.ngrok-free.dev"

BOT_CONFIG_FILE = "bot_config.json"

def load_bot_config():
    if os.path.exists(BOT_CONFIG_FILE):
        try:
            with open(BOT_CONFIG_FILE, 'r') as f:
                return json.load(f)
        except:
            return {}
    return {}

def save_bot_config(config):
    with open(BOT_CONFIG_FILE, 'w') as f:
        json.dump(config, f, indent=2)

class TokenResponse(BaseModel):
    access_token: str
    refresh_token: Optional[str] = None
    expires_in: int
    token_type: str = "bearer"

class BotCreateResponse(BaseModel):
    id: str
    secret: str

class IncomingEvent(BaseModel):
    chat_id: str
    event: Dict[str, Any]
    author_id: Optional[str] = None

@dataclass
class MessageHandler:
    chat_id: str
    message: str
    author_id: str

class LiveChatAuth:
    
    def __init__(self):
        self.client_id = os.getenv("LIVECHAT_CLIENT_ID")
        self.client_secret = os.getenv("LIVECHAT_CLIENT_SECRET")
        self.redirect_uri = os.getenv("LIVECHAT_REDIRECT_URI", "http://localhost:8000/callback")
        self.region = os.getenv("LIVECHAT_REGION", "us")
        self.organization_id = os.getenv("LIVECHAT_ORGANIZATION_ID")
        self.account_id = os.getenv("LIVECHAT_ACCOUNT_ID")
        self.email = os.getenv("LIVECHAT_EMAIL")
        
        if not self.client_id or not self.client_secret:
            logger.warning("Brak LIVECHAT_CLIENT_ID lub LIVECHAT_CLIENT_SECRET w .env")
        
        self.accounts_url = "https://accounts.livechatinc.com"
        self.api_url = "https://api.livechatinc.com"
        
        self.oauth_token = "eu-west3:oLOLIreQRpV7zhu4MuWS0h38GYo"
        
        self.http_client = httpx.AsyncClient(timeout=30.0)
    
    async def get_valid_token(self) -> str:
        return self.oauth_token
    
    def is_authenticated(self) -> bool:
        return bool(self.oauth_token)
    
    async def close(self):
        await self.http_client.aclose()

auth_client = LiveChatAuth()

class LiveChatAPIClient:
    
    def __init__(self):
        self.base_url = "https://api.livechatinc.com"
        self.http_client = httpx.AsyncClient(timeout=30.0)
    
    async def _request(self, method: str, path: str, **kwargs) -> Any:
        token = await auth_client.get_valid_token()
        
        headers = kwargs.pop("headers", {})
        headers["Authorization"] = f"Bearer {token}"
        headers["Content-Type"] = "application/json"
        
        url = f"{self.base_url}{path}"
        
        try:
            print(f"\n🔍 API Request: {method} {url}")
            
            response = await self.http_client.request(
                method, url, headers=headers, **kwargs
            )
            
            print(f"📊 Response status: {response.status_code}")
            
            if response.status_code >= 400:
                error_msg = f"Błąd API {response.status_code}"
                try:
                    error_data = response.json()
                    error_msg += f": {json.dumps(error_data)}"
                    print(f"❌ {error_msg}")
                except:
                    error_msg += f": {response.text}"
                    print(f"❌ {error_msg}")
                response.raise_for_status()
            
            response_body = response.text
            if response_body.strip().startswith('{') or response_body.strip().startswith('['):
                return response.json()
            else:
                return response_body
                    
        except Exception as e:
            print(f"💥 Błąd: {e}")
            raise
    
    async def create_bot(self, name: str) -> Tuple[str, str]:
        """Tworzy nowego bota w LiveChat"""
        token = await auth_client.get_valid_token()
        
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "name": name,
            "default_group_priority": "normal",
            "max_chats_count": 5,
            "timezone": "Europe/Warsaw"
        }
        
        print(f"\n🤖 Tworzenie nowego bota o nazwie {name}")
        print(f"📦 Payload: {json.dumps(payload, indent=2)}")
        
        async with httpx.AsyncClient() as client:
            response = await client.post(
                "https://api.livechatinc.com/v3.6/configuration/action/create_bot",
                headers=headers,
                json=payload
            )
            
            print(f"📊 Response status: {response.status_code}")
            
            if response.status_code == 200 or response.status_code == 201:
                data = response.json()
                bot_id = data.get("id")
                bot_secret = data.get("secret")
                if bot_id and bot_secret:
                    print(f"✅ Bot utworzony! ID: {bot_id}")
                    return bot_id, bot_secret
                else:
                    raise Exception("Brak ID lub sekretu w odpowiedzi")
            else:
                error_data = response.json()
                raise Exception(f"Błąd tworzenia bota: {error_data}")
    
    async def get_bot_token(self, bot_id: str, bot_secret: str) -> Optional[str]:
        """Generuje token dla bota"""
        try:
            token = await auth_client.get_valid_token()
            
            headers = {
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "bot_id": bot_id,
                "client_id": auth_client.client_id,
                "organization_id": auth_client.organization_id,
                "bot_secret": bot_secret
            }
            
            print(f"\n🔑 Generowanie tokena dla bota {bot_id}")
            
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    "https://api.livechatinc.com/v3.6/configuration/action/issue_bot_token",
                    headers=headers,
                    json=payload
                )
                
                print(f"📊 Response status: {response.status_code}")
                
                if response.status_code == 200:
                    data = response.json()
                    bot_token = data.get("token")
                    print(f"✅ Token bota wygenerowany: {bot_token[:20]}...")
                    return bot_token
                else:
                    print(f"❌ Błąd generowania tokena: {response.text}")
                    return None
                    
        except Exception as e:
            print(f"❌ Błąd: {e}")
            return None
    
    async def set_routing_status(self, bot_id: str, status: str = "accepting_chats") -> Dict[str, Any]:
        agent_token = await auth_client.get_valid_token()
        
        headers = {
            "Authorization": f"Bearer {agent_token}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "id": bot_id,
            "status": status
        }
        
        url = f"{self.base_url}/v3.6/agent/action/set_routing_status"
        
        print(f"\n🤖 Ustawianie statusu routingu dla bota {bot_id} na {status}")
        print(f"📦 Payload: {json.dumps(payload, indent=2)}")
        
        async with httpx.AsyncClient() as client:
            response = await client.post(url, headers=headers, json=payload)
            print(f"📊 Response status: {response.status_code}")
            
            if response.status_code == 200 or response.status_code == 202:
                print(f"✅ Bot aktywowany (status: {status})")
                return {"success": True}
            else:
                try:
                    error_data = response.json()
                    print(f"❌ Błąd aktywacji: {json.dumps(error_data, indent=2)}")
                    return {"success": False, "error": error_data}
                except:
                    print(f"❌ Błąd aktywacji: {response.text}")
                    return {"success": False, "error": response.text}
    
    async def join_chat(self, chat_id: str, bot_id: str) -> Dict[str, Any]:
        agent_token = await auth_client.get_valid_token()
        
        headers = {
            "Authorization": f"Bearer {agent_token}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "chat_id": chat_id,
            "user_id": bot_id,
            "user_type": "agent",
            "visibility": "all",
            "ignore_requester_presence": True
        }
        
        url = f"{self.base_url}/v3.6/agent/action/add_user_to_chat"
        
        print(f"\n👋 Dołączanie do czatu {chat_id}")
        print(f"📦 Payload: {json.dumps(payload, indent=2)}")
        
        async with httpx.AsyncClient() as client:
            response = await client.post(url, headers=headers, json=payload)
            print(f"📊 Response status: {response.status_code}")
            
            try:
                response_data = response.json()
                print(f"📄 Response body: {json.dumps(response_data, indent=2)}")
            except:
                response_data = response.text
                print(f"📄 Response body: {response_data}")
            
            if response.status_code == 200 or response.status_code == 202:
                print(f"✅ Bot dołączył do czatu!")
                return {"success": True}
            else:
                print(f"❌ Błąd dołączania: {response_data}")
                return {"success": False, "error": response_data}
    
    async def send_message_as_bot(self, chat_id: str, text: str, bot_token: str) -> Dict[str, Any]:
        """Wysyła wiadomość jako bot przez Web API - używając tokena bota"""
        
        headers = {
            "Authorization": f"Bearer {bot_token}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "chat_id": chat_id,
            "event": {
                "type": "message",
                "text": text,
                "visibility": "all"
            }
        }
        
        url = f"{self.base_url}/v3.6/agent/action/send_event"
        
        print(f"\n📤 Wysyłanie wiadomości do czatu {chat_id}")
        print(f"📋 Headers: Authorization: Bearer {bot_token[:20]}...")
        print(f"📦 Payload: {json.dumps(payload, indent=2)}")
        
        async with httpx.AsyncClient() as client:
            response = await client.post(url, headers=headers, json=payload)
            print(f"📊 Response status: {response.status_code}")
            
            if response.status_code == 202:
                try:
                    response_data = response.json()
                    event_id = response_data.get("event_id", "unknown")
                    logger.info(f"✅ Wiadomość wysłana! Event ID: {event_id}")
                    return {"success": True, "event_id": event_id}
                except:
                    logger.info(f"✅ Wiadomość wysłana!")
                    return {"success": True}
            else:
                try:
                    error_data = response.json()
                    print(f"❌ Błąd: {json.dumps(error_data, indent=2)}")
                except:
                    print(f"❌ Błąd: {response.text}")
                return {"success": False, "error": response.text}
    
    async def test_connection(self) -> tuple[bool, str]:
        try:
            print("\n🔍 Testowanie połączenia z API...")
            result = await self._request(
                "POST",
                "/v3.6/configuration/action/list_agents",
                json={}
            )
            
            if isinstance(result, dict) and "agents" in result:
                agents_count = len(result.get("agents", []))
                print(f"✅ Połączenie działa. Znaleziono {agents_count} agentów")
                return True, f"OK, {agents_count} agentów"
            else:
                print(f"✅ Połączenie działa")
                return True, "OK"
                
        except Exception as e:
            error_msg = str(e)
            print(f"❌ Problem z połączeniem: {error_msg}")
            return False, error_msg

    async def transfer_chat(self, chat_id: str, target_group_id: int, bot_id: str) -> Dict[str, Any]:
        agent_token = await auth_client.get_valid_token()
        
        headers = {
            "Authorization": f"Bearer {agent_token}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "id": chat_id,
            "target": {
                "type": "group",
                "ids": [target_group_id]
            },
            "ignore_agents_availability": False
        }
        
        url = f"{self.base_url}/v3.6/agent/action/transfer_chat"
        
        print(f"\n🔄 Przekazywanie czatu {chat_id} do grupy {target_group_id}")
        print(f"📦 Payload: {json.dumps(payload, indent=2)}")
        
        async with httpx.AsyncClient() as client:
            response = await client.post(url, headers=headers, json=payload)
            print(f"📊 Response status: {response.status_code}")
            
            if response.status_code == 202:
                logger.info(f"✅ Czat {chat_id} przekazany do grupy {target_group_id}")
                return {"success": True}
            else:
                logger.error(f"❌ Błąd transferu: {response.text}")
                return {"success": False, "error": response.text}
    
    async def close(self):
        await self.http_client.aclose()

api_client = LiveChatAPIClient()

# ============================================
# ZAAWANSOWANA KLASA BOTA Z RAG
# ============================================

class YourBot:
    
    def __init__(self):
        logger.info("Inicjalizacja bota z RAG...")
        self.conversation_state = {}
        self.rag_service = get_rag_service()
        logger.info("Bot gotowy")
    
    async def process_message(self, text: str, chat_id: str = None) -> tuple[str, bool]:
        text_lower = text.lower().strip()
        
        # Inicjalizuj stan dla nowej rozmowy
        if chat_id and chat_id not in self.conversation_state:
            self.conversation_state[chat_id] = {
                "failed_attempts": 0,
                "last_topic": None,
                "context": []
            }
        
        if chat_id:
            state = self.conversation_state[chat_id]
        else:
            state = {"failed_attempts": 0}
        
        # Sprawdź czy to prośba o konsultanta
        handoff_keywords = ['konsultant', 'człowiek', 'agent', 'handoff', 'konsultanta', 'człowiekiem']
        if any(keyword in text_lower for keyword in handoff_keywords):
            return "Łączę z konsultantem...", True
        
        # Wykryj typ pytania
        question_type = "general"
        if any(word in text_lower for word in ['specyfikacj', 'dane', 'parametr', 'silnik', 'moc', 'km']):
            question_type = "specyfikacje"
        elif any(word in text_lower for word in ['różni', 'różnica', 'porównaj', 'vs', 'contra']):
            question_type = "porownanie"
        elif any(word in text_lower for word in ['rodzin', 'dzieci', 'osób', 'przestrzeń']):
            question_type = "rodzinny"
        elif any(word in text_lower for word in ['sport', 'szybk', 'mocny', 'm pakiet']):
            question_type = "sportowy"
        elif any(word in text_lower for word in ['elektryczn', 'ev', 'elektryk', 'hybryd']):
            question_type = "elektryczny"
        elif any(word in text_lower for word in ['cen', 'koszt', 'drogi', 'wyposażenie']):
            question_type = "cena"
        
        # Wykryj modele BMW
        bmw_models = ['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'xm',
                     'i3', 'i4', 'i5', 'i7', 'i8', 'ix',
                     'm2', 'm3', 'm4', 'm5', 'm8', 'z4',
                     'seria 2', 'seria 3', 'seria 4', 'seria 5', 'seria 7', 'seria 8']
        
        detected_models = []
        for model in bmw_models:
            if model in text_lower:
                model_upper = model.upper()
                if 'SERIA' in model_upper:
                    model_upper = model_upper.replace('SERIA', 'Seria')
                detected_models.append(model_upper)
        
        # Użyj RAG dla pytań wymagających wiedzy
        rag_result = None
        rag_data = ""
        sources_count = 0
        
        # Zapytaj RAG jeśli to potrzebne
        if question_type in ["specyfikacje", "porownanie", "elektryczny", "sportowy"] or detected_models:
            print(f"\n🔍 Używam RAG dla pytania typu: {question_type}")
            rag_result = await self.rag_service.retrieve(
                query=text,
                top_k=2,
                similarity_threshold=0.5
            )
            
            if hasattr(rag_result, 'documents') and rag_result.documents:
                valid_docs = [d for d in rag_result.documents if d.get('validated', False)]
                sources_count = len(valid_docs)
                
                if valid_docs:
                    # Przygotuj kontekst z RAG
                    context_parts = []
                    for i, doc in enumerate(valid_docs[:2], 1):
                        content = doc['content'][:200] + "..." if len(doc['content']) > 200 else doc['content']
                        metadata = doc.get('metadata', {})
                        source = metadata.get('title', 'Źródło')[:50]
                        context_parts.append(f"Źródło {i} ({source}): {content}")
                    
                    rag_data = "\n\n".join(context_parts)
                    print(f"✅ Znaleziono {sources_count} źródeł w RAG")
        
        # Generuj odpowiedź na podstawie typu pytania i RAG
        if question_type == "specyfikacje" and rag_data:
            response = f"Na podstawie dostępnych danych:\n\n{rag_data}\n\nCzy chcesz poznać więcej szczegółów?"
        elif question_type == "porownanie" and rag_data:
            response = f"Porównanie modeli BMW:\n\n{rag_data}\n\nKtóry z tych modeli Cię interesuje?"
        elif detected_models and rag_data:
            model_str = ', '.join(detected_models)
            response = f"Oto informacje o {model_str}:\n\n{rag_data}"
        elif any(word in text_lower for word in ['cześć', 'witaj', 'hej', 'dzień dobry']):
            response = "Cześć! Jestem Leo, ekspertem BMW w ZK Motors. W czym mogę pomóc?"
        elif any(word in text_lower for word in ['dziękuję', 'dzięki', 'thx']):
            response = "Proszę bardzo! Czy mogę pomóc w czymś jeszcze?"
        elif any(word in text_lower for word in ['godziny', 'otwarcia', 'czynne']):
            response = "Jesteśmy czynni od poniedziałku do piątku w godzinach 9:00-17:00. Zapraszamy do salonu ZK Motors!"
        elif any(word in text_lower for word in ['adres', 'gdzie', 'siedziba']):
            response = "Nasza siedziba znajduje się przy ul. Przykładowej 123 w Warszawie. Serdecznie zapraszamy!"
        elif 'bmw' in text_lower and not rag_data:
            response = "BMW oferuje wiele modeli dopasowanych do różnych potrzeb. Który segment Cię interesuje? SUV-y, sedany, czy może sportowe auta?"
        else:
            state["failed_attempts"] = state.get("failed_attempts", 0) + 1
            
            if state["failed_attempts"] >= 3:
                return "Przepraszam, nie mogę pomóc. Łączę z konsultantem.", True
            
            if detected_models:
                response = f"Przepraszam, nie znalazłem wystarczających informacji o {', '.join(detected_models)}. Czy możesz sprecyzować pytanie?"
            else:
                response = "Przepraszam, nie zrozumiałem. Czy możesz powiedzieć inaczej?"
        
        # Zapisz kontekst
        if chat_id:
            state["last_topic"] = question_type
            state["failed_attempts"] = 0
        
        return response, False

class LiveChatBotIntegration:
    
    def __init__(self, your_bot: YourBot):
        self.bot = your_bot
        self.bot_agent_id: Optional[str] = None
        self.bot_secret: Optional[str] = None
        self.bot_token: Optional[str] = None
        self.target_group_id = int(os.getenv("LIVECHAT_TARGET_GROUP_ID", "0"))
        self._running = False
        self.connection_ok = False
        self.webhook_url = f"{NGROK_URL}/webhook"
    
    async def start(self):
        if self._running:
            return
        
        try:
            config = load_bot_config()
            
            if "bot_id" in config and config["bot_id"]:
                self.bot_agent_id = config["bot_id"]
                self.bot_secret = config.get("bot_secret", "")
                print(f"✅ Wczytano bota z konfiguracji: {self.bot_agent_id}")
                print(f"🎯 Grupa docelowa: {self.target_group_id}")
                print(f"🌍 Webhook URL: {self.webhook_url}")
                
                if self.bot_secret:
                    print(f"\n🔑 Próba wygenerowania tokena bota...")
                    self.bot_token = await api_client.get_bot_token(self.bot_agent_id, self.bot_secret)
                    if self.bot_token:
                        print(f"✅ Token bota wygenerowany pomyślnie")
                    else:
                        print(f"⚠️ Nie udało się wygenerować tokena bota - sprawdź bot_secret w configu")
                else:
                    print(f"⚠️ Brak bot_secret w configu - nie można wygenerować tokena bota")
                
                ok, msg = await api_client.test_connection()
                self.connection_ok = ok
                
                if ok:
                    print(f"✅ Połączono z API LiveChat: {msg}")
                    
                    print(f"\n🤖 Próba aktywacji bota {self.bot_agent_id}...")
                    activation_result = await api_client.set_routing_status(self.bot_agent_id, "accepting_chats")
                    if activation_result.get("success"):
                        print(f"✅ Bot aktywowany pomyślnie")
                    else:
                        print(f"⚠️ Problem z aktywacją bota: {activation_result.get('error')}")
                        print(f"⚠️ Bot może nie działać poprawnie - sprawdź uprawnienia tokena")
                    
                else:
                    print(f"⚠️ Problem z połączeniem: {msg}")
                    print(f"⚠️ Bot uruchomiony ale może nie działać - sprawdź token")
                
                self._running = True
                logger.info("✅ Bot LiveChat zainicjalizowany")
                return
            
            error_msg = "❌ BRAK ID BOTA W KONFIGURACJI! Utwórz plik bot_config.json z Twoim ID bota"
            print(error_msg)
            raise Exception(error_msg)
            
        except Exception as e:
            logger.error(f"Błąd inicjalizacji: {e}")
            raise
    
    async def handle_webhook(self, payload: dict) -> None:
        try:
            action = payload.get("action")
            print(f"\n📨 Webhook action: {action}")
            
            if action == "incoming_event":
                await self._handle_incoming_event(payload)
            elif action == "incoming_chat":
                print(f"🆕 Nowy czat rozpoczęty")
                print(f"📦 Payload: {json.dumps(payload, indent=2)}")
            else:
                print(f"⏭️ Nieobsługiwana akcja: {action}")
                print(f"📦 Payload: {json.dumps(payload, indent=2)}")
                
        except Exception as e:
            logger.error(f"Błąd webhooka: {e}")
            traceback.print_exc()
    
    async def _handle_incoming_event(self, payload: dict):
        try:
            payload_data = payload.get("payload", {})
            event = payload_data.get("event", {})
            chat_id = payload_data.get("chat_id")
            
            print(f"📊 Chat ID: {chat_id}")
            print(f"📊 Event type: {event.get('type')}")
            
            if event.get("type") != "message":
                print(f"⏭️ Pomijam event typu: {event.get('type')}")
                return
            
            text = event.get("text", "")
            author_id = event.get("author_id")
            
            print(f"👤 Author ID: {author_id}")
            print(f"💬 Wiadomość: {text[:100]}...")
            
            if author_id == self.bot_agent_id:
                print(f"⏭️ Ignoruję wiadomość od samego bota")
                return
            
            if not self.bot_agent_id:
                print("❌ Brak ID bota - nie mogę odpowiedzieć")
                return
            
            if not self.connection_ok:
                print(f"⚠️ Brak połączenia z API - nie wysyłam odpowiedzi")
                return
            
            print(f"\n🔍 Próba dołączenia bota do czatu {chat_id}...")
            join_result = await api_client.join_chat(chat_id, self.bot_agent_id)
            
            if not join_result.get("success"):
                print(f"⚠️ Nie udało się dołączyć do czatu - próbuję mimo to...")
            else:
                print(f"✅ Bot dołączył do czatu")
            
            await asyncio.sleep(1)
            
            print(f"\n🤖 Przetwarzanie wiadomości z RAG...")
            response, should_transfer = await self.bot.process_message(text, chat_id)
            print(f"💬 Odpowiedź: {response}")
            print(f"🔄 Transfer: {should_transfer}")
            
            if response:
                if not self.bot_token:
                    print(f"❌ Brak tokena bota - nie mogę wysłać wiadomości")
                else:
                    print(f"📤 Wysyłanie odpowiedzi z tokenem bota...")
                    result = await api_client.send_message_as_bot(chat_id, response, self.bot_token)
                    if result.get("success"):
                        print(f"✅ Odpowiedź wysłana!")
                    else:
                        print(f"❌ Błąd wysyłania: {result.get('error')}")
            
            if should_transfer:
                print(f"🔄 Przekazywanie czatu...")
                transfer_result = await api_client.transfer_chat(chat_id, self.target_group_id, self.bot_agent_id)
                if transfer_result.get("success"):
                    print(f"✅ Czat przekazany")
                else:
                    print(f"❌ Błąd transferu: {transfer_result.get('error')}")
                
        except Exception as e:
            logger.error(f"Błąd w _handle_incoming_event: {e}")
            traceback.print_exc()
    
    async def stop(self):
        self._running = False
        logger.info("Bot zatrzymany")
    
    def is_running(self) -> bool:
        return self._running
    
    def get_bot_agent_id(self) -> Optional[str]:
        return self.bot_agent_id
    
    def is_connected(self) -> bool:
        return self.connection_ok

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.sessions import SessionMiddleware

app = FastAPI(title="Bot LiveChat")

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])
app.add_middleware(SessionMiddleware, secret_key=os.getenv("SECRET_KEY", secrets.token_urlsafe(32)))

templates = Jinja2Templates(directory=os.path.join(os.path.dirname(__file__), "templates"))
bot_integration: Optional[LiveChatBotIntegration] = None

@app.get("/")
async def home():
    return RedirectResponse(url="/panel")

@app.get("/panel")
async def panel(request: Request):
    return templates.TemplateResponse("panel.html", {"request": request})

@app.get("/token-page", response_class=HTMLResponse)
async def token_page(request: Request):
    return templates.TemplateResponse("token-page.html", {"request": request})

@app.get("/test")
async def test():
    return {
        "status": "ok", 
        "message": "Webhook endpoint is accessible", 
        "time": str(datetime.now()),
        "ngrok_url": NGROK_URL,
        "webhook_url": f"{NGROK_URL}/webhook"
    }

@app.post("/webhook")
async def webhook_receiver(request: Request):
    try:
        body = await request.body()
        print(f"\n{'='*60}")
        print(f"📨 RAW WEBHOOK RECEIVED at {datetime.now().strftime('%H:%M:%S')}")
        print(f"Headers: {dict(request.headers)}")
        print(f"Raw body: {body.decode('utf-8', errors='ignore')[:500]}")
        
        try:
            payload = json.loads(body)
            print(f"📦 Parsed JSON: {json.dumps(payload, indent=2)}")
        except json.JSONDecodeError as e:
            print(f"❌ Failed to parse JSON: {e}")
            payload = {}
        
        if bot_integration and bot_integration.is_running():
            print(f"✅ Forwarding to bot handler")
            asyncio.create_task(bot_integration.handle_webhook(payload))
        else:
            print(f"⚠️ Bot not running, ignoring webhook")
            print(f"💡 Start the bot in the panel first: http://localhost:8000/panel")
        
        return JSONResponse(content={"status": "ok"}, status_code=200)
        
    except Exception as e:
        print(f"❌ Webhook error: {e}")
        traceback.print_exc()
        return JSONResponse(content={"status": "error", "message": str(e)}, status_code=500)

@app.get("/api/bot/status")
async def get_status():
    return {
        "running": bot_integration is not None and bot_integration.is_running(),
        "bot_agent_id": bot_integration.get_bot_agent_id() if bot_integration else None,
        "target_group_id": bot_integration.target_group_id if bot_integration else None,
        "connected": bot_integration.is_connected() if bot_integration else False,
        "webhook_url": bot_integration.webhook_url if bot_integration else None
    }

@app.get("/api/bot/config")
async def get_bot_config():
    config = load_bot_config()
    return {
        "has_bot_config": bool(config.get("bot_id")),
        "bot_id": config.get("bot_id"),
        "running_bot_id": bot_integration.get_bot_agent_id() if bot_integration else None
    }

@app.get("/api/auth/check")
async def check_auth():
    try:
        result = await api_client._request(
            "POST",
            "/v3.6/configuration/action/list_agents",
            json={}
        )
        
        if isinstance(result, dict):
            agents = result.get("agents", [])
            return {
                "valid": True,
                "message": f"Token ważny, znaleziono {len(agents)} agentów",
                "agents_count": len(agents)
            }
        else:
            return {
                "valid": True,
                "message": "Token ważny",
                "agents_count": 0
            }
            
    except Exception as e:
        error_str = str(e)
        return {
            "valid": False,
            "message": f"Token nieważny: {error_str[:200]}",
            "error": error_str
        }

@app.get("/api/webhooks/list")
async def list_webhooks():
    try:
        token = await auth_client.get_valid_token()
        
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "owner_client_id": auth_client.client_id
        }
        
        async with httpx.AsyncClient() as client:
            response = await client.post(
                "https://api.livechatinc.com/v3.6/configuration/action/list_webhooks",
                headers=headers,
                json=payload
            )
            result = response.json()
            print(f"📋 Webhooks list: {json.dumps(result, indent=2)}")
            return result
    except Exception as e:
        print(f"❌ Error listing webhooks: {e}")
        return {"error": str(e)}

@app.post("/api/bot/start")
async def start_bot():
    global bot_integration
    if bot_integration and bot_integration.is_running():
        return {"success": True, "message": "Bot już działa"}
    
    your_bot = YourBot()
    bot_integration = LiveChatBotIntegration(your_bot)
    asyncio.create_task(bot_integration.start())
    await asyncio.sleep(2)
    return {"success": True, "message": "Bot uruchomiony"}

@app.post("/api/bot/stop")
async def stop_bot():
    global bot_integration
    if bot_integration:
        await bot_integration.stop()
        bot_integration = None
    return {"success": True, "message": "Bot zatrzymany"}

@app.post("/register-webhook")
async def register_webhook():
    try:
        token = await auth_client.get_valid_token()
        webhook_url = f"{NGROK_URL}/webhook"
        
        print(f"\n🔍 Registering webhook with URL: {webhook_url}")
        
        print("📋 Sprawdzanie istniejących webhooków...")
        list_response = await list_webhooks()
        print(f"Istniejące webhooki: {json.dumps(list_response, indent=2)}")
        
        secret_key = secrets.token_urlsafe(32)
        
        payload = {
            "url": webhook_url,
            "action": "incoming_event",
            "description": "Bot webhook",
            "type": "bot",
            "owner_client_id": auth_client.client_id,
            "secret_key": secret_key
        }
        
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json"
        }
        
        print(f"📦 Rejestracja z payload: {json.dumps(payload, indent=2)}")
        
        async with httpx.AsyncClient() as client:
            response = await client.post(
                "https://api.livechatinc.com/v3.6/configuration/action/register_webhook",
                headers=headers,
                json=payload
            )
            result = response.json()
            print(f"✅ Wynik rejestracji: {json.dumps(result, indent=2)}")
            return result
    except Exception as e:
        print(f"❌ Błąd rejestracji webhooka: {e}")
        traceback.print_exc()
        return {"error": str(e)}

@app.post("/unregister-webhook")
async def unregister_webhook(request: Request):
    try:
        data = await request.json()
        token = await auth_client.get_valid_token()
        
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json"
        }
        
        if "id" in data:
            payload = {
                "id": data["id"],
                "owner_client_id": auth_client.client_id
            }
        elif "url" in data:
            payload = {
                "url": data["url"],
                "owner_client_id": auth_client.client_id
            }
        else:
            return {"error": "Wymagane 'id' lub 'url'"}
        
        print(f"\n🔍 Unregistering webhook with payload: {json.dumps(payload, indent=2)}")
        
        async with httpx.AsyncClient() as client:
            response = await client.post(
                "https://api.livechatinc.com/v3.6/configuration/action/unregister_webhook",
                headers=headers,
                json=payload
            )
            result = response.json()
            print(f"✅ Unregistered webhook: {result}")
            return result
    except Exception as e:
        print(f"❌ Error unregistering webhook: {e}")
        traceback.print_exc()
        return {"error": str(e)}

@app.post("/unregister-all-webhooks")
async def unregister_all_webhooks():
    try:
        token = await auth_client.get_valid_token()
        
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json"
        }
        
        list_response = await list_webhooks()
        
        webhooks = []
        if isinstance(list_response, dict) and "webhooks" in list_response:
            webhooks = list_response["webhooks"]
        elif isinstance(list_response, list):
            webhooks = list_response
        
        removed = []
        errors = []
        
        for webhook in webhooks:
            webhook_id = webhook.get("id") if isinstance(webhook, dict) else None
            
            if webhook_id:
                try:
                    payload = {
                        "id": webhook_id,
                        "owner_client_id": auth_client.client_id
                    }
                    
                    async with httpx.AsyncClient() as client:
                        response = await client.post(
                            "https://api.livechatinc.com/v3.6/configuration/action/unregister_webhook",
                            headers=headers,
                            json=payload
                        )
                        if response.status_code == 202:
                            removed.append(webhook_id)
                            print(f"✅ Usunięto webhook: {webhook_id}")
                        else:
                            errors.append(webhook_id)
                            print(f"❌ Nie udało się usunąć: {webhook_id}")
                except Exception as e:
                    errors.append(webhook_id)
                    print(f"❌ Błąd: {e}")
        
        return {
            "success": True,
            "removed": removed,
            "errors": errors,
            "message": f"Usunięto {len(removed)} webhooków"
        }
    except Exception as e:
        return {"error": str(e)}

@app.post("/create-bot")
async def create_bot_endpoint():
    """Endpoint do tworzenia nowego bota przez API"""
    try:
        bot_id, bot_secret = await api_client.create_bot("ChatbotBMW")
        
        # Automatycznie zapisz do configu
        save_bot_config({
            "bot_id": bot_id,
            "bot_secret": bot_secret
        })
        
        return {
            "success": True,
            "bot_id": bot_id,
            "bot_secret": bot_secret,
            "message": "Nowy bot utworzony i zapisany w bot_config.json"
        }
    except Exception as e:
        return {"error": str(e), "message": "Nie udało się utworzyć bota"}

@app.get("/register-webhook-simple")
async def register_webhook_simple():
    return await register_webhook()

@app.get("/unregister-webhook-by-id/{webhook_id}")
async def unregister_webhook_by_id(webhook_id: str):
    try:
        token = await auth_client.get_valid_token()
        
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "id": webhook_id,
            "owner_client_id": auth_client.client_id
        }
        
        async with httpx.AsyncClient() as client:
            response = await client.post(
                "https://api.livechatinc.com/v3.6/configuration/action/unregister_webhook",
                headers=headers,
                json=payload
            )
            return response.json()
    except Exception as e:
        return {"error": str(e)}

@app.get("/api/bot/routing-status/{bot_id}")
async def get_bot_routing_status(bot_id: str):
    try:
        token = await auth_client.get_valid_token()
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json"
        }
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"https://api.livechatinc.com/v3.6/agent/action/get_routing_status",
                headers=headers,
                json={"id": bot_id}
            )
            return response.json()
    except Exception as e:
        return {"error": str(e)}

# ============================================
# RAG ENDPOINTS
# ============================================

@app.get("/rag/info")
async def get_rag_info():
    """Informacje o RAG"""
    try:
        rag_service = get_rag_service()
        health = await rag_service.health_check()
        stats = await rag_service.get_stats()
        
        return {
            "healthy": health.get("status") == "healthy" and not stats.get("is_dummy", False),
            "chunks": stats.get("total_chunks", 0),
            "vectors": stats.get('total_vectors', 0),
            "embedding_model": stats.get('model_name', "unknown"),
            "status": health.get("status", "unknown"),
            "is_dummy": stats.get("is_dummy", False),
            "available": RAG_AVAILABLE,
            "details": stats
        }
    except Exception as e:
        return {
            "healthy": False,
            "error": str(e),
            "available": RAG_AVAILABLE,
            "is_dummy": True
        }

@app.get("/rag/test")
async def test_rag(query: str = "BMW X5 dane techniczne"):
    """Test RAG z podanym zapytaniem"""
    try:
        rag_service = get_rag_service()
        result = await rag_service.retrieve(query=query, top_k=2)
        
        return {
            "query": query,
            "result": result.to_api_response() if hasattr(result, 'to_api_response') else {"sources": []},
            "documents_count": len(result.documents) if hasattr(result, 'documents') else 0,
            "has_valid_data": result.has_valid_data if hasattr(result, 'has_valid_data') else False,
            "detected_models": result.detected_models if hasattr(result, 'detected_models') else []
        }
    except Exception as e:
        return {"error": str(e)}

@app.on_event("startup")
async def startup():
    print("\n" + "="*60)
    print("🚀 SERWER BOTA LIVECHAT Z RAG")
    print("="*60)
    print("📊 Panel: http://localhost:8000/panel")
    print("📥 Webhook: http://localhost:8000/webhook")
    print("🔑 Token page: http://localhost:8000/token-page")
    print("🧪 Test: http://localhost:8000/test")
    print("🌍 Ngrok URL:", NGROK_URL)
    print("🌍 Webhook URL:", f"{NGROK_URL}/webhook")
    print("📝 Rejestracja webhooka (POST): http://localhost:8000/register-webhook")
    print("📝 Rejestracja webhooka (GET test): http://localhost:8000/register-webhook-simple")
    print("🗑️ Usuń webhook po ID (GET): http://localhost:8000/unregister-webhook-by-id/ID")
    print("🗑️ Usuń wszystkie webhooki (POST): http://localhost:8000/unregister-all-webhooks")
    print("🤖 Utwórz nowego bota (POST): http://localhost:8000/create-bot")
    print("🔍 Sprawdź status routingu bota (GET): http://localhost:8000/api/bot/routing-status/ID_BOTA")
    print("📚 RAG info: http://localhost:8000/rag/info")
    print("🧪 Test RAG: http://localhost:8000/rag/test?query=BMW X5")
    print("="*60)
    
    # Sprawdź RAG
    rag_service = get_rag_service()
    rag_health = await rag_service.health_check()
    rag_stats = await rag_service.get_stats()
    
    if rag_stats.get("is_dummy", False):
        print("⚠️ RAG: DUMMY MODE (baza niezaładowana)")
    else:
        print(f"✅ RAG: {rag_stats.get('total_chunks', 0)} fragmentów")
        print(f"✅ RAG model: {rag_stats.get('model_name', 'unknown')}")
    
    config = load_bot_config()
    if config.get("bot_id"):
        print(f"✅ Bot ID w configu: {config['bot_id']}")
        if config.get("bot_secret"):
            print(f"✅ Bot secret znaleziony w configu")
        else:
            print(f"⚠️ Brak bot_secret w configu - użyj /create-bot aby utworzyć nowego bota")
    else:
        print("❌ BRAK ID BOTA! Użyj /create-bot aby utworzyć nowego bota")
    
    token = auth_client.oauth_token
    if token and token != "NOWY_TOKEN":
        print(f"✅ Token OAuth skonfigurowany: {token[:20]}...")
        
        try:
            result = await api_client._request(
                "POST",
                "/v3.6/configuration/action/list_agents",
                json={}
            )
            if isinstance(result, dict):
                agents = result.get("agents", [])
                print(f"✅ Token działa! Znaleziono {len(agents)} agentów")
            else:
                print(f"✅ Token działa")
        except Exception as e:
            print(f"❌ Token NIE DZIAŁA: {e}")
            print("   Zdobądź nowy token na /token-page")
    else:
        print("❌ BRAK TOKENA OAuth! Wpisz token w linii 93")
    
    print("="*60 + "\n")

@app.on_event("shutdown")
async def shutdown():
    global bot_integration
    if bot_integration:
        await bot_integration.stop()
    await auth_client.close()
    await api_client.close()
    print("👋 Serwer zatrzymany")

async def main():
    logger.info("🚀 Uruchamianie bota z RAG...")
    
    integration = LiveChatBotIntegration(YourBot())
    
    try:
        await integration.start()
        logger.info("✅ Bot zainicjalizowany. Naciśnij Ctrl+C aby zatrzymać.")
        while integration.is_running():
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        await integration.stop()
    finally:
        await auth_client.close()
        await api_client.close()

if __name__ == "__main__":
    if not os.getenv("RUNNING_IN_UVICORN"):
        asyncio.run(main())