"""
Główny plik aplikacji BMW Assistant - ZK Motors Edition.
POPRAWIONA WERSJA - WYSOKA JAKOŚĆ ODPOWIEDZI, RAG JEDYNIE JAKO ŹRÓDŁO DANYCH
"""
import json
import time
import re
from datetime import datetime
from typing import Optional, Dict, Any, List, Tuple
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
# USUNIĘTO: from app.services.prompt_service import PromptService, get_prompt_service
from app.services.cache import init_cache

# ============================================
# IMPORT NASZEGO DZIAŁAJĄCEGO RAG
# ============================================

import sys
import os
from pathlib import Path

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
        
        print(f"Załadowano moduł RAG: {module_name}")
        
        if not hasattr(module, 'RAGSystem'):
            raise AttributeError("Brak klasy RAGSystem w module")
        
        if not hasattr(module, 'find_latest_vector_db'):
            raise AttributeError("Brak funkcji find_latest_vector_db w module")
        
        return module
        
    except Exception as e:
        print(f"Błąd ładowania modułu RAG: {e}")
        raise

try:
    rag_module = import_rag_module()
    RAGSystem = rag_module.RAGSystem
    find_latest_vector_db = rag_module.find_latest_vector_db
    RAG_AVAILABLE = True
    print("RAG system gotowy do użycia")
except Exception as e:
    print(f"Warning: Could not import RAG module: {e}")
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
# RAG SINGLETON
# ============================================

_rag_service_instance = None

def get_rag_service_singleton():
    """Singleton dla RAG service"""
    global _rag_service_instance
    if _rag_service_instance is None:
        print("Tworzę singleton RAG service...")
        _rag_service_instance = SimpleRAGService()
    return _rag_service_instance

# ============================================
# POPRAWIONY RAG SERVICE Z LEPSZĄ WALIDACJĄ
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
            test_result = self.query("BMW X5", k=1, use_model_filter=True)
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

async def get_rag_service():
    """Zwraca singleton RAG service"""
    return get_rag_service_singleton()

# ============================================
# INITIALIZATION
# ============================================

logger = setup_logger(__name__)
security = HTTPBearer(auto_error=False)

BASE_DIR = Path(__file__).parent.absolute()
TEMPLATES_DIR = BASE_DIR / "templates"
STATIC_DIR = BASE_DIR / "static"

conversation_memory: Dict[str, List[Dict]] = {}
MAX_HISTORY = 8

# ============================================
# MODELS
# ============================================

class ChatRequest(BaseModel):
    """Request model dla endpointu chat"""
    message: str = Field(..., min_length=1, max_length=1000)
    session_id: Optional[str] = Field(default="default", description="ID sesji")
    stream: bool = Field(default=False, description="Czy streamować odpowiedź")
    temperature: float = Field(
        default=0.7,
        ge=0.0, 
        le=1.0,
        description="Kreatywność odpowiedzi"
    )
    language: str = Field(
        default="pl",
        pattern="^(pl|en|de)$",
        description="Język odpowiedzi"
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
        description="Pewność odpowiedzi"
    )
    rag_info: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Informacje o RAG"
    )
    data_quality: Optional[str] = Field(
        default=None,
        description="Jakość danych: high/medium/low"
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
# POPRAWIONE FUNKCJE POMOCNICZE
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
    
    if len(history) > MAX_HISTORY:
        conversation_memory[session_id] = history[-MAX_HISTORY:]

def format_history_for_prompt(history: List[Dict]) -> List[Dict[str, str]]:
    """Formatuje historię na format dla PromptService"""
    formatted = []
    for msg in history[-4:]:
        formatted.append({
            "role": msg["role"],
            "content": msg["message"]
        })
    return formatted

def validate_and_correct_bmw_data(text: str, query_type: str = "", sources: List[Dict] = None) -> Tuple[str, str, Dict[str, Any]]:
    """
    Waliduje i poprawia dane BMW w tekście.
    Zwraca: (poprawiony_tekst, jakość_danych, szczegóły_walidacji)
    """
    if sources:
        # Sprawdź jakość źródeł
        source_quality_scores = [s.get('quality', 0.0) for s in sources if s.get('has_technical_data', False)]
        avg_source_quality = sum(source_quality_scores) / len(source_quality_scores) if source_quality_scores else 0.0
        
        if avg_source_quality >= 0.8:
            # Źródła wysokiej jakości - minimalne poprawki
            return text.strip(), "high", {"source_quality": avg_source_quality, "corrections": 0}
    
    text_lower = text.lower()
    corrections = []
    warnings = []
    
    # Tylko stylistyczne poprawki
    style_corrections = [
        (r'koni mechanicznych', 'KM'),
        (r'bmw i', 'BMW i'),
        (r'bmw x', 'BMW X'),
        (r'bmw m', 'BMW M'),
        (r'limuzyna.*?x5', 'SUV X5'),
        (r'x5.*?limuzyna', 'SUV X5'),
    ]
    
    for wrong, correct in style_corrections:
        if re.search(wrong, text, re.IGNORECASE):
            text = re.sub(wrong, correct, text, flags=re.IGNORECASE)
            corrections.append((wrong, correct))
    
    # Sprawdź spójność danych technicznych
    lines = text.split('\n')
    tech_lines = [line for line in lines if any(keyword in line.lower() for keyword in 
                ['km', 'km/h', '0-100', 'silnik', 'moc', 'skrzynia', 'bieg'])]
    
    if len(tech_lines) >= 2:
        # Sprawdź czy dane są spójne
        km_values = []
        for line in tech_lines:
            km_matches = re.findall(r'(\d+)\s*km', line, re.IGNORECASE)
            km_values.extend([int(km) for km in km_matches])
        
        if km_values:
            if max(km_values) > 700:
                warnings.append(f"Nierealistyczna moc: {max(km_values)} KM")
    
    # Określ jakość danych na podstawie treści
    if query_type == "specyfikacje":
        # Dla pytań o specyfikacje, wymagamy dokładności
        has_technical_data = any(keyword in text_lower for keyword in 
                               ['km', 'km/h', '0-100', 'silnik', 'moc', 'skrzynia'])
        
        if has_technical_data and len(corrections) == 0:
            data_quality = "high"
        elif has_technical_data:
            data_quality = "medium"
        else:
            data_quality = "low"
    else:
        # Dla innych pytań mniejsze wymagania
        data_quality = "high" if len(corrections) == 0 else "medium"
    
    validation_details = {
        "corrections_made": len(corrections),
        "warnings": warnings,
        "technical_lines_found": len(tech_lines),
        "has_sources": len(sources) > 0 if sources else False
    }
    
    return text.strip(), data_quality, validation_details

def create_smart_prompt(user_message: str, history: List[Dict], context: str = "", 
                       question_type: str = "", detected_models: List[str] = None) -> str:
    """
    Tworzy inteligentny prompt na podstawie typu pytania
    BEZ DANYCH TECHNICZNYCH - tylko struktura odpowiedzi
    """
    detected_models = detected_models or []
    is_first_message = len(history) == 0
    
    # SYSTEM PROMPT - tylko ogólne instrukcje
    system_prompt = """Jesteś Leo - ekspertem BMW w ZK Motors, oficjalnym dealerze BMW i MINI.

ZASADY ODPOWIADANIA:
1. Odpowiadaj KROTKO i konkretnie - maksymalnie 3-4 zdania lub punktory
2. Bądź rzetelny i profesjonalny
3. Używaj poprawnych terminów motoryzacyjnych
4. Jeśli nie masz pewności - odwołaj się do źródła lub zaproś do salonu
5. Zawsze kończ zachętą do kontaktu z ZK Motors"""

    if is_first_message:
        system_prompt += "\nPrzywitaj się krótko i zapytaj czym możesz pomóc."
    
    # SPECJALNE INSTRUKCJE DLA RÓŻNYCH TYPÓW PYTAŃ (BEZ DANYCH!)
    special_instructions = ""
    
    if question_type == "specyfikacje":
        special_instructions = """INSTRUKCJE DLA SPECYFIKACJI:
• Używaj DANYCH Z ŹRÓDEŁ - nie wymyślaj!
• Przedstaw dane techniczne w punktach
• Podkreśl kluczowe parametry: silnik, moc, skrzynia, napęd
• Wskaż różnice między wersjami
• Jeśli dane się różnią - zaznacz to"""
    
    elif question_type == "porownanie":
        special_instructions = """INSTRUKCJE DLA PORÓWNANIA:
• Wymień 2-3 kluczowe różnice między modelami
• Porównaj: przeznaczenie, rozmiar, charakterystykę
• Zachęć do wizyty w salonie aby zobaczyć różnice"""
    
    elif question_type == "rodzinny":
        special_instructions = """INSTRUKCJE DLA PYTAŃ RODZINNYCH:
• Polecaj modele odpowiednie dla rodzin
• Wymień praktyczne cechy: przestrzeń, bezpieczeństwo, funkcjonalność
• Zapytaj o szczególne potrzeby (liczba osób, bagaż, itp.)"""
    
    elif question_type == "sportowy":
        special_instructions = """INSTRUKCJE DLA SPORTOWYCH:
• Polecaj modele z serii M i sportowe
• Opisz charakterystykę jazdy BMW M
• Podkreśl osiągi i doświadczenie z jazdy"""
    
    elif question_type == "elektryczny":
        special_instructions = """INSTRUKCJE DLA ELEKTRYCZNYCH:
• Polecaj modele z serii i (i4, i5, i7, iX)
• Opisz zalety elektryków BMW
• Wymień kluczowe parametry: zasięg, ładowanie, technologie"""
    
    elif question_type == "cena":
        special_instructions = """INSTRUKCJE DLA CEN:
• Nie podawaj dokładnych cen - zmieniają się!
• Wymień zakres cenowy jeśli znasz ze źródeł
• Podkreśl, że cena zależy od wyposażenia
• Zaproś do salonu po indywidualną wycenę"""
    
    # WYKRYCIE INNYCH MAREK
    user_query_lower = user_message.lower()
    other_brands = ['audi', 'mercedes', 'tesla', 'volvo', 'skoda', 'toyota']
    detected_other_brand = None
    
    for brand in other_brands:
        if brand in user_query_lower:
            detected_other_brand = brand.capitalize()
            break
    
    if detected_other_brand:
        special_instructions += f"""

PYTANIE DOTYCZY MARKI {detected_other_brand.upper()}:
• Jesteś ekspertem BMW - skup się na BMW
• Podkreśl mocne strony BMW w porównaniu
• Zaproś do salonu ZK Motors gdzie można zobaczyć różnice"""
    
    # KONTEKST Z RAG (jeśli jest)
    context_part = ""
    if context:
        context_part = f"DANE Z BAZY WIEDZY:\n{context}\n\nUWAGA: Używaj tych danych w odpowiedzi!"
    else:
        context_part = "BRAK DANYCH W BAZE - odpowiadaj ogólnie lub zaproś do salonu po szczegóły."
    
    # HISTORIA
    history_part = ""
    if history and not is_first_message:
        recent = history[-2:]
        history_lines = []
        for h in recent:
            role = "Klient" if h['role'] == 'user' else 'Ty'
            history_lines.append(f"{role}: {h['content']}")
        history_part = f"OSTATNIA ROZMOWA:\n" + "\n".join(history_lines)
    
    # PYTANIE
    question_part = f"PYTANIE KLIENTA: \"{user_message}\""
    
    # INFORMACJE O WYKRYTYCH MODELACH (tylko informacyjnie)
    models_info = ""
    if detected_models:
        models_info = f"WYKRYTE MODELE: {', '.join(detected_models)}"
    
    # FINALNE INSTRUKCJE
    final_instructions = f"""STRUKTURA ODPOWIEDZI:
1. Odpowiedz bezpośrednio na pytanie
2. Użyj danych z kontekstu jeśli są dostępne
3. Bądź konkretny ale nie techniczny bez potrzeby
4. Jeśli brakuje danych - zaproś do kontaktu z ZK Motors
5. Używaj punktów • dla list
6. Zakończ zachętą do test drive w ZK Motors

{models_info}

ODPOWIEDŹ (po polsku, 3-4 zdania):"""

    # SKŁADAJ PROMPT
    prompt_parts = [
        system_prompt,
        special_instructions,
        context_part,
        history_part,
        question_part,
        final_instructions
    ]
    
    return "\n\n".join(filter(None, prompt_parts))

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

if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["*"]
)

@app.middleware("http")
async def add_cors_middleware(request: Request, call_next):
    response = await call_next(request)
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
    chat_html_path = TEMPLATES_DIR / "chat.html"
    
    if not chat_html_path.exists():
        return HTMLResponse(f"""
        <!DOCTYPE html>
        <html>
        <head><title>Error - File Not Found</title></head>
        <body>
            <h1>Error: chat.html not found</h1>
            <p>Expected path: {chat_html_path}</p>
        </body>
        </html>
        """)
    
    try:
        with open(chat_html_path, 'r', encoding='utf-8') as f:
            html_content = f.read()
        
        logger.info(f"Loaded chat.html from {chat_html_path}")
        
        js_config = f"""
        <script>
            window.API_BASE_URL = window.location.origin;
            window.API_ENDPOINTS = {{
                chat: '/chat',
                health: '/health',
                ping: '/ping',
                reset: '/chat/reset',
                status: '/api/status',
                rag_info: '/rag/info'
            }};
            console.log('API Base URL:', window.API_BASE_URL);
        </script>
        """
        
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
        </body>
        </html>
        """)

@app.get("/ping")
async def ping():
    return {"status": "online", "timestamp": datetime.utcnow().isoformat()}

@app.get("/api/status")
async def api_status():
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
    try:
        rag_health = await rag_service.health_check()
        llm_health = {"status": "operational"}
        
        services_status = {
            "rag_system": rag_health.get("status", "unknown"),
            "llm_service": llm_health.get("status", "unknown"),
            "api": "healthy",
            "cache": "connected" if settings.REDIS_URL else "in_memory",
            "memory": "enabled"
        }
        
        critical_services = ["rag_system", "llm_service", "api"]
        all_critical_healthy = all(
            services_status[s] in ["healthy", "operational"]
            for s in critical_services
        )
        
        overall_status = "healthy" if all_critical_healthy else "degraded"
        
        memory_stats = {
            "active_sessions": len(conversation_memory),
            "total_messages": sum(len(h) for h in conversation_memory.values()),
            "max_history": MAX_HISTORY
        }
        
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
    try:
        health = await rag_service.health_check()
        stats = await rag_service.get_stats()
        
        return {
            "healthy": health.get("status") == "healthy" and not stats.get("is_dummy", False),
            "chunks": stats.get("total_chunks", 0),
            "vectors": stats.get('total_vectors', 0),
            "embedding_model": stats.get('model_name', "unknown"),
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
        }
    ]
    
    return {
        "models": models,
        "default_model": settings.COHERE_CHAT_MODEL,
        "memory_enabled": True,
        "max_history": MAX_HISTORY,
        "rag_available": RAG_AVAILABLE
    }

# ============================================
# POPRAWIONY CHAT ENDPOINT
# ============================================

@app.post("/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    background_tasks: BackgroundTasks,
    rag_service: SimpleRAGService = Depends(get_rag_service),
    llm_service: LLMService = Depends(get_llm_service)
):
    """
    Główny endpoint chat - używa RAG jako jedynego źródła danych technicznych
    """
    start_time = time.time()
    
    try:
        session_id = request.session_id
        user_message = request.message
        user_query_lower = user_message.lower()
        
        logger.info(f"Chat request: {user_message[:50]}...")
        
        # 1. Pobierz historię
        history = get_conversation_history(session_id)
        conversation_history = format_history_for_prompt(history)
        is_first_message = len(history) == 0
        
        # 2. Wykryj typ pytania
        question_type = "general"
        if any(word in user_query_lower for word in ['specyfikacj', 'dane', 'parametr', 'silnik', 'moc']):
            question_type = "specyfikacje"
        elif any(word in user_query_lower for word in ['różni', 'różnica', 'porównaj', 'vs', 'contra']):
            question_type = "porownanie"
        elif any(word in user_query_lower for word in ['rodzin', 'dzieci', 'osób']):
            question_type = "rodzinny"
        elif any(word in user_query_lower for word in ['sport', 'szybk', 'mocny']):
            question_type = "sportowy"
        elif any(word in user_query_lower for word in ['elektryczn', 'ev', 'elektryk']):
            question_type = "elektryczny"
        elif any(word in user_query_lower for word in ['cen', 'koszt', 'drogi']):
            question_type = "cena"
        
        # 3. Wykryj modele BMW
        bmw_models = ['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'xm',
                     'i3', 'i4', 'i5', 'i7', 'i8', 'ix',
                     'm2', 'm3', 'm4', 'm5', 'm8', 'z4',
                     'seria 2', 'seria 3', 'seria 4', 'seria 5', 'seria 7', 'seria 8']
        
        detected_models = []
        for model in bmw_models:
            if model in user_query_lower:
                model_upper = model.upper()
                if 'SERIA' in model_upper:
                    model_upper = model_upper.replace('SERIA', 'Seria')
                detected_models.append(model_upper)
        
        # 4. Użyj RAG dla pytań technicznych
        needs_rag = False
        context_text = ""
        rag_result = None
        sources_count = 0
        confidence_score = 0.8
        
        # RAG tylko dla pytań wymagających danych
        rag_needed_types = ["specyfikacje", "porownanie", "elektryczny", "sportowy", "cena"]
        if question_type in rag_needed_types or detected_models:
            needs_rag = True
        
        if needs_rag:
            rag_result = await rag_service.retrieve(
                query=user_message,
                top_k=3,
                similarity_threshold=0.5
            )
            
            if hasattr(rag_result, 'documents') and rag_result.documents:
                # Weź tylko zwalidowane dokumenty z danymi technicznymi
                valid_docs = [
                    d for d in rag_result.documents 
                    if d.get('validated', False) and d.get('quality_score', 0) >= 0.6
                ]
                sources_count = len(valid_docs)
                
                if valid_docs:
                    # Przygotuj kontekst z najlepszych źródeł
                    context_parts = []
                    for i, doc in enumerate(valid_docs[:2], 1):
                        content = doc['content']
                        # Skróć ale zachowaj kluczowe informacje
                        if len(content) > 250:
                            # Znajdź kluczowe zdania
                            sentences = re.split(r'[.!?]+', content)
                            key_sentences = []
                            keywords = ['km', 'km/h', '0-100', 'silnik', 'moc', 'skrzynia', 'napęd']
                            
                            for sentence in sentences:
                                if any(keyword in sentence.lower() for keyword in keywords):
                                    key_sentences.append(sentence.strip())
                            
                            if key_sentences:
                                content = ' '.join(key_sentences[:3])
                        
                        metadata = doc.get('metadata', {})
                        source = metadata.get('title', 'Źródło')[:50]
                        quality = f" (jakość: {doc.get('quality_score', 0):.1f})"
                        
                        context_parts.append(f"{i}. [{source}{quality}]: {content[:200]}...")
                    
                    context_text = "\n\n".join(context_parts)
                    
                    # Oblicz pewność na podstawie jakości źródeł
                    quality_scores = [d.get('quality_score', 0) for d in valid_docs]
                    confidence_score = sum(quality_scores) / len(quality_scores)
        
        # 5. Stwórz prompt BEZ DANYCH TECHNICZNYCH
        prompt = create_smart_prompt(
            user_message=user_message,
            history=conversation_history,
            context=context_text if context_text else "BRAK DANYCH TECHNICZNYCH W BAZIE",
            question_type=question_type,
            detected_models=detected_models
        )
        
        # 6. Generuj odpowiedź
        try:
            llm_result = await llm_service.generate(
                prompt=prompt,
                model=settings.COHERE_CHAT_MODEL,
                temperature=0.7,
                max_tokens=500  # Więcej tokenów dla bardziej szczegółowych odpowiedzi
            )
            
            # Wyodrębnij tekst
            if hasattr(llm_result, 'text'):
                response_text = llm_result.text
            elif isinstance(llm_result, dict) and 'text' in llm_result:
                response_text = llm_result['text']
            elif isinstance(llm_result, dict) and 'generations' in llm_result:
                response_text = llm_result['generations'][0]['text']
            else:
                response_text = str(llm_result)
            
            # 7. OCZYŚĆ ODPOWIEDŹ
            # Usuń powtarzające się frazy
            cleanup_patterns = [
                r'Dziękuję.*?za pytanie.*?\.',
                r'Zapraszam.*?do kontaktu.*?salonu.*?\.',
                r'Jestem Leo.*?ZK Motors.*?\.',
                r'Specjalizuję się.*?BMW.*?\.',
            ]
            
            for pattern in cleanup_patterns:
                response_text = re.sub(pattern, '', response_text, flags=re.IGNORECASE)
            
            # Usuń nadmiarowe białe znaki
            response_text = re.sub(r'\n\s*\n+', '\n\n', response_text)
            response_text = re.sub(r'\s+', ' ', response_text)
            response_text = response_text.strip()
            
            # Dodaj przywitanie jeśli pierwsza wiadomość
            if is_first_message and not response_text.startswith(("Cześć", "Dzień dobry", "Witaj", "Hej")):
                response_text = f"Cześć! Jestem Leo, ekspert BMW w ZK Motors.\n\n{response_text}"
            
            # 8. WALIDUJ odpowiedź na podstawie źródeł RAG
            sources_data = []
            if rag_result and hasattr(rag_result, 'to_api_response'):
                sources_response = rag_result.to_api_response()
                sources_data = sources_response.get("sources", [])
            
            corrected_text, data_quality, validation_details = validate_and_correct_bmw_data(
                response_text, 
                question_type,
                sources_data
            )
            
            # Jeśli słaba jakość i mamy źródła, dodaj informację
            if data_quality == "low" and sources_data:
                corrected_text += "\n\nℹ️ *Dane oparte na dostępnych źródłach. Dla najświeższych informacji zapraszam do salonu ZK Motors.*"
            
            tokens_used = None
            if hasattr(llm_result, 'tokens_used'):
                tokens_used = llm_result.tokens_used
            elif isinstance(llm_result, dict) and 'tokens_used' in llm_result:
                tokens_used = llm_result['tokens_used']
                
        except Exception as llm_error:
            logger.error(f"LLM error: {str(llm_error)}")
            
            # FALLBACK odpowiedzi na podstawie RAG jeśli dostępny
            if rag_result and hasattr(rag_result, 'documents') and rag_result.documents:
                # Stwórz odpowiedź z danych RAG
                valid_docs = [d for d in rag_result.documents if d.get('validated', False)]
                if valid_docs:
                    doc = valid_docs[0]
                    content = doc['content'][:200] + "..." if len(doc['content']) > 200 else doc['content']
                    corrected_text = f"""Cześć!

Na podstawie dostępnych danych:
• {content}

Dla dokładnych i aktualnych informacji zapraszam do salonu ZK Motors!"""
                else:
                    corrected_text = "Cześć! Jestem Leo, ekspertem BMW w ZK Motors. W czym mogę pomóc?"
            else:
                corrected_text = "Cześć! Jestem Leo, ekspertem BMW w ZK Motors. W czym mogę pomóc?"
            
            data_quality = "medium"
            validation_details = {"fallback_used": True, "llm_error": str(llm_error)}
            tokens_used = None
        
        # 9. Dodaj do historii
        add_to_history(session_id, "user", user_message)
        add_to_history(session_id, "assistant", corrected_text)
        
        # 10. Przygotuj odpowiedź
        processing_time = time.time() - start_time
        
        # 11. Przygotuj źródła do odpowiedzi
        sources = []
        if rag_result and hasattr(rag_result, 'to_api_response'):
            sources_response = rag_result.to_api_response()
            sources = sources_response.get("sources", [])[:2]  # Maks 2 źródła w odpowiedzi
        
        # 12. Stwórz odpowiedź API
        response = ChatResponse(
            answer=corrected_text,
            session_id=session_id,
            history_length=len(get_conversation_history(session_id)),
            processing_time=processing_time,
            sources=sources,
            model_used=settings.COHERE_CHAT_MODEL,
            tokens_used=tokens_used,
            confidence=confidence_score,
            rag_info={
                "question_type": question_type,
                "detected_models": detected_models,
                "sources_count": sources_count,
                "data_validated": True,
                "validation_details": validation_details,
                "used_rag": needs_rag
            },
            data_quality=data_quality
        )
        
        # 13. Loguj w tle
        background_tasks.add_task(
            log_interaction,
            user_message=user_message,
            assistant_response=corrected_text,
            session_id=session_id,
            sources_count=sources_count,
            tokens_used=tokens_used,
            processing_time=processing_time,
            confidence=confidence_score,
            rag_info={"question_type": question_type, "models": detected_models}
        )
        
        logger.info(f"Response in {processing_time:.2f}s, quality: {data_quality}", extra={
            "question_type": question_type,
            "models": detected_models,
            "used_rag": needs_rag,
            "sources": sources_count
        })
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Chat error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Błąd serwera" if settings.IS_PRODUCTION else str(e)
        )

@app.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    async def event_generator():
        yield f"data: {json.dumps({'error': 'Streaming not implemented'})}\n\n"
        yield "data: [DONE]\n\n"
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache"}
    )

@app.post("/chat/reset", response_model=ResetResponse)
async def reset_chat(session_id: str = "default"):
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
    history = get_conversation_history(session_id)
    
    return HistoryResponse(
        session_id=session_id,
        history=history[-limit:],
        total_messages=len(history),
        limit=limit
    )

# ============================================
# DEBUG ENDPOINTS
# ============================================

@app.get("/debug/rag")
async def debug_rag(
    query: str,
    top_k: int = 3,
    rag_service: SimpleRAGService = Depends(get_rag_service)
):
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
        "average_similarity": result.average_similarity if hasattr(result, 'average_similarity') else 0,
        "has_valid_data": result.has_valid_data if hasattr(result, 'has_valid_data') else False,
        "quality_scores": result.quality_scores if hasattr(result, 'quality_scores') else []
    }

@app.get("/debug/stats")
async def debug_stats(
    rag_service: SimpleRAGService = Depends(get_rag_service),
    llm_service: LLMService = Depends(get_llm_service)
):
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
    try:
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "session_id": session_id,
            "user_message_preview": user_message[:100],
            "assistant_response_preview": assistant_response[:100],
            "sources_count": sources_count,
            "processing_time": round(processing_time, 2),
            "confidence": round(confidence, 2) if confidence else None,
            "question_type": rag_info.get('question_type', 'unknown') if rag_info else 'unknown'
        }
        
        logger.info(f"Interaction logged", extra=log_entry)
        
    except Exception as e:
        logger.error(f"Failed to log interaction: {str(e)}")

# ============================================
# APPLICATION EVENTS
# ============================================

@app.on_event("startup")
async def startup_event():
    try:
        validate_configuration()
        await init_cache()
        
        rag_info = "NOT AVAILABLE"
        if RAG_AVAILABLE:
            try:
                if RAG_FILE_PATH.exists():
                    rag_service = get_rag_service_singleton()
                    rag_health = await rag_service.health_check()
                    rag_stats = await rag_service.get_stats()
                    
                    if rag_stats.get("is_dummy", False):
                        rag_info = f"DUMMY MODE"
                    else:
                        rag_info = f"LOADED ({rag_stats.get('total_chunks', 0)} chunks)"
                else:
                    rag_info = f"FILE NOT FOUND: {RAG_FILE_PATH.name}"
            except Exception as rag_error:
                rag_info = f"ERROR: {str(rag_error)[:50]}"
        else:
            rag_info = "IMPORT FAILED"
        
        logger.info(f"{settings.APP_NAME} v{settings.APP_VERSION} starting up...")
        logger.info(f"Environment: {settings.ENVIRONMENT}")
        logger.info(f"LLM Model: {settings.COHERE_CHAT_MODEL}")
        logger.info(f"RAG: {rag_info}")
        logger.info(f"Memory: {MAX_HISTORY} messages per session")
        logger.info(f"API: http://{settings.HOST}:{settings.PORT}")
        logger.info(f"Chat: http://{settings.HOST}:{settings.PORT}/")
        
        chat_html_path = TEMPLATES_DIR / "chat.html"
        if chat_html_path.exists():
            logger.info(f"HTML Interface: chat.html found")
        else:
            logger.warning(f"HTML Interface: chat.html NOT FOUND")
        
        logger.info("Application started successfully")
        
        if settings.IS_DEVELOPMENT:
            logger.warning("Running in DEVELOPMENT mode")
        
    except Exception as e:
        logger.critical(f"Startup failed: {str(e)}", exc_info=True)
        raise

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Shutting down application...")
    logger.info(f"Memory stats: {len(conversation_memory)} sessions, {sum(len(h) for h in conversation_memory.values())} total messages")

# ============================================
# ERROR HANDLERS
# ============================================

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
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
            print(f"Model: {settings.COHERE_CHAT_MODEL}")
            print(f"RAG: {'AVAILABLE' if RAG_AVAILABLE else 'NOT AVAILABLE'}")
            print(f"Data validation: ENABLED")
            print(f"Memory: {MAX_HISTORY} messages per session")
            print(f"Test: http://{settings.HOST}:{settings.PORT}/ping")
            print(f"{'='*60}\n")
        
        uvicorn.run(**config)
        
    except Exception as e:
        print(f"Failed to start: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()