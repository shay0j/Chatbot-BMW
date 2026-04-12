import os
import asyncio
import json
import base64
import hmac
import hashlib
import secrets
import traceback
import sys
import re
from typing import Optional, Dict, Any, List, Tuple
from datetime import datetime
from pathlib import Path

import httpx
import cohere
from loguru import logger
from dotenv import load_dotenv
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse, RedirectResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.sessions import SessionMiddleware

# ============================================
# IMPORT RAG SERVICE
# ============================================

current_dir = Path(__file__).parent.absolute()
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

try:
    from app.services.rag_service_faiss import get_rag_service as get_rag_service_new
    from app.services.rag_service_faiss import RAGService
    RAG_AVAILABLE = True
    print("✅ RAG service (FAISS) załadowany")
except Exception as e:
    print(f"⚠️ Ostrzeżenie: Could not import new RAG module: {e}")
    RAG_AVAILABLE = False

    class RAGService:
        async def retrieve_with_intent_check(self, query, top_k=3, confidence_threshold=0.5):
            return {
                "has_data": False,
                "skip_rag": False,
                "below_threshold": True,
                "confidence": 0.0,
                "intent": "general",
                "detected_models": [],
                "tech": False,
                "documents": [],
                "sources": []
            }
        async def health_check(self):
            return {"status": "unavailable", "is_dummy": True}
        async def get_stats(self):
            return {"total_chunks": 0, "is_dummy": True}

    async def get_rag_service_new():
        return RAGService()

# ============================================
# RAG SERVICE SINGLETON
# ============================================

_rag_service_instance = None

async def get_rag_service():
    global _rag_service_instance
    if _rag_service_instance is None:
        print("Tworzę singleton RAG service...")
        _rag_service_instance = await get_rag_service_new()
    return _rag_service_instance

# ============================================
# KONFIGURACJA
# ============================================
load_dotenv()

print("\n" + "="*60)
print("🔍 DEBUG - Environment variables:")
print(f"CRISP_IDENTIFIER: {os.getenv('CRISP_IDENTIFIER')}")
print(f"CRISP_KEY: {os.getenv('CRISP_KEY')[:5] if os.getenv('CRISP_KEY') else 'None'}...")
print(f"CRISP_WEBHOOK_SECRET: {os.getenv('CRISP_WEBHOOK_SECRET')[:5] if os.getenv('CRISP_WEBHOOK_SECRET') else 'None'}...")
print(f"COHERE_API_KEY: {os.getenv('COHERE_API_KEY')[:5] if os.getenv('COHERE_API_KEY') else 'None'}...")
print(f"COHERE_MODEL: {os.getenv('COHERE_MODEL', 'command-a-03-2025')}")
print("="*60 + "\n")

# ============================================
# CRISP KONFIGURACJA
# ============================================
CRISP_IDENTIFIER = os.getenv("CRISP_IDENTIFIER")
CRISP_KEY = os.getenv("CRISP_KEY")
CRISP_WEBHOOK_SECRET = os.getenv("CRISP_WEBHOOK_SECRET")

if not CRISP_IDENTIFIER or not CRISP_KEY:
    logger.error("❌ CRISP_IDENTIFIER lub CRISP_KEY nie znalezione w .env - bot nie zadziała!")
    exit(1)

BASE_URL = os.getenv("BASE_URL", "https://crinklier-ruddily-leonore.ngrok-free.dev")

# ============================================
# FUNKCJE POMOCNICZE CRISP
# ============================================

# Deduplikacja wiadomości
_last_processed = {}

def is_duplicate(session_id: str, message: str, timestamp: int) -> bool:
    """Sprawdza czy wiadomość była już przetwarzana"""
    key = f"{session_id}:{message}"
    if key in _last_processed:
        last_time = _last_processed[key]
        if timestamp - last_time < 2000:
            return True
    _last_processed[key] = timestamp
    if len(_last_processed) > 100:
        for k in list(_last_processed.keys()):
            if _last_processed[k] < timestamp - 10000:
                del _last_processed[k]
    return False

def verify_crisp_signature(payload: bytes, signature: str, timestamp: str, secret: str) -> bool:
    if not secret or not signature:
        return False
    try:
        message = timestamp.encode('utf-8') + payload
        expected_signature = hmac.new(
            key=secret.encode('utf-8'),
            msg=message,
            digestmod=hashlib.sha256
        ).hexdigest()
        return hmac.compare_digest(expected_signature, signature)
    except Exception as e:
        print(f"❌ Błąd weryfikacji sygnatury: {e}")
        return False

async def send_crisp_message(website_id: str, session_id: str, text: str) -> Dict[str, Any]:
    try:
        auth_str = f"{CRISP_IDENTIFIER}:{CRISP_KEY}"
        auth_b64 = base64.b64encode(auth_str.encode()).decode()

        headers = {
            "Authorization": f"Basic {auth_b64}",
            "Content-Type": "application/json",
            "X-Crisp-Tier": "plugin",
            "ngrok-skip-browser-warning": "true"
        }

        payload = {
            "type": "text",
            "from": "operator",
            "origin": "chat",
            "content": text
        }

        url = f"https://api.crisp.chat/v1/website/{website_id}/conversation/{session_id}/message"

        async with httpx.AsyncClient() as client:
            response = await client.post(url, headers=headers, json=payload)
            
            if response.status_code == 200 or response.status_code == 202:
                return {"success": True}
            else:
                return {"success": False, "error": response.text}

    except Exception as e:
        print(f"❌ Crisp wyjątek przy wysyłaniu: {e}")
        traceback.print_exc()
        return {"success": False, "error": str(e)}

# ============================================
# COHERE SERVICE
# ============================================

class CohereService:
    def __init__(self):
        self.api_key = os.getenv("COHERE_API_KEY")
        self.model = os.getenv("COHERE_MODEL", "command-a-03-2025")
        
        if self.api_key:
            try:
                self.client = cohere.ClientV2(self.api_key)
                print(f"✅ Cohere zainicjalizowany (model: {self.model})")
            except AttributeError:
                self.client = cohere.Client(self.api_key, v2=True)
                print(f"✅ Cohere zainicjalizowany (model: {self.model})")
        else:
            self.client = None
            print("⚠️ COHERE_API_KEY brak")

    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 500
    ) -> Dict[str, Any]:
        if not self.client or not self.api_key:
            return {"success": False, "text": "Brak API"}

        try:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})

            response = await asyncio.to_thread(
                self.client.chat,
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )

            if response and hasattr(response, 'message'):
                if hasattr(response.message, 'content') and len(response.message.content) > 0:
                    text = response.message.content[0].text
                else:
                    text = str(response.message)
                return {"success": True, "text": text, "model": self.model}
            else:
                return {"success": False, "text": "Brak odpowiedzi od Cohere"}

        except Exception as e:
            logger.error(f"Cohere error: {e}")
            return {"success": False, "text": f"Błąd Cohere: {str(e)}"}

# ============================================
# DANE KONTAKTOWE
# ============================================

SALON_HOURS = "poniedziałek-piątek 9:00-18:00, sobota 9:00-15:00"

SALON_CONTACT = """
SALONY ZK MOTORS:
- Kielce: ul. Wystawowa 2, tel +48 734 188 400
- Radom: ul. Warszawska 234, tel +48 734 188 500
- Rzeszów: ul. Krasne 9a, tel +48 734 132 100

SERWISY ZK MOTORS:
- Kielce: ul. Wystawowa 2, tel +48 734 188 420
- Radom: ul. Warszawska 234, tel +48 734 188 500
- Rzeszów: ul. Krasne 9a, tel +48 734 132 120

GODZINY OTWARCIA: pon-pt 9:00-18:00, sob 9:00-15:00
"""

# ============================================
# HARDCODOWANE ODPOWIEDZI
# ============================================

def get_greeting() -> str:
    """Zwraca wiadomość powitalną"""
    return """👋 Witaj w ZK Motors! 🚗

Jestem Leo, Twój wirtualny doradca BMW. Specjalizuję się w doborze idealnego modelu do Twoich potrzeb, a także w usługach salonu i serwisu ZK Motors.

Chętnie pomogę Ci wybrać auto, porównać modele, odpowiem na pytania o osiągi, cenę lub umówię Cię na jazdę próbną.

Zapraszam do rozmowy! 😊"""

def get_motorcycle_response() -> str:
    """Odpowiedź na pytania o motocykle"""
    return """🏍️ Zapraszamy do zapoznania się z ofertą motocykli BMW:

🆕 **Nowe motocykle BMW:**
https://stok.zkmotors.pl/pojazd/lista?PojazdSearch%5Brodzaj_id%5D=2&PojazdSearch%5Bstatus_id%5D=1

🔄 **Używane motocykle BMW:**
https://stok.zkmotors.pl/pojazd/lista?PojazdSearch%5Bstatus_id%5D=3

🛠️ **Akcesoria do motocykli BMW:**
https://fliphtml5.com/bookcase/ooqtq

Więcej informacji udzielą nasi doradcy w salonach ZK Motors. 🏍️

Dane kontaktowe:
- Kielce: tel +48 734 188 400
- Radom: tel +48 734 188 500
- Rzeszów: tel +48 734 132 100"""

def get_mini_response() -> str:
    """Odpowiedź na pytania o MINI - odsyła do salonu"""
    return """Przepraszam, w tej chwili nie mam jeszcze szczegółowych informacji o samochodach MINI. 🚗

Zachęcam do kontaktu z naszym salonem ZK Motors - nasi doradcy chętnie odpowiedzą na wszystkie pytania dotyczące MINI.

Dane kontaktowe:
- Kielce: tel +48 734 188 400
- Radom: tel +48 734 188 500
- Rzeszów: tel +48 734 132 100

Godziny otwarcia: pon-pt 9:00-18:00, sob 9:00-15:00"""

def get_accessories_response() -> str:
    """Odpowiedź na pytania o akcesoria"""
    return """🛠️ **Akcesoria i części BMW:**

**Sklep z oryginalnymi częściami:**
https://sklep-bmw.pl/

**Katalog akcesoriów (samochody):**
https://online.fliphtml5.com/pjkxj/bnih/

**Katalog akcesoriów (motocykle):**
https://fliphtml5.com/bookcase/ooqtq

Zapraszamy do zakupów! 🔧"""

def get_catalogs_response() -> str:
    """Odpowiedź na pytania o katalogi modeli"""
    return """📚 **Katalogi modeli BMW i MINI:**

https://fliphtml5.com/bookcase/iaioz

Znajdziesz tam katalogi wszystkich modeli BMW i MINI. Zapraszamy do przeglądania! 🚗"""

def get_configurator_response() -> str:
    """Odpowiedź na pytania o konfigurator"""
    return """⚙️ **Konfigurator BMW:**

https://www.bmw.pl/pl/konfigurator.html

Możesz tam skonfigurować wymarzone BMW - wybrać model, kolor, felgi, wyposażenie i wiele więcej!

**Oferty i promocje:**
https://www.bmw.pl/pl/Shop-Online/bmw-oferty.html#bmw-m

Zapraszamy do konfiguracji! 🚗"""

def get_available_models_response() -> str:
    """Odpowiedź na pytania o dostępne modele"""
    return """🚗 **Sprawdź dostępne pojazdy w ZK Motors:**

**Nowe samochody BMW:**
https://stok.zkmotors.pl/pojazd/lista?PojazdSearch%5Bmarka_id%5D=1&PojazdSearch%5Brodzaj_id%5D=1&PojazdSearch%5Bstatus_id%5D=1

**Używane BMW:**
https://najlepszeoferty.bmw.pl/uzywane/

**Nowe motocykle BMW:**
https://stok.zkmotors.pl/pojazd/lista?PojazdSearch%5Brodzaj_id%5D=2&PojazdSearch%5Bstatus_id%5D=1

**Używane motocykle BMW:**
https://stok.zkmotors.pl/pojazd/lista?PojazdSearch%5Bstatus_id%5D=3

Aktualny stan magazynowy zmienia się dynamicznie. Zapraszamy do kontaktu z salonem po szczegóły! 🚙"""

# ============================================
# GŁÓWNA KLASA BOTA
# ============================================

class CrispBot:
    def __init__(self):
        logger.info("Inicjalizacja bota Crisp z RAG i Cohere...")
        self.conversation_state = {}
        self.rag_service = None
        self.cohere = CohereService()
        logger.info("✅ Bot gotowy")

    async def _ensure_rag(self):
        if self.rag_service is None:
            try:
                self.rag_service = await get_rag_service()
                health = await self.rag_service.health_check()
                if health.get("status") == "healthy":
                    logger.info("✅ RAG zainicjalizowany")
                else:
                    logger.warning(f"⚠️ RAG w stanie: {health.get('status')}")
            except Exception as e:
                logger.error(f"❌ RAG init failed: {e}")
                self.rag_service = None
        return self.rag_service

    def _get_conversation_context(self, state: Dict, last_n: int = 3) -> str:
        """Pobiera ostatnie N wymian konwersacji dla kontekstu.
        
        OPTYMALIZACJA: Używa TYLKO wiadomości użytkownika aby zapobiec
        efektowi 'snowball' — gdzie odpowiedzi bota (np. o leasingu) 
        wracają do RAG i powodują więcej wyników o leasingu.
        """
        if not state.get("context"):
            return ""
        
        # ZMIANA: Tylko wiadomości użytkownika trafiają do kontekstu RAG
        user_messages = [msg for msg in state["context"] if msg["role"] == "user"]
        recent_user = user_messages[-last_n:] if len(user_messages) > last_n else user_messages
        
        context_parts = []
        for msg in recent_user:
            context_parts.append(f"Klient wcześniej pytał: {msg['content'][:200]}")
        
        return "\n".join(context_parts)

    def _detect_intent(self, text_lower: str) -> str:
        """Wykrywa intencję użytkownika"""
        
        # Specjalne przypadki
        if any(phrase in text_lower for phrase in ['która godzina', 'aktualna godzina', 'jaka godzina']):
            return "time"
        
        if any(keyword in text_lower for keyword in ['konsultant', 'człowiek', 'agent', 'handoff']):
            return "handoff"
        
        # Motocykle
        if 'motocykl' in text_lower or 'motor' in text_lower:
            return "motorcycle"
        
        # MINI - odsyła do salonu (ROZSZERZONE o nazwy modeli MINI)
        mini_models = ['countryman', 'clubman', 'cooper', 'paceman', 'one', 'cabrio', 'john cooper']
        if 'mini' in text_lower or any(m in text_lower for m in mini_models):
            return "mini"
        
        # Akcesoria
        if any(phrase in text_lower for phrase in ['akcesoria', 'części', 'czesci', 'akcesoriów', 'części zamienne']):
            return "accessories"
        
        # Katalogi
        if any(phrase in text_lower for phrase in ['katalog', 'katalogi', 'broszura', 'broszury', 'prospekt']):
            return "catalogs"
        
        # Konfigurator
        if any(phrase in text_lower for phrase in ['konfigurator', 'skonfiguruj', 'złóż', 'konfiguracja']):
            return "configurator"
        
        # Dostępne modele
        if any(phrase in text_lower for phrase in ['dostępne modele', 'jakie modele', 'modele bmw', 'co macie', 'jakie samochody', 'jakie auta', 'nowe samochody', 'dostępne pojazdy']):
            return "available_models"
        
        # Serwis
        if any(word in text_lower for word in ['serwis', 'napraw', 'stłuczk', 'przywieź', 'naprawiacie', 'przegląd']):
            return "service"
        
        # Kontakt
        if any(word in text_lower for word in ['kontakt', 'telefon', 'email', 'adres', 'gdzie']):
            return "contact"
        
        # Sprzedaż / leasing / rabaty
        if any(word in text_lower for word in ['sprzedajecie', 'kupić', 'zakup', 'leasing', 'kredyt', 'rabat', 'promocja', 'cena']):
            return "sales"
        
        # Godziny otwarcia
        if any(phrase in text_lower for phrase in ['godziny otwarcia', 'czynny', 'czynne', 'otwarte']):
            return "salon_hours"
        
        return "general"

    def _is_offtopic(self, text_lower: str) -> bool:
        """Sprawdza czy pytanie jest off-top"""
        
        moto_keywords = ['bmw', 'samochód', 'auto', 'silnik', 'skrzynia', 'napęd', 'koła', 
                        'opony', 'hamulce', 'zawieszenie', 'serwis', 'napraw', 'salon', 'sprzedaż',
                        'motocykl', 'motor', 'części', 'akcesoria', 'katalog', 'konfigurator',
                        'sedan', 'coupe', 'kabriolet', 'hatchback', 'kombi', 'touring', 'suv', 'suva',
                        'leasing', 'kredyt', 'rabat', 'promocja', 'cena', 'model', 'modele',
                        'x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'm2', 'm3', 'm4', 'm5', 'm8',
                        'i3', 'i4', 'i5', 'i7', 'i8', 'ix', 'z4', 'seria']
        
        has_moto_context = any(kw in text_lower for kw in moto_keywords)
        
        if has_moto_context:
            return False
        
        # Off-top kategorie
        offtop_categories = {
            'humor': ['żart', 'humor', 'dowcip', 'kawał', 'śmieszny'],
            'polityka': ['polityk', 'polityka', 'rząd', 'prezydent', 'premier', 'poseł', 'wybory'],
            'sport': ['sport', 'piłka', 'mecz', 'liga', 'siatkówka', 'koszykówka'],
            'jedzenie': ['jedzenie', 'obiad', 'kolacja', 'przepis', 'pizza', 'burger'],
        }
        
        for keywords in offtop_categories.values():
            if any(kw in text_lower for kw in keywords):
                return True
        
        return False

    def _check_hallucination(self, response: str, rag_has_data: bool) -> bool:
        """Sprawdza czy odpowiedź zawiera potencjalne halucynacje.
        
        OPTYMALIZACJA: Rozszerzono sprawdzanie poza same ceny —
        teraz łapie też wymyślone dane techniczne.
        """
        response_lower = response.lower()
        
        if not rag_has_data:
            # Jeśli RAG nie miał danych, a odpowiedź zawiera konkretne informacje - to halucynacja
            suspicious_phrases = [
                'cena', 'zł', 'leasing', 'rabat', 'promocja', 'kosztuje', 'startuje od',
                'standardowa cena', 'zazwyczaj', 'przykładowo', 'wynosi', 'zapłacisz'
            ]
            if any(phrase in response_lower for phrase in suspicious_phrases):
                print(f"⚠️ WYKRYTO HALUCYNACJĘ: odpowiedź zawiera liczby/ceny bez źródła")
                return True
        
        # Ogólne sprawdzenie: podejrzane zwroty sugerujące wymyślanie
        fabrication_indicators = [
            'standardowa cena', 'zazwyczaj kosztuje', 'przykładowo',
            'z reguły', 'typowo', 'szacunkowo', 'orientacyjnie',
            'w przybliżeniu wynosi'
        ]
        if any(phrase in response_lower for phrase in fabrication_indicators):
            print(f"⚠️ WYKRYTO HALUCYNACJĘ: zwroty sugerujące wymyślanie danych")
            return True
        
        return False

    async def _get_rag_response(self, text: str, intent: str, context: str = "") -> str:
        """Pobiera odpowiedź z RAG i Cohere z zabezpieczeniem przed halucynacjami.
        
        OPTYMALIZACJA:
        - Kontekst rozmowy używa tylko wiadomości użytkownika (nie bota)
        - System prompt wymusza język polski i strukturę odpowiedzi
        - Dodano fallback z szerszym wyszukiwaniem gdy pierwsze nie da wyników
        - Lepsza integracja z detected_models z RAG
        """
        try:
            # Wzmocnij zapytanie o kontekst intencji
            enhanced_query = text
            intent_keywords = {
                "general": "BMW samochód model",
                "sales": "sprzedaż cena leasing oferta",
                "service": "serwis naprawa godziny",
                "contact": "kontakt salon telefon adres",
            }
            
            if intent in intent_keywords:
                enhanced_query = f"{text} {intent_keywords[intent]}"
            
            # Dodaj kontekst rozmowy (TYLKO wiadomości użytkownika!)
            if context:
                enhanced_query = f"Kontekst rozmowy:\n{context}\n\nAktualne pytanie: {enhanced_query}"
            
            rag_results = await self.rag_service.retrieve_with_intent_check(query=enhanced_query, top_k=5)
            rag_has_data = rag_results.get("has_data") and rag_results.get("documents")
            
            # OPTYMALIZACJA: Fallback — jeśli pierwsze wyszukiwanie nie dało wyników,
            # spróbuj z samym tekstem (bez intent keywords)
            if not rag_has_data:
                print(f"🔄 Pierwsze wyszukiwanie puste, próbuję szersze...")
                rag_results = await self.rag_service.retrieve_with_intent_check(query=text, top_k=5)
                rag_has_data = rag_results.get("has_data") and rag_results.get("documents")
            
            if rag_has_data:
                detected_models = rag_results.get("detected_models", [])
                confidence = rag_results.get("confidence", 0)
                
                context_parts = []
                for doc in rag_results.get("documents", [])[:4]:
                    content = doc.get('content', '')
                    metadata = doc.get('metadata', {})
                    source = metadata.get('title', metadata.get('filename', 'dokument'))
                    category = metadata.get('category', 'general')
                    context_parts.append(f"[{category.upper()}] Źródło: {source}\n{content}")
                
                rag_context = "\n\n---\n\n".join(context_parts)
                
                # Opis intencji klienta
                intent_desc = {
                    "general": "ogólne pytanie o BMW",
                    "sales": "pytanie o sprzedaż/cenę/leasing",
                    "service": "pytanie o serwis",
                    "contact": "pytanie o kontakt/lokalizację",
                    "salon_hours": "pytanie o godziny otwarcia",
                }.get(intent, "ogólne pytanie")
                
                models_str = ", ".join(detected_models) if detected_models else "nie wykryto"
                
                # NOWY system prompt — ustrukturyzowany, z wymuszeniem języka polskiego
                system_prompt = f"""Jesteś Leo — ekspert BMW w salonie ZK Motors (Kielce, Radom, Rzeszów).

ZASADY ODPOWIEDZI:
1. ZAWSZE odpowiadaj po POLSKU, nawet jeśli pytanie jest w innym języku
2. Odpowiadaj WYŁĄCZNIE na podstawie danych z sekcji DANE Z BAZY WIEDZY
3. Jeśli danych BRAK w bazie — powiedz szczerze i odsyłaj do salonu ZK Motors
4. NIE wymyślaj żadnych liczb (cen, mocy, momentu obrotowego, przyspieszenia)
5. NIE używaj zwrotów: "zazwyczaj", "przykładowo", "z reguły", "standardowo"
6. PRIORYTET: dane ze źródeł [MODEL_SPECS] > [LEASING] > [LINKS]

FORMAT:
- Zacznij od modelu/tematu (bez powitania)
- Podaj KONKRETNE dane z bazy (moc, moment, cena) jeśli są dostępne
- Zakończ zaproszeniem do salonu ZK Motors lub jazdą próbną
- Maksymalnie 3-5 zdań

WYKRYTE MODELE BMW: {models_str}
INTENCJA KLIENTA: {intent_desc}

INFORMACJE O SERWISIE (zawsze dostępne):
- Godziny: {SALON_HOURS}
- Serwis przyjmuje auta po stłuczkach"""
                
                user_prompt = f"""DANE Z BAZY WIEDZY (użyj TYLKO tych danych):
{rag_context}

---
PYTANIE KLIENTA: {text}

ODPOWIEDŹ PO POLSKU (na podstawie powyższych danych):"""
                
                cohere_result = await self.cohere.generate(
                    prompt=user_prompt,
                    system_prompt=system_prompt,
                    temperature=0.4,  # ZMIANA: z 0.5 na 0.4 — mniej kreatywności
                    max_tokens=500    # ZMIANA: z 400 na 500 — więcej miejsca na dane
                )
                
                if cohere_result.get("success"):
                    response = cohere_result.get("text", "")
                    # Sprawdź czy nie halucynuje
                    if self._check_hallucination(response, rag_has_data):
                        return self._fallback_response(text, intent)
                    return response
            
            # Fallback gdy RAG nic nie znalazł - odsyła do salonu
            return self._fallback_response(text, intent)
            
        except Exception as e:
            logger.error(f"RAG response error: {e}")
            return self._fallback_response(text, intent)

    async def process_message(self, text: str, session_id: str) -> tuple[str, bool]:
        """Przetwarza wiadomość"""
        
        text_lower = text.lower().strip()
        
        # Inicjalizacja stanu
        if session_id not in self.conversation_state:
            self.conversation_state[session_id] = {
                "context": [],
                "last_model": None,
                "greeting_sent": False
            }
        
        state = self.conversation_state[session_id]
        is_first_message = len(state["context"]) == 0
        
        # Pobierz kontekst rozmowy
        conversation_context = self._get_conversation_context(state, last_n=3)
        
        # Wykryj intencję
        intent = self._detect_intent(text_lower)
        
        print(f"📝 INTENT: {intent}")
        print(f"📝 IS_FIRST: {is_first_message}")
        
        # === SPECJALNE PRZYPADKI ===
        
        # Godzina
        if intent == "time":
            aktualna_godzina = datetime.now().strftime('%H:%M')
            response = f"🕐 Aktualna godzina to {aktualna_godzina}. Czy mogę pomóc w sprawie BMW lub usług ZK Motors? 😊"
            state["context"].append({"role": "user", "content": text})
            state["context"].append({"role": "assistant", "content": response})
            if is_first_message and not state["greeting_sent"]:
                state["greeting_sent"] = True
                return get_greeting() + "\n\n" + response, False
            return response, False
        
        # Motocykle
        if intent == "motorcycle":
            response = get_motorcycle_response()
            state["context"].append({"role": "user", "content": text})
            state["context"].append({"role": "assistant", "content": response})
            if is_first_message and not state["greeting_sent"]:
                state["greeting_sent"] = True
                return get_greeting() + "\n\n" + response, False
            return response, False
        
        # MINI - odsyła do salonu
        if intent == "mini":
            response = get_mini_response()
            state["context"].append({"role": "user", "content": text})
            state["context"].append({"role": "assistant", "content": response})
            if is_first_message and not state["greeting_sent"]:
                state["greeting_sent"] = True
                return get_greeting() + "\n\n" + response, False
            return response, False
        
        # Akcesoria
        if intent == "accessories":
            response = get_accessories_response()
            state["context"].append({"role": "user", "content": text})
            state["context"].append({"role": "assistant", "content": response})
            if is_first_message and not state["greeting_sent"]:
                state["greeting_sent"] = True
                return get_greeting() + "\n\n" + response, False
            return response, False
        
        # Katalogi
        if intent == "catalogs":
            response = get_catalogs_response()
            state["context"].append({"role": "user", "content": text})
            state["context"].append({"role": "assistant", "content": response})
            if is_first_message and not state["greeting_sent"]:
                state["greeting_sent"] = True
                return get_greeting() + "\n\n" + response, False
            return response, False
        
        # Konfigurator
        if intent == "configurator":
            response = get_configurator_response()
            state["context"].append({"role": "user", "content": text})
            state["context"].append({"role": "assistant", "content": response})
            if is_first_message and not state["greeting_sent"]:
                state["greeting_sent"] = True
                return get_greeting() + "\n\n" + response, False
            return response, False
        
        # Dostępne modele
        if intent == "available_models":
            response = get_available_models_response()
            state["context"].append({"role": "user", "content": text})
            state["context"].append({"role": "assistant", "content": response})
            if is_first_message and not state["greeting_sent"]:
                state["greeting_sent"] = True
                return get_greeting() + "\n\n" + response, False
            return response, False
        
        # Handoff
        if intent == "handoff":
            response = "Łączę z konsultantem..."
            state["context"].append({"role": "user", "content": text})
            state["context"].append({"role": "assistant", "content": response})
            if is_first_message and not state["greeting_sent"]:
                state["greeting_sent"] = True
                return get_greeting() + "\n\n" + response, True
            return response, True
        
        # Off-top
        if self._is_offtopic(text_lower):
            response = "😊 Jestem tu po to, żeby pomagać w sprawach BMW i ZK Motors! 🚗\n\nJeśli masz pytanie o modele, serwis, leasing lub jazdę próbną – śmiało pytaj!"
            state["context"].append({"role": "user", "content": text})
            state["context"].append({"role": "assistant", "content": response})
            if is_first_message and not state["greeting_sent"]:
                state["greeting_sent"] = True
                return get_greeting() + "\n\n" + response, False
            return response, False
        
        # === RESZTA - UŻYWA RAG ===
        
        await self._ensure_rag()
        
        if self.rag_service and RAG_AVAILABLE:
            response = await self._get_rag_response(text, intent, conversation_context)
        else:
            response = self._fallback_response(text, intent)
        
        # Usuń ewentualne powitania
        greeting_phrases = ["witaj", "cześć", "hej", "dzień dobry", "witam", "jestem leo"]
        for phrase in greeting_phrases:
            if response.lower().strip().startswith(phrase):
                lines = response.split('\n')
                if len(lines) > 1:
                    response = '\n'.join(lines[1:]).strip()
                else:
                    response = response.replace(phrase, "", 1).strip()
                break
        
        # Zapisz kontekst
        state["context"].append({"role": "user", "content": text})
        state["context"].append({"role": "assistant", "content": response})
        if len(state["context"]) > 10:
            state["context"] = state["context"][-10:]
        
        # Dodaj powitanie jeśli pierwsza wiadomość
        if is_first_message and not state["greeting_sent"]:
            state["greeting_sent"] = True
            return get_greeting() + "\n\n" + response, False
        
        return response, False

    def _fallback_response(self, text: str, intent: str) -> str:
        """Fallback gdy RAG nie działa - odsyła do salonu"""
        text_lower = text.lower()
        
        # Motocykle
        if intent == "motorcycle" or any(word in text_lower for word in ['motocykl', 'motor']):
            return get_motorcycle_response()
        
        # MINI
        if intent == "mini" or 'mini' in text_lower:
            return get_mini_response()
        
        # Akcesoria
        if intent == "accessories" or any(word in text_lower for word in ['akcesoria', 'części']):
            return get_accessories_response()
        
        # Katalogi
        if intent == "catalogs":
            return get_catalogs_response()
        
        # Konfigurator
        if intent == "configurator":
            return get_configurator_response()
        
        # Dostępne modele
        if intent == "available_models":
            return get_available_models_response()
        
        # Serwis
        if intent == "service" or any(word in text_lower for word in ['serwis', 'napraw', 'stłuczk']):
            return f"""Tak, prowadzimy serwis BMW w ZK Motors.

Godziny: {SALON_HOURS}

Serwis Kielce: ul. Wystawowa 2, tel +48 734 188 420
Serwis Radom: ul. Warszawska 234, tel +48 734 188 500
Serwis Rzeszów: ul. Krasne 9a, tel +48 734 132 120

Przyjmujemy auta po stłuczkach."""
        
        # Kontakt
        if intent == "contact":
            return f"""ZK Motors - dane kontaktowe:

Kielce: ul. Wystawowa 2, tel +48 734 188 400
Radom: ul. Warszawska 234, tel +48 734 188 500
Rzeszów: ul. Krasne 9a, tel +48 734 132 100

Godziny: {SALON_HOURS}"""
        
        # Sprzedaż / leasing - odsyła do salonu (bez wymyślania cen)
        if intent == "sales":
            return f"""Zapraszamy do kontaktu z salonami ZK Motors w sprawie oferty sprzedaży, leasingu i aktualnych promocji!

Kielce: tel +48 734 188 400
Radom: tel +48 734 188 500
Rzeszów: tel +48 734 132 100

Godziny: {SALON_HOURS}

Konfigurator BMW: https://www.bmw.pl/pl/konfigurator.html"""
        
        # Godziny salonu
        if intent == "salon_hours":
            return f"""Godziny otwarcia salonów ZK Motors:

Kielce: {SALON_HOURS}
Radom: {SALON_HOURS}
Rzeszów: {SALON_HOURS}

Zapraszamy! 🚗"""
        
        # Domyślna - odsyła do salonu
        return f"""Dziękuję za pytanie! Aby uzyskać dokładne informacje, zachęcam do kontaktu z naszym salonem:

Kielce: tel +48 734 188 400
Radom: tel +48 734 188 500
Rzeszów: tel +48 734 132 100

Godziny otwarcia: {SALON_HOURS}

Możesz też sprawdzić konfigurator BMW: https://www.bmw.pl/pl/konfigurator.html"""

# ============================================
# FASTAPI SERVER
# ============================================

app = FastAPI(title="Crisp Bot z RAG i Cohere")

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])
app.add_middleware(SessionMiddleware, secret_key=os.getenv("SECRET_KEY", secrets.token_urlsafe(32)))

bot = CrispBot()

_logs = []
_MAX_LOGS = 50

def add_log(message: str):
    global _logs
    timestamp = datetime.now().strftime('%H:%M:%S')
    _logs.append(f"[{timestamp}] {message}")
    if len(_logs) > _MAX_LOGS:
        _logs = _logs[-_MAX_LOGS:]

@app.get("/")
async def home():
    return RedirectResponse(url="/panel")

@app.get("/panel", response_class=HTMLResponse)
async def panel():
    html = """<!DOCTYPE html>
<html>
<head><title>Crisp Bot Panel</title>
<style>
    body { font-family: Arial; margin: 40px; background: #f5f5f5; }
    .container { max-width: 800px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; }
    .url { background: #f0f0f0; padding: 10px; font-family: monospace; border-radius: 5px; }
    .status { padding: 10px; border-radius: 5px; margin: 10px 0; }
    .good { background: #d4edda; }
</style>
</head>
<body>
<div class="container">
    <h1>🤖 Crisp Bot z RAG i Cohere</h1>
    <div class="status good">✅ Bot aktywny</div>
    <h3>🌍 URL webhooka:</h3>
    <div class="url" id="url"></div>
    <h3>📊 Ostatnie logi:</h3>
    <pre id="logs" style="background:#f0f0f0; padding:10px; max-height:200px; overflow:auto;">Brak</pre>
</div>
<script>
    document.getElementById('url').innerText = window.location.origin + '/crisp/webhook';
    setInterval(() => {
        fetch('/logs').then(r=>r.text()).then(logs => {
            document.getElementById('logs').innerText = logs || 'Brak logów';
        });
    }, 2000);
</script>
</body>
</html>"""
    return HTMLResponse(content=html)

# ============================================
# WEBHOOK ENDPOINT
# ============================================

@app.api_route("/crisp/webhook", methods=["POST", "GET"])
@app.api_route("/crisp", methods=["POST", "GET"])
async def crisp_webhook(request: Request):
    try:
        body = await request.body()
        headers = dict(request.headers)
        timestamp = int(datetime.now().timestamp() * 1000)

        if request.method == "GET":
            return JSONResponse({"status": "webhook endpoint active"}, status_code=200)

        if not body:
            return JSONResponse({"status": "no body"}, status_code=200)

        # Obsługa różnych kodowań
        try:
            body_str = body.decode('utf-8')
        except UnicodeDecodeError:
            try:
                body_str = body.decode('latin-1')
            except UnicodeDecodeError:
                body_str = body.decode('utf-8', errors='ignore')
        
        data = json.loads(body_str)

        if CRISP_WEBHOOK_SECRET:
            sig = headers.get("x-crisp-signature", headers.get("X-Crisp-Signature"))
            ts = headers.get("x-crisp-request-timestamp", headers.get("X-Crisp-Request-Timestamp"))
            if sig and ts:
                if not verify_crisp_signature(body, sig, ts, CRISP_WEBHOOK_SECRET):
                    add_log("⚠️ Nieprawidłowa sygnatura - kontynuuję")

        event = data.get('event', 'unknown')
        
        if event != 'message:send':
            return JSONResponse({"status": "ignored"}, status_code=200)

        msg_data = data.get('data', {})
        if msg_data.get('from') != 'user':
            return JSONResponse({"status": "ignored"}, status_code=200)

        website_id = data.get('website_id')
        session_id = msg_data.get('session_id')
        message = msg_data.get('content', '')
        
        if is_duplicate(session_id, message, timestamp):
            add_log(f"⏭️ Duplikat: {message[:30]}...")
            return JSONResponse({"status": "duplicate ignored"}, status_code=200)

        add_log(f"Wiadomość: {message[:50]}...")
        print(f"💬 {message[:200]}")

        response, transfer = await bot.process_message(message, session_id)

        if response:
            print(f"📤 ODPOWIEDŹ: {response[:150]}...")
            await send_crisp_message(website_id, session_id, response)
            add_log("✅ Odpowiedź wysłana")

        if transfer:
            await send_crisp_message(website_id, session_id, "🔄 Łączę z konsultantem...")

        return JSONResponse({"status": "ok"}, status_code=200)

    except Exception as e:
        add_log(f"❌ BŁĄD: {str(e)[:50]}")
        traceback.print_exc()
        return JSONResponse({"error": str(e)}, status_code=500)

# ============================================
# ENDPOINTY DIAGNOSTYCZNE
# ============================================

@app.get("/logs")
async def get_logs():
    return "\n".join(_logs)

@app.get("/rag/info")
async def rag_info():
    try:
        rag = await get_rag_service()
        health = await rag.health_check()
        stats = await rag.get_stats()
        return {
            "healthy": health.get("status") == "healthy",
            "documents": stats.get("documents_in_store", 0),
            "available": RAG_AVAILABLE
        }
    except Exception as e:
        return {"healthy": False, "available": RAG_AVAILABLE, "error": str(e)}

@app.get("/cohere/info")
async def cohere_info():
    return {
        "available": bool(os.getenv("COHERE_API_KEY")),
        "model": os.getenv("COHERE_MODEL", "command-a-03-2025")
    }

@app.get("/health")
async def health():
    return {"status": "ok", "time": datetime.now().isoformat()}

# ============================================
# ENDPOINTY TESTOWE
# ============================================

@app.get("/test/rag")
async def test_rag():
    try:
        response, transfer = await bot.process_message("gdzie znajdę katalogi modeli", "test_session")
        return {"success": True, "response": response, "transfer": transfer}
    except Exception as e:
        return {"success": False, "error": str(e)}

@app.get("/test/query")
async def test_query(q: str = "gdzie znajdę katalogi modeli"):
    try:
        response, transfer = await bot.process_message(q, "test_session")
        return {"query": q, "response": response, "transfer": transfer}
    except Exception as e:
        return {"query": q, "success": False, "error": str(e)}

@app.on_event("startup")
async def startup():
    print("\n" + "="*60)
    print("🚀 CRISP BOT Z RAG I COHERE")
    print("="*60)
    print(f"📊 Panel: http://localhost:8000/panel")
    print(f"🌍 Webhook: {BASE_URL}/crisp/webhook")
    print("="*60)
    print("\n📝 ENDPOINTY TESTOWE:")
    print(f"   http://localhost:8000/test/rag")
    print(f"   http://localhost:8000/test/query?q=twoje pytanie")
    print("="*60)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)