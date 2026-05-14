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

# Async lock per session — zapobiega podwójnym odpowiedziom (Issue #1)
_session_locks: Dict[str, asyncio.Lock] = {}

# Konkurencyjne marki — bot NIGDY nie powinien o nich mówić (Issue #4)
COMPETITOR_BRANDS = [
    'mercedes', 'audi', 'alfa romeo', 'toyota', 'honda', 'volkswagen',
    'porsche', 'lexus', 'volvo', 'tesla', 'hyundai', 'kia', 'ford',
    'opel', 'renault', 'peugeot', 'citroen', 'skoda', 'seat', 'fiat',
    'jaguar', 'land rover', 'mazda', 'subaru', 'nissan', 'chevrolet',
    'jeep', 'dodge', 'mitsubishi', 'suzuki', 'dacia', 'cupra',
]

def is_duplicate(session_id: str, message: str, timestamp: int) -> bool:
    """Sprawdza czy wiadomość była już przetwarzana"""
    key = f"{session_id}:{message}"
    if key in _last_processed:
        last_time = _last_processed[key]
        if timestamp - last_time < 30000:  # ZMIANA: z 5000ms na 30000ms
            logger.debug(f"[{session_id[:8]}] DUPLIKAT wykryty (delta={timestamp - last_time}ms)")
            return True
    _last_processed[key] = timestamp
    if len(_last_processed) > 100:
        for k in list(_last_processed.keys()):
            if _last_processed[k] < timestamp - 30000:
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

def get_trade_in_response() -> str:
    """Odpowiedź na pytania o trade-in / odkup / wymianę samochodu"""
    return """🔄 **Wymiana samochodu na BMW (Trade-in / Odkup):**

Możesz oddać swój obecny samochód (dowolnej marki!) w rozliczeniu przy zakupie nowego BMW.

**Korzyści:**
- Konkurencyjna wycena pojazdu
- Uproszczone formalności i dokumentacja
- Bezpłatne oględziny pojazdu w salonie
- Wartość auta odliczana od ceny nowego BMW

**Wymagane dokumenty:** dowód rejestracyjny, dokumentacja serwisowa (pomocna)

**Wycena online:**
https://www.bmw.pl/pl/odkup/

Zapraszamy też do salonu ZK Motors po dokładną wycenę:
- Kielce: tel +48 734 188 400
- Radom: tel +48 734 188 500
- Rzeszów: tel +48 734 132 100"""

def get_test_drive_response() -> str:
    """Lista modeli dostępnych do jazd próbnych — bezpośrednio z dostepne_samochody_probne.txt."""
    return """🚗 **Modele dostępne do jazd próbnych w ZK Motors:**

- BMW Serii 1
- BMW M235
- BMW Serii 7
- BMW iX2
- BMW iX3
- BMW X6
- BMW M5

Dostępność może się różnić w zależności od dnia i obłożenia — zalecana wcześniejsza rezerwacja.

Jazdę próbną można umówić u doradcy handlowego lub telefonicznie. Zapraszamy do salonu ZK Motors!"""


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

    def _clean_response(self, response: str) -> str:
        """Post-processing odpowiedzi — usuwa CJK chars, naprawia formatowanie.

        Issue #2/#5: Cohere command-r7b czasem wstawia chińskie/japońskie/koreańskie znaki.
        Rozszerzony zakres Unicode obejmuje wszystkie bloki CJK + fullwidth forms.
        """
        # Usuń WSZYSTKIE znaki CJK — pełny zakres Unicode bloków
        response = re.sub(
            r'['
            r'\u2e80-\u2eff'   # CJK Radicals Supplement
            r'\u3000-\u303f'   # CJK Symbols and Punctuation
            r'\u3040-\u309f'   # Hiragana
            r'\u30a0-\u30ff'   # Katakana
            r'\u3100-\u312f'   # Bopomofo
            r'\u3130-\u318f'   # Hangul Compatibility Jamo
            r'\u31a0-\u31bf'   # Bopomofo Extended
            r'\u31f0-\u31ff'   # Katakana Phonetic Extensions
            r'\u3200-\u32ff'   # Enclosed CJK Letters
            r'\u3400-\u4dbf'   # CJK Unified Ideographs Extension A
            r'\u4e00-\u9fff'   # CJK Unified Ideographs (main block)
            r'\ua960-\ua97f'   # Hangul Jamo Extended-A
            r'\uac00-\ud7af'   # Hangul Syllables
            r'\uf900-\ufaff'   # CJK Compatibility Ideographs
            r'\ufe30-\ufe4f'   # CJK Compatibility Forms
            r'\uff00-\uffef'   # Halfwidth and Fullwidth Forms
            r']', '', response
        )
        # Usuń podwójne spacje po usunięciu znaków
        response = re.sub(r'  +', ' ', response)
        # Usuń puste linie nadmiarowe
        response = re.sub(r'\n{3,}', '\n\n', response)
        return response.strip()

    def _mentions_competitor(self, text_lower: str) -> bool:
        """Sprawdza czy wiadomość wspomina konkurencyjne marki.
        
        Issue #4: Bot NIGDY nie powinien dyskutować o innych markach.
        """
        return any(brand in text_lower for brand in COMPETITOR_BRANDS)

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

        # Motocykle / skutery
        if any(kw in text_lower for kw in ['motocykl', 'motor', 'skuter', 'scooter', 'motorrad']):
            return "motorcycle"

        # MINI
        mini_models = ['countryman', 'clubman', 'cooper', 'paceman', 'john cooper']
        if 'mini' in text_lower or any(m in text_lower for m in mini_models):
            return "mini"

        # Akcesoria
        if any(phrase in text_lower for phrase in [
            'akcesoria', 'części', 'czesci', 'akcesoriów', 'części zamienne',
            'accessories', 'parts', 'spare parts',
        ]):
            return "accessories"

        # Katalogi
        if any(phrase in text_lower for phrase in [
            'katalog', 'katalogi', 'broszura', 'broszury', 'prospekt', 'catalog', 'brochure',
        ]):
            return "catalogs"

        # Konfigurator — stems "konfigur" / "skonfigur" cover all Polish inflections
        # (konfigurator, konfiguracja, konfigurować, skonfiguruj, skonfigurować, …)
        if any(phrase in text_lower for phrase in [
            'konfigur', 'skonfigur', 'złóż', 'configurator', 'configure',
        ]):
            return "configurator"

        # Jazda próbna — PRZED available_models (Bug #2)
        # Regex absorbs Polish inflection: jazda/jazdy/jazd + próbn*/probn*/testow*
        if re.search(r'\bjazd\w*\s+(próbn|probn|testow)', text_lower) or \
           any(phrase in text_lower for phrase in ['test drive', 'test-drive', 'testdrive']):
            return "test_drive"

        # Dostępne modele / stok / co jest w ofercie
        # Skip when the query is asking something more specific than "what do you have?":
        #   - names a specific BMW model code (M3 Touring, iX3, …) → wants model info
        #   - uses a superlative (najtańszy, najdroższy, …) → wants RAG comparison
        # In both cases the URL dump is the wrong answer.
        has_specific_model = bool(re.search(
            r'\b(x[1-7]|xm|m[2-8]|m235|m240|m340|m440|m550|m760|z4|'
            r'i[3-8]|ix[1-3]?|seria\s*[1-8])\b',
            text_lower
        ))
        has_superlative = bool(re.search(
            r'najta[nń]sz|najdro[zż]sz|najszybsz|najmocniejsz|najlepsz|największ|najmniejsz',
            text_lower
        ))
        has_budget = bool(re.search(r'\d{2,3}\s*0{3}|tys|zł|pln|budżet|budzet', text_lower))
        if not has_budget and not has_specific_model and not has_superlative and any(phrase in text_lower for phrase in [
            'dostępne modele', 'jakie modele', 'modele bmw', 'co macie',
            'jakie macie modele', 'jakie macie samochody', 'jakie macie auta',
            'jakie samochody', 'jakie auta', 'nowe samochody', 'nowe bmw', 'dostępne pojazdy',
            'samochody na sprzedaż', 'auta na sprzedaż', 'auta do sprzedania',
            'samochody do sprzedania', 'link do samochodów', 'link do aut',
            'zobaczyć samochody', 'zobaczyć auta', 'zobaczyć ofertę',
            'oferta samochodów', 'oferta aut', 'stok', 'stock', 'magazyn',
            'od ręki', 'od reki', 'na stanie', 'w sprzedaży', 'w ofercie',
            'dostępne auta', 'dostępne samochody', 'co jest dostępne',
            'pokaż samochody', 'pokaż auta', 'pokaż modele', 'pokaż bmw', 'pokaż ofertę',
            'pokażcie samochody', 'pokażcie auta', 'pokażcie modele', 'pokażcie bmw', 'pokażcie ofertę',
            'available cars', 'cars for sale', 'see all cars', 'what cars',
            'show me your cars', 'show all cars', 'show all models',
        ]):
            return "available_models"

        # Trade-in / odkup / wymiana samochodu (Bug #D — English keywords added)
        if any(phrase in text_lower for phrase in [
            'trade-in', 'trade in', 'tradein', 'odkup', 'wymiana samochodu',
            'oddać samochód', 'oddac samochod',
            'oddać auto', 'oddac auto',
            'oddać pojazd', 'oddac pojazd',
            'rozliczenie',
            'wycena samochodu', 'wycena auta', 'sprzedać swój',
            'sprzedac swoj', 'zostawić auto', 'zostawic auto',
            'wymienić auto', 'wymienic auto', 'zamienić auto',
            'zamienic auto', 'w rozliczeniu',
            'trade', 'exchange my car', 'sell my car', 'swap my car',
        ]):
            return "trade_in"

        # Serwis (Bug #5/#C — dodano olej/filtr/English)
        if any(word in text_lower for word in [
            'serwis', 'napraw', 'stłuczk', 'przywieź', 'naprawiacie', 'przegląd',
            'olej', 'filtr', 'opony', 'hamulce', 'klimatyz', 'płyn',
            'wymiana oleju', 'wymiana filtr',
            'service', 'repair', 'maintenance', 'oil change', 'filter', 'tyres', 'brakes',
        ]):
            return "service"

        # Kontakt (Bug #D — English keywords added)
        if any(word in text_lower for word in [
            'kontakt', 'telefon', 'email', 'adres', 'gdzie',
            'contact', 'phone', 'address', 'location', 'where', 'how to reach',
        ]):
            return "contact"

        # Sprzedaż / leasing / rabaty
        if any(word in text_lower for word in [
            'sprzedajecie', 'kupić', 'zakup', 'leasing', 'kredyt', 'rabat', 'promocja', 'cena',
            'buy', 'purchase', 'price', 'financing', 'discount',
        ]):
            return "sales"

        # Godziny otwarcia
        if any(phrase in text_lower for phrase in [
            'godziny otwarcia', 'czynny', 'czynne', 'otwarte',
            'opening hours', 'open', 'hours',
        ]):
            return "salon_hours"

        return "general"

    def _is_offtopic(self, text_lower: str) -> bool:
        """Sprawdza czy pytanie jest off-topic.
        
        Issue #5: ODWRÓCONA LOGIKA — zamiast sprawdzać 'czy to off-topic?',
        sprawdzamy 'czy to jest związane z BMW/motoryzacją/salonem?'.
        Jeśli NIE jest — to jest off-topic.
        """
        # Krok 1: Sprawdź czy zawiera słowa kluczowe BMW/motoryzacja
        moto_keywords = [
            'bmw', 'samochód', 'auto', 'silnik', 'skrzynia', 'napęd', 'koła', 
            'opony', 'hamulce', 'zawieszenie', 'serwis', 'napraw', 'salon', 'sprzedaż',
            'motocykl', 'motor', 'części', 'akcesoria', 'katalog', 'konfigurator',
            'sedan', 'coupe', 'kabriolet', 'hatchback', 'kombi', 'touring', 'suv',
            'leasing', 'kredyt', 'rabat', 'promocja', 'cena', 'model', 'modele',
            'x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'm2', 'm3', 'm4', 'm5', 'm8',
            'i3', 'i4', 'i5', 'i7', 'i8', 'ix', 'z4', 'seria', 'xm',
            'jazda próbna', 'test drive', 'godziny', 'otwar',
            'kontakt', 'telefon', 'adres', 'gdzie',
            'kupic', 'kupić', 'zakup', 'oferta', 'ofert',
            'car', 'vehicle', 'engine', 'drive', 'price', 'buy',
            'stłuczk', 'wypadek', 'collision', 'repair',
            'trade', 'wycen', 'premium selection',
            'elektrycz', 'hybryd', 'ev', 'bateri', 'zasięg', 'zasieg', 'ładowan',
            # Service vocab missing from original list
            'opon', 'klocki', 'klocek', 'hamulc', 'klimatyz', 'olej', 'filtr', 'filtry', 'akumulator',
            # Contact terms
            'email', 'mail', 'where', 'dealer', 'showroom',
            # Vague-but-on-topic — customer describing what kind of car they want
            'sport', 'sportow', 'rodzin', 'rodzinn', 'luksus', 'ekonomi',
            'oszczęd', 'oszczed', 'szybki', 'wygodn', 'komfort', 'miejsk',
            'polecasz', 'polećcie', 'doradz', 'pomóż wybrać', 'pomoz wybrac',
        ]
        
        has_moto_context = any(kw in text_lower for kw in moto_keywords)
        
        if has_moto_context:
            return False
        
        # Krok 2: Sprawdź czy to powitanie (nie off-topic)
        greetings = ['cześć', 'hej', 'witam', 'dzień dobry', 'siema', 'hello', 'hi', 'hey', 'halo']
        if text_lower.strip() in greetings:
            return False
        
        # Krok 3: Jeśli jest krótkie i nie zawiera BMW keywordów — off-topic
        # Np: "git pull origin main", "Paweł Piwpwarczyk", "Roman Banasik"
        if len(text_lower.split()) <= 5 and not has_moto_context:
            logger.info(f"OFF-TOPIC: krótka wiadomość bez moto kontekstu: '{text_lower[:50]}'")
            return True
        
        # Krok 4: Sprawdź znane kategorie off-topic
        offtop_categories = {
            'humor': ['żart', 'humor', 'dowcip', 'kawał', 'śmieszny'],
            'polityka': ['polityk', 'polityka', 'rząd', 'prezydent', 'premier', 'poseł', 'wybory'],
            'sport': ['sport', 'piłka', 'mecz', 'liga', 'siatkówka', 'koszykówka'],
            'jedzenie': ['jedzenie', 'obiad', 'kolacja', 'przepis', 'pizza', 'burger'],
            'tech': ['git ', 'npm ', 'code', 'python', 'javascript', 'docker'],
            'personal': ['właściciel', 'owner', 'kto jest', 'who is'],
        }
        
        for cat, keywords in offtop_categories.items():
            if any(kw in text_lower for kw in keywords):
                logger.info(f"OFF-TOPIC: kategoria '{cat}' wykryta w: '{text_lower[:50]}'")
                return True
        
        return False

    def _check_hallucination(self, response: str, rag_has_data: bool, user_message: str = "") -> bool:
        """Sprawdza czy odpowiedź zawiera potencjalne halucynacje.

        OPTYMALIZACJA: Rozszerzono sprawdzanie poza same ceny —
        teraz łapie też wymyślone dane techniczne i naruszenia budżetu.
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

        # Sprawdzenie naruszenia budżetu: jeśli użytkownik podał budżet,
        # a odpowiedź zawiera ceny wyższe — to halucynacja/zła odpowiedź
        if user_message:
            budget_match = re.search(r'do\s+([\d\s.,]+)\s*(zł|pln|złotych|tys)', user_message.lower())
            if budget_match:
                try:
                    budget_str = budget_match.group(1).replace(' ', '').replace('.', '').replace(',', '')
                    budget = int(budget_str)
                    # Szukaj cen w odpowiedzi
                    price_matches = re.findall(r'([\d\s.,]+)\s*(zł|pln|złotych)', response_lower)
                    for price_str, _ in price_matches:
                        try:
                            price = int(price_str.replace(' ', '').replace('.', '').replace(',', ''))
                            if price > budget * 1.1:  # 10% tolerancja
                                print(f"⚠️ WYKRYTO NARUSZENIE BUDŻETU: cena {price} > budżet {budget}")
                                return True
                        except (ValueError, TypeError):
                            continue
                except (ValueError, TypeError):
                    pass

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
                "contact": "kontakt salon telefon adres doradca",
                "test_drive": "jazda próbna dostępne modele demo pojazdy",
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
                logger.info(f"RAG: pierwsze wyszukiwanie puste, próbuję szersze...")
                rag_results = await self.rag_service.retrieve_with_intent_check(query=text, top_k=5)
                rag_has_data = rag_results.get("has_data") and rag_results.get("documents")
            
            if rag_has_data:
                detected_models = rag_results.get("detected_models", [])
                confidence = rag_results.get("confidence", 0)
                n_docs = len(rag_results.get("documents", []))
                logger.info(f"RAG: has_data={rag_has_data} | confidence={confidence:.3f} | docs={n_docs} | models={detected_models}")
                
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
                    "contact": "pytanie o kontakt/lokalizację/doradcę",
                    "salon_hours": "pytanie o godziny otwarcia",
                    "test_drive": "pytanie o jazdę próbną / dostępne modele demo",
                    "trade_in": "pytanie o wymianę/odkup samochodu",
                }.get(intent, "ogólne pytanie")
                
                models_str = ", ".join(detected_models) if detected_models else "nie wykryto"
                
                # System prompt z blokadą konkurencji i kontaktami
                system_prompt = f"""Jesteś Leo — ekspert BMW w salonie ZK Motors (Kielce, Radom, Rzeszów).

ZASADY ODPOWIEDZI:
1. ZAWSZE odpowiadaj po POLSKU, nawet jeśli pytanie jest w innym języku
2. Odpowiadaj WYŁĄCZNIE na podstawie danych z sekcji DANE Z BAZY WIEDZY
3. Jeśli danych BRAK w bazie — powiedz szczerze i odsyłaj do salonu ZK Motors
4. NIE wymyślaj żadnych liczb (cen, mocy, momentu obrotowego, przyspieszenia)
5. NIE używaj zwrotów: "zazwyczaj", "przykładowo", "z reguły", "standardowo"
6. PRIORYTET danych: [CONTACT] > [MODEL_SPECS] > [SERVICE] > [LEASING] > [LINKS]
7. NIGDY nie wspominaj o konkurencyjnych markach (Mercedes, Audi, Toyota itp.)
8. Jeśli klient pyta o porównanie z inną marką — grzecznie odmów i skup się na zaletach BMW
9. NIE odpowiadaj na pytania niezwiązane z BMW/salonem (np. kto jest właścicielem, osobiste pytania)
10. BUDŻET KLIENTA: Jeśli klient podaje budżet, sprawdź CENY w danych z bazy. Jeśli żaden model nie mieści się w budżecie — powiedz to wprost i zaproponuj: (a) używane BMW Premium Selection, (b) kontakt z salonem ZK Motors. NIGDY nie proponuj modelu przekraczającego budżet.
11. LINKI DO STOKU: Podawaj link do stoku TYLKO gdy klient pyta ogólnie o dostępne/nowe samochody — NIE przy pytaniach o jazdę próbną, kontakty czy serwis.
12. NIGDY nie używaj nawiasów kwadratowych ani placeholderów jak [dane z bazy danych], [data from database], [HP], [KM], [s] itp. Jeśli konkretnych danych (mocy, przyspieszenia, ceny) nie ma w DANE Z BAZY WIEDZY — napisz wprost: "Nie posiadam szczegółowych danych technicznych dla tego modelu" i zaproś do salonu.

FORMAT:
- Zacznij od modelu/tematu (bez powitania)
- Podaj KONKRETNE dane z bazy (moc, moment, cena) jeśli są dostępne
- Zakończ zaproszeniem do salonu ZK Motors lub jazdą próbną
- Maksymalnie 3-5 zdań
- Wysyłaj JEDNĄ spójną odpowiedź (nie mieszaj wielu tematów)
- NIE cytuj i NIE powtarzaj pytania użytkownika na początku odpowiedzi.
- Unikaj pisania WIELKIMI LITERAMI (caps lock) słów potwierdzających/zaprzeczających (np. TAK, NIE).

WYKRYTE MODELE BMW: {models_str}
INTENCJA KLIENTA: {intent_desc}

KONTAKT OGÓLNY (używaj TYLKO gdy baza wiedzy nie zawiera bardziej szczegółowych danych kontaktowych — np. imiennych doradców):
- Kielce: ul. Wystawowa 2, tel +48 734 188 400 (serwis: +48 734 188 420)
- Radom: ul. Warszawska 234, tel +48 734 188 500
- Rzeszów: ul. Krasne 9a, tel +48 734 132 100 (serwis: +48 734 132 120)

INFORMACJE O SERWISIE (zawsze dostępne):
- Godziny: {SALON_HOURS}
- Serwis wykonuje pełen zakres usług BMW: olej, filtry, przeglądy, opony, hamulce, diagnostyka i inne
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
                    if self._check_hallucination(response, rag_has_data, text):
                        # Jeśli to kwestia budżetu — daj specjalną odpowiedź
                        budget_match = re.search(r'do\s+([\d\s.,]+)\s*(zł|pln|złotych|tys)', text.lower())
                        if budget_match:
                            return self._budget_fallback_response(text)
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

        # === POWITANIA — krótki obieg przed RAG (Bug #4) ===
        # Czyste powitania nigdy nie trafiają do RAG — wracają TYLKO przywitaniem.
        GREETINGS_EXACT = {
            'cześć', 'czesc', 'hej', 'witam', 'dzień dobry', 'dzien dobry',
            'siema', 'hello', 'hi', 'hey', 'halo', 'good morning',
            'dobry wieczór', 'dobry wieczor', 'dobranoc',
            'cześć!', 'hej!', 'witam!',
        }
        # Short 1-2 word queries with any of these are NOT greetings — they're real questions.
        # Includes BMW model codes (so "ix3 zasieg" doesn't get greeted) and common
        # one-word information-seeking words.
        BMW_CONTEXT_WORDS = {
            'model', 'bmw', 'auto', 'serwis', 'motocykl', 'test', 'jazda', 'cena', 'kontakt',
            # Model codes
            'x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'xm', 'z4',
            'i3', 'i4', 'i5', 'i7', 'i8', 'ix', 'ix1', 'ix2', 'ix3',
            'm2', 'm3', 'm4', 'm5', 'm8', 'm235', 'm240', 'm340', 'm440', 'm550', 'm760',
            'seria',
            # Common short info-seeking words
            'zasięg', 'zasieg', 'moc', 'ile', 'cena', 'info', 'spec',
            'olej', 'filtr', 'opony', 'hamulce',
        }
        is_pure_greeting = (
            text_lower.strip() in GREETINGS_EXACT
            or (len(text_lower.split()) <= 2 and not any(kw in text_lower for kw in BMW_CONTEXT_WORDS))
        )
        if is_pure_greeting:
            greeting_response = get_greeting()
            state["greeting_sent"] = True
            state["context"].append({"role": "user", "content": text})
            state["context"].append({"role": "assistant", "content": greeting_response})
            return greeting_response, False

        # Pobierz kontekst rozmowy
        conversation_context = self._get_conversation_context(state, last_n=3)

        # Wykryj intencję
        intent = self._detect_intent(text_lower)
        
        logger.info(f"[{session_id[:8]}] INTENT={intent} | is_first={is_first_message} | msg='{text[:50]}'")
        
        # === KONKURENCJA — Issue #4 ===
        # NIGDY nie dyskutuj o innych markach.
        # WYJĄTEK: trade-in — klient wymienia markę SWOJEGO obecnego auta,
        # nie porównuje marek; źródło trade_in_BMW.txt mówi "DOWOLNEJ MARKI".
        if intent != "trade_in" and self._mentions_competitor(text_lower):
            logger.info(f"[{session_id[:8]}] COMPETITOR wykryty — blokuję")
            response = """Specjalizuję się wyłącznie w ofercie BMW i ZK Motors. 😊

Nie mogę porównywać z innymi markami, ale chętnie opowiem Ci o zaletach naszych modeli BMW!

Który model BMW Cię interesuje? Mamy w ofercie SUV-y (X1-X7), sedany (seria 3, 5, 7), sportowe (M3, M4, M5) i elektryczne (i4, i5, i7, iX). 🚗"""
            state["context"].append({"role": "user", "content": text})
            state["context"].append({"role": "assistant", "content": response})
            if is_first_message and not state["greeting_sent"]:
                state["greeting_sent"] = True
                return get_greeting() + "\n\n" + response, False
            return response, False
        
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
        
        # Motocykle / skutery
        if intent == "motorcycle":
            # Jeśli klient pyta o KONTAKT / DORADCĘ — idź do RAG (Bug #1)
            # kontakt_do_doradcow.txt zawiera Martę Magielską i Adama Obarę z bezpośrednimi danymi
            moto_contact_kws = [
                'kontakt', 'doradca', 'doradcy', 'telefon', 'numer',
                'email', 'kto', 'dane kontaktowe', 'contact', 'advisor',
                'sprzedawca', 'pracownik', 'osoba',
            ]
            if any(kw in text_lower for kw in moto_contact_kws):
                pass  # fall through do RAG poniżej
            elif any(kw in text_lower for kw in ['skuter', 'scooter']):
                response = """W ofercie ZK Motors nie ma skuterów. Salon oferuje nowe i używane motocykle BMW.

Jeśli interesują Cię motocykle BMW, sprawdź naszą ofertę:

🆕 Nowe motocykle BMW:
https://stok.zkmotors.pl/pojazd/lista?PojazdSearch%5Brodzaj_id%5D=2&PojazdSearch%5Bstatus_id%5D=1

🔄 Używane motocykle BMW:
https://stok.zkmotors.pl/pojazd/lista?PojazdSearch%5Bstatus_id%5D=3

Zapraszam do kontaktu z salonem ZK Motors po więcej informacji!"""
                state["context"].append({"role": "user", "content": text})
                state["context"].append({"role": "assistant", "content": response})
                if is_first_message and not state["greeting_sent"]:
                    state["greeting_sent"] = True
                    return get_greeting() + "\n\n" + response, False
                return response, False
            else:
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

        # Jazda próbna — lista pochodzi bezpośrednio z dostepne_samochody_probne.txt;
        # RAG mis-ranguje ten plik vs linki_dostepne_pojazdy_stok.txt, więc obsługujemy
        # podstawowe pytanie statycznie. Bardziej szczegółowe pytania nadal idą do RAG.
        if intent == "test_drive":
            response = get_test_drive_response()
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
        
        # Off-topic (Issue #5 — wzmocniona detekcja)
        # Bug fix: trust _detect_intent. Only fall back to off-topic heuristic
        # when intent was "general" — otherwise short on-topic service questions
        # like "wymień olej i filtry" (5 words, no moto keyword) get misrouted.
        if intent == "general" and self._is_offtopic(text_lower):
            logger.info(f"[{session_id[:8]}] OFF-TOPIC wykryty")
            response = "😊 Jestem tu po to, żeby pomagać w sprawach BMW i ZK Motors! 🚗\n\nMogę pomóc z:\n- Informacjami o modelach BMW\n- Serwisem i naprawami\n- Ofertami i finansowaniem\n- Jazdą próbną\n\nW czym mogę pomóc? 😊"
            state["context"].append({"role": "user", "content": text})
            state["context"].append({"role": "assistant", "content": response})
            if is_first_message and not state["greeting_sent"]:
                state["greeting_sent"] = True
                return get_greeting() + "\n\n" + response, False
            return response, False
        
        # === RESZTA — UŻYWA RAG ===
        
        await self._ensure_rag()
        
        if self.rag_service and RAG_AVAILABLE:
            response = await self._get_rag_response(text, intent, conversation_context)
        else:
            response = self._fallback_response(text, intent)
        
        # === POST-PROCESSING ===
        
        # Issue #2: Usuń CJK znaki (chińskie/japońskie)
        response = self._clean_response(response)
        
        # Issue #4: Jeśli odpowiedź LLM mimo wszystko wspomina konkurencję — wyczyść
        response_lower_check = response.lower()
        for brand in COMPETITOR_BRANDS:
            if brand in response_lower_check:
                logger.warning(f"[{session_id[:8]}] LLM wspomniał konkurenta '{brand}' — blokuję odpowiedź")
                response = self._fallback_response(text, intent)
                break
        
        # Issue #6: Dodaj kontakty do serwisowych odpowiedzi
        if intent == "service" and "tel" not in response.lower():
            response += f"\n\nKontakt do serwisów ZK Motors:\n- Kielce: ul. Wystawowa 2, tel +48 734 188 420\n- Radom: ul. Warszawska 234, tel +48 734 188 500\n- Rzeszów: ul. Krasne 9a, tel +48 734 132 120\n\nGodziny: {SALON_HOURS}"

        # Bug #6: Jeśli odpowiedź mówi o dostępnych/aktualnych samochodach ale nie ma linku do stoku
        stock_keywords = ['dostępn', 'od ręki', 'od reki', 'na stanie', 'w ofercie', 'aktualn', 'w salonie możesz zobaczyć', 'sprawdź']
        has_stock_link = 'stok.zkmotors.pl' in response or 'najlepszeoferty.bmw.pl' in response
        if any(kw in response.lower() for kw in stock_keywords) and not has_stock_link and intent != "test_drive":
            if any(kw in text_lower for kw in ['dostępn', 'od ręki', 'od reki', 'na stanie', 'w ofercie', 'co macie', 'jaki', 'zobaczyć', 'where', 'link']):
                response += "\n\nSprawdź aktualny stok: https://stok.zkmotors.pl/pojazd/lista?PojazdSearch%5Bmarka_id%5D=1&PojazdSearch%5Brodzaj_id%5D=1&PojazdSearch%5Bstatus_id%5D=1"

        # Trade-in post-processing (Bug #2)
        if intent == "trade_in" and "bmw.pl/pl/odkup" not in response.lower():
            response += "\n\nWycena online: https://www.bmw.pl/pl/odkup/"
        
        # Usuń ewentualne powitania (LLM czasem dodaje "Witaj" mimo instrukcji)
        greeting_phrases = ["witaj", "cześć", "hej", "dzień dobry", "witam", "jestem leo"]
        for phrase in greeting_phrases:
            if response.lower().strip().startswith(phrase):
                lines = response.split('\n')
                if len(lines) > 1:
                    response = '\n'.join(lines[1:]).strip()
                else:
                    response = response.replace(phrase, "", 1).strip()
                break
        
        logger.info(f"[{session_id[:8]}] RESPONSE: len={len(response)} | intent={intent}")
        
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

    def _budget_fallback_response(self, text: str) -> str:
        """Fallback gdy klient podaje budżet, ale żaden nowy model się nie mieści."""
        return f"""W aktualnej ofercie nowych samochodów BMW może nie być modelu w podanym budżecie.

Proponuję rozważyć:

🔄 **Używane BMW (Premium Selection)** — certyfikowane auta z gwarancją:
https://najlepszeoferty.bmw.pl/uzywane/

🚗 **Aktualny stok nowych samochodów** — ceny mogą się różnić w zależności od promocji:
https://stok.zkmotors.pl/pojazd/lista?PojazdSearch%5Bmarka_id%5D=1&PojazdSearch%5Brodzaj_id%5D=1&PojazdSearch%5Bstatus_id%5D=1

Zapraszam też do salonu ZK Motors — nasi doradcy pomogą znaleźć najlepszą opcję w Twoim budżecie!
- Kielce: tel +48 734 188 400
- Radom: tel +48 734 188 500
- Rzeszów: tel +48 734 132 100"""

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

        # Trade-in / odkup
        if intent == "trade_in" or any(word in text_lower for word in ['trade-in', 'odkup', 'wymiana', 'rozliczeni']):
            return get_trade_in_response()

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
        
        logger.info(f"[{session_id[:8]}] WEBHOOK | msg='{message[:60]}' | event={event}")
        
        if is_duplicate(session_id, message, timestamp):
            add_log(f"⏭️ Duplikat: {message[:30]}...")
            logger.info(f"[{session_id[:8]}] DUPLIKAT — pominięto")
            return JSONResponse({"status": "duplicate ignored"}, status_code=200)

        add_log(f"Wiadomość: {message[:50]}...")

        # ASYNC LOCK — zapobiega podwójnym odpowiedziom (Issue #1)
        # Jeśli ten sam session_id wysyła 2 wiadomości jednocześnie,
        # druga czeka aż pierwsza się skończy
        lock = _session_locks.setdefault(session_id, asyncio.Lock())
        async with lock:
            logger.info(f"[{session_id[:8]}] LOCK acquired — przetwarzam...")
            response, transfer = await bot.process_message(message, session_id)

            if response:
                # Podziel długie odpowiedzi na części (max 1500 znaków)
                # Crisp może źle obsługiwać bardzo długie wiadomości
                if len(response) > 1500:
                    parts = []
                    current = ""
                    for paragraph in response.split('\n\n'):
                        if len(current) + len(paragraph) < 1500:
                            current += paragraph + '\n\n'
                        else:
                            if current:
                                parts.append(current.strip())
                            current = paragraph + '\n\n'
                    if current:
                        parts.append(current.strip())
                    
                    logger.info(f"[{session_id[:8]}] SPLIT: {len(response)} znaków → {len(parts)} części")
                    for i, part in enumerate(parts):
                        await send_crisp_message(website_id, session_id, part)
                        if i < len(parts) - 1:
                            await asyncio.sleep(0.5)  # mały delay między częściami
                else:
                    logger.info(f"[{session_id[:8]}] WYSYŁAM: {len(response)} znaków")
                    await send_crisp_message(website_id, session_id, response)
                
                add_log("✅ Odpowiedź wysłana")

            if transfer:
                await send_crisp_message(website_id, session_id, "🔄 Łączę z konsultantem...")

        # Cleanup starych locków
        if len(_session_locks) > 200:
            _session_locks.clear()

        return JSONResponse({"status": "ok"}, status_code=200)

    except Exception as e:
        add_log(f"❌ BŁĄD: {str(e)[:50]}")
        logger.error(f"WEBHOOK ERROR: {e}", exc_info=True)
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

@app.post("/admin/rebuild-index")
async def admin_rebuild_index(request: Request):
    """Przebudowuje indeks FAISS z aktualnych plików w RAG_sources/. Użyj po dodaniu/edycji plików RAG."""
    admin_token = os.getenv("ADMIN_TOKEN", "")
    auth_header = request.headers.get("Authorization", "")
    if admin_token and auth_header != f"Bearer {admin_token}":
        raise HTTPException(status_code=401, detail="Unauthorized")

    try:
        from app.services.rag_service_faiss import rebuild_index
        global _rag_service_instance
        _rag_service_instance = None
        bot.rag_service = None

        logger.info("🔄 Rozpoczynam przebudowę indeksu FAISS...")
        success = await rebuild_index()

        if success:
            new_service = await get_rag_service()
            stats = await new_service.get_stats()
            return {"status": "ok", "documents": stats.get("documents_in_store", 0)}
        else:
            return JSONResponse({"status": "error", "detail": "Rebuild failed — check server logs"}, status_code=500)
    except Exception as e:
        logger.error(f"Rebuild error: {e}")
        return JSONResponse({"status": "error", "detail": str(e)}, status_code=500)

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