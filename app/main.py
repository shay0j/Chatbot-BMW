import os
import asyncio
import json
import base64
import hmac
import hashlib
import secrets
import traceback
import sys
from typing import Optional, Dict, Any, List, Tuple
from datetime import datetime
from pathlib import Path

import httpx
import cohere
from loguru import logger
from dotenv import load_dotenv
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse, RedirectResponse, HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.sessions import SessionMiddleware

# ============================================
# IMPORT NOWEGO RAG SERVICE
# ============================================

current_dir = Path(__file__).parent.absolute()
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

try:
    from app.services.rag_service import get_rag_service as get_rag_service_new
    from app.services.rag_service import RAGService
    RAG_AVAILABLE = True
    print("✅ Nowy RAG service załadowany")
except Exception as e:
    print(f"⚠️ Ostrzeżenie: Could not import new RAG module: {e}")
    RAG_AVAILABLE = False

    # Dummy RAG dla fallback
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
    """Zwraca singleton RAG service"""
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

def verify_crisp_signature(payload: bytes, signature: str, timestamp: str, secret: str) -> bool:
    """Weryfikuje sygnaturę webhooka z Crisp"""
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
    """Wysyła wiadomość przez API Crisp z pełnym logowaniem"""
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

        print(f"📤 Crisp wysyłanie do {session_id[:8]}...: {text[:50]}...")

        async with httpx.AsyncClient() as client:
            response = await client.post(url, headers=headers, json=payload)
            
            # Pełne logowanie odpowiedzi
            print(f"📊 Crisp response status: {response.status_code}")
            
            if response.status_code == 200:
                print(f"✅ Crisp: wiadomość wysłana pomyślnie (status 200)")
                try:
                    response_data = response.json()
                    print(f"📦 Crisp response data: {json.dumps(response_data)[:200]}")
                except:
                    print(f"📄 Crisp response text: {response.text[:200]}")
                return {"success": True}
            else:
                print(f"❌ Crisp błąd {response.status_code}: {response.text[:200]}")
                return {"success": False, "error": response.text}

    except Exception as e:
        print(f"❌ Crisp wyjątek przy wysyłaniu: {e}")
        traceback.print_exc()
        return {"success": False, "error": str(e)}

# ============================================
# COHERE SERVICE (Chat API)
# ============================================

class CohereService:
    def __init__(self):
        self.api_key = os.getenv("COHERE_API_KEY")
        self.model = os.getenv("COHERE_MODEL", "command-a-03-2025")
        
        if self.api_key:
            # Używamy ClientV2 dla Chat API
            self.client = cohere.ClientV2(self.api_key)
            print(f"✅ Cohere zainicjalizowany (model: {self.model}, API v2)")
        else:
            self.client = None
            print("⚠️ COHERE_API_KEY brak - używam fallback")

    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1000
    ) -> Dict[str, Any]:
        """Generuje odpowiedź używając Cohere Chat API"""
        
        if not self.client or not self.api_key:
            return {"success": False, "text": "Brak API"}

        try:
            # Przygotuj wiadomości w formacie Chat API
            messages = []
            
            # Dodaj system prompt jeśli istnieje
            if system_prompt:
                messages.append({
                    "role": "system",
                    "content": system_prompt
                })
            
            # Dodaj wiadomość użytkownika
            messages.append({
                "role": "user",
                "content": prompt
            })

            # Wywołanie Cohere Chat API (przez thread pool)
            response = await asyncio.to_thread(
                self.client.chat,
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )

            # Sprawdź odpowiedź
            if response and hasattr(response, 'message'):
                # Nowy format odpowiedzi - message.content to lista
                if hasattr(response.message, 'content') and len(response.message.content) > 0:
                    text = response.message.content[0].text
                else:
                    # Fallback do starego formatu
                    text = str(response.message)
                
                return {
                    "success": True,
                    "text": text,
                    "model": self.model
                }
            else:
                return {"success": False, "text": "Brak odpowiedzi od Cohere"}

        except Exception as e:
            logger.error(f"Cohere error: {e}")
            return {"success": False, "text": f"Błąd Cohere: {str(e)}"}

# ============================================
# GŁÓWNA KLASA BOTA (z RAG i Cohere)
# ============================================

class CrispBot:
    def __init__(self):
        logger.info("Inicjalizacja bota Crisp z RAG i Cohere...")
        self.conversation_state = {}
        self.rag_service = None
        self.cohere = CohereService()

        # Inicjalizuj RAG asynchronicznie
        asyncio.create_task(self._init_rag())

        logger.info("✅ Bot gotowy")

    async def _init_rag(self):
        """Inicjalizuje RAG service"""
        try:
            self.rag_service = await get_rag_service()
            health = await self.rag_service.health_check()
            stats = await self.rag_service.get_stats()

            if health.get("status") == "healthy":
                logger.info(f"✅ RAG zainicjalizowany: {stats.get('total_chunks', 0)} dokumentów")
            else:
                logger.warning(f"⚠️ RAG w stanie: {health.get('status')}")
        except Exception as e:
            logger.error(f"❌ RAG init failed: {e}")

    async def process_message(self, text: str, session_id: str) -> tuple[str, bool]:
        """Przetwarza wiadomość z użyciem RAG i Cohere"""
        text_lower = text.lower().strip()

        # Inicjalizuj stan
        if session_id not in self.conversation_state:
            self.conversation_state[session_id] = {
                "failed_attempts": 0,
                "last_topic": None,
                "context": []
            }

        state = self.conversation_state[session_id]

        # Sprawdź czy to prośba o konsultanta
        handoff_keywords = ['konsultant', 'człowiek', 'agent', 'handoff', 'konsultanta', 'człowiekiem']
        if any(keyword in text_lower for keyword in handoff_keywords):
            return "Łączę z konsultantem...", True

        # === RAG ===
        rag_results = {
            "has_data": False,
            "skip_rag": False,
            "confidence": 0.0,
            "intent": "general",
            "detected_models": [],
            "documents": [],
            "sources": []
        }

        if self.rag_service:
            try:
                rag_results = await self.rag_service.retrieve_with_intent_check(
                    query=text,
                    top_k=3,
                    confidence_threshold=0.5
                )
            except Exception as e:
                logger.error(f"RAG error: {e}")

        # Przygotuj kontekst z RAG
        rag_context = ""
        if rag_results.get("has_data") and rag_results.get("documents"):
            docs = rag_results.get("documents", [])[:3]
            context_parts = []
            for doc in docs:
                content = doc.get('content', '')[:500]
                metadata = doc.get('metadata', {})
                source = metadata.get('title', 'Źródło')[:50]
                context_parts.append(f"Z {source}:\n{content}")
            rag_context = "\n\n".join(context_parts)

        # Wykryj modele BMW (jeśli RAG nie podał)
        detected_models = rag_results.get('detected_models', [])
        if not detected_models:
            bmw_models = ['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'xm',
                         'i3', 'i4', 'i5', 'i7', 'i8', 'ix',
                         'm2', 'm3', 'm4', 'm5', 'm8', 'z4',
                         'seria 2', 'seria 3', 'seria 4', 'seria 5', 'seria 7', 'seria 8']

            for model in bmw_models:
                if model in text_lower:
                    detected_models.append(model.upper())

        # System prompt dla Cohere
        system_prompt = """Jesteś Leo - ekspertem BMW w ZK Motors, oficjalnym dealerze BMW i MINI.

ZASADY:
1. Odpowiadaj KONKRETNIE i RZETELNIE - maksymalnie 5-6 zdań
2. Używaj DANYCH Z KONTEKSTU - nie wymyślaj informacji
3. Jeśli brakuje danych - powiedz to i zaproś do salonu
4. Używaj polskiego języka
5. Jesteś przedstawicielem ZK Motors"""

        # Zbuduj prompt
        models_info = f"Wykryte modele: {', '.join(detected_models)}" if detected_models else ""
        context_section = f"DANE Z BAZY:\n{rag_context}" if rag_context else "BRAK DANYCH W BAZIE"

        # Historia rozmowy
        history = ""
        if len(state.get("context", [])) > 0:
            recent = state["context"][-4:]  # ostatnie 2 wymiany
            history = "HISTORIA:\n" + "\n".join([
                f"{'Klient' if i%2==0 else 'Ty'}: {msg['content']}"
                for i, msg in enumerate(recent)
            ])

        user_prompt = f"""PYTANIE: {text}

{models_info}

{context_section}

{history}

ODPOWIEDŹ (po polsku, 5-6 zdań):"""

        # Wywołaj Cohere
        cohere_result = await self.cohere.generate(
            prompt=user_prompt,
            system_prompt=system_prompt
        )

        if cohere_result.get("success"):
            response = cohere_result.get("text", "")
            print(f"✅ Cohere wygenerował odpowiedź")
        else:
            print(f"⚠️ Cohere błąd, używam fallback: {cohere_result.get('text')}")
            # Fallback
            response = self._fallback_response(text_lower, detected_models, rag_context, state)

        # Zapisz kontekst
        state["context"].append({"role": "user", "content": text})
        state["context"].append({"role": "assistant", "content": response})
        if len(state["context"]) > 10:
            state["context"] = state["context"][-10:]

        return response, False

    def _fallback_response(self, text_lower: str, detected_models: List[str],
                          rag_context: str, state: Dict) -> str:
        """Fallback gdy Cohere nie działa"""

        if any(word in text_lower for word in ['cześć', 'witaj', 'hej']):
            return "Cześć! Jestem Leo, ekspertem BMW w ZK Motors. W czym mogę pomóc?"
        elif any(word in text_lower for word in ['dziękuję', 'dzięki']):
            return "Proszę bardzo! Czy mogę pomóc w czymś jeszcze?"
        elif any(word in text_lower for word in ['godziny', 'otwarcia']):
            return "Jesteśmy czynni pon-pt 9:00-17:00, sob 9:00-14:00. Zapraszamy!"
        elif any(word in text_lower for word in ['adres', 'gdzie']):
            return "Nasza siedziba: ul. Przykładowa 123, Warszawa. Zapraszamy!"
        elif detected_models and rag_context:
            return f"Oto informacje o {', '.join(detected_models)}:\n{rag_context[:200]}...\n\nWięcej w salonie ZK Motors!"
        else:
            state["failed_attempts"] = state.get("failed_attempts", 0) + 1
            if state["failed_attempts"] >= 3:
                return "Przepraszam, nie mogę pomóc. Łączę z konsultantem."
            return "Przepraszam, nie zrozumiałem. Czy możesz powiedzieć inaczej?"

# ============================================
# FASTAPI SERVER
# ============================================

app = FastAPI(title="Crisp Bot z RAG i Cohere")

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])
app.add_middleware(SessionMiddleware, secret_key=os.getenv("SECRET_KEY", secrets.token_urlsafe(32)))

bot = CrispBot()

# Bufor na logi
_logs = []
_MAX_LOGS = 50

def add_log(message: str):
    """Dodaje log do wewnętrznego bufora"""
    global _logs
    timestamp = datetime.now().strftime('%H:%M:%S')
    _logs.append(f"[{timestamp}] {message}")
    if len(_logs) > _MAX_LOGS:
        _logs = _logs[-_MAX_LOGS:]

# ============================================
# ENDPOINTY
# ============================================

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
    <h3>🌍 URL webhooka (wklej to w marketplace):</h3>
    <div class="url" id="url"></div>
    <p><strong>WAŻNE:</strong> Użyj tego URL w marketplace → Settings → Events → Production</p>
    <h3>📋 Instrukcja:</h3>
    <ol>
        <li>Wejdź na <a href="https://marketplace.crisp.chat" target="_blank">Marketplace</a></li>
        <li>Twój plugin → Settings → Events</li>
        <li>Wklej URL powyżej w "Production"</li>
        <li>Zaznacz event <code>message:send</code></li>
        <li>Skopiuj "signing secret" do .env</li>
    </ol>
    <h3>🔍 Status:</h3>
    <ul>
        <li>RAG: <span id="rag-status"></span></li>
        <li>Cohere: <span id="cohere-status"></span></li>
    </ul>
    <h3>📊 Ostatnie logi:</h3>
    <pre id="logs" style="background:#f0f0f0; padding:10px; max-height:200px; overflow:auto;">Brak</pre>
</div>
<script>
    document.getElementById('url').innerText = window.location.origin + '/crisp/webhook';

    fetch('/rag/info').then(r=>r.json()).then(d=>{
        document.getElementById('rag-status').innerText = d.healthy ? '✅' : '⚠️ ' + d.status;
    });

    fetch('/cohere/info').then(r=>r.json()).then(d=>{
        document.getElementById('cohere-status').innerText = d.available ? '✅' : '❌ brak klucza';
    });

    // Odświeżaj logi co 2 sekundy
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
# GŁÓWNY ENDPOINT WEBHOOKA
# ============================================

@app.api_route("/crisp", methods=["POST", "GET"])
@app.api_route("/crisp/webhook", methods=["POST", "GET"])
@app.api_route("/webhook", methods=["POST", "GET"])
@app.api_route("/", methods=["POST"])
async def catch_all_webhooks(request: Request):
    """
    Uniwersalny endpoint łapiący wszystkie webhooki Crisp
    """
    try:
        body = await request.body()
        headers = dict(request.headers)
        path = request.url.path
        query = str(request.url.query)

        log_msg = f"\n{'='*60}\n📨 CRISP WEBHOOK at {datetime.now().strftime('%H:%M:%S')}\n📌 Ścieżka: {path}\n🔍 Query: {query}"
        print(log_msg)
        add_log(f"Webhook na {path}")

        if request.method == "GET":
            add_log("GET request - OK")
            return JSONResponse({"status": "webhook endpoint active"}, status_code=200)

        if not body:
            add_log("Brak body - ignoruję")
            return JSONResponse({"status": "no body"}, status_code=200)

        # Weryfikacja sygnatury
        if CRISP_WEBHOOK_SECRET:
            sig = headers.get("x-crisp-signature")
            ts = headers.get("x-crisp-request-timestamp")

            if sig and ts:
                if not verify_crisp_signature(body, sig, ts, CRISP_WEBHOOK_SECRET):
                    add_log("❌ Nieprawidłowa sygnatura!")
                    return JSONResponse({"status": "invalid signature"}, status_code=403)
                add_log("✅ Sygnatura poprawna")
            else:
                add_log("⚠️ Brak nagłówków sygnatury")

        # Parsuj JSON
        try:
            data = json.loads(body)
            event = data.get('event', 'unknown')
            add_log(f"Event: {event}")
            print(f"📦 Event: {event}")
        except json.JSONDecodeError as e:
            add_log(f"❌ Błąd parsowania JSON: {e}")
            return JSONResponse({"status": "invalid json"}, status_code=400)

        if event != 'message:send':
            add_log(f"⏭️ Pomijam event: {event}")
            return JSONResponse({"status": "ignored"}, status_code=200)

        msg_data = data.get('data', {})
        if msg_data.get('from') != 'user':
            sender = msg_data.get('from')
            add_log(f"⏭️ Pomijam wiadomość od: {sender}")
            return JSONResponse({"status": "ignored"}, status_code=200)

        website_id = data.get('website_id')
        session_id = msg_data.get('session_id')
        message = msg_data.get('content', '')

        add_log(f"Wiadomość: {message[:50]}...")
        print(f"💬 Wiadomość: {message[:200]}")

        # Przetwarzanie przez bota
        response, transfer = await bot.process_message(message, session_id)

        print(f"🤖 Odpowiedź: {response[:100]}...")
        add_log(f"Odpowiedź: {response[:50]}...")

        if response:
            result = await send_crisp_message(website_id, session_id, response)
            if result.get("success"):
                add_log("✅ Odpowiedź wysłana")
            else:
                add_log(f"❌ Błąd wysyłania: {result.get('error')}")

        if transfer:
            await send_crisp_message(website_id, session_id, "🔄 Łączę z konsultantem...")
            add_log("🔄 Transfer do konsultanta")

        add_log("✅ Webhook przetworzony")
        return JSONResponse({"status": "ok"}, status_code=200)

    except Exception as e:
        error_msg = f"❌ BŁĄD: {e}"
        print(error_msg)
        add_log(f"❌ BŁĄD: {str(e)[:50]}...")
        traceback.print_exc()
        return JSONResponse({"error": str(e)}, status_code=500)

# ============================================
# ENDPOINTY DIAGNOSTYCZNE
# ============================================

@app.get("/logs")
async def get_logs():
    """Zwraca ostatnie logi"""
    return "\n".join(_logs)

@app.get("/rag/info")
async def rag_info():
    """Informacje o RAG"""
    try:
        rag = await get_rag_service()
        health = await rag.health_check()
        stats = await rag.get_stats()
        return {
            "healthy": health.get("status") == "healthy",
            "status": health.get("status"),
            "documents": stats.get("documents_in_store", 0),
            "available": RAG_AVAILABLE
        }
    except Exception as e:
        return {"healthy": False, "available": RAG_AVAILABLE, "error": str(e)}

@app.get("/cohere/info")
async def cohere_info():
    """Informacje o Cohere"""
    return {
        "available": bool(os.getenv("COHERE_API_KEY")),
        "model": os.getenv("COHERE_MODEL", "command-a-03-2025"),
        "api_key_set": bool(os.getenv("COHERE_API_KEY"))
    }

@app.get("/crisp/test")
async def crisp_test():
    """Test konfiguracji Crisp"""
    return {
        "configured": True,
        "identifier": CRISP_IDENTIFIER[:5] + "..." if CRISP_IDENTIFIER else None,
        "webhook_secret": bool(CRISP_WEBHOOK_SECRET),
        "webhook_url": f"{BASE_URL}/crisp/webhook",
        "instruction": "Użyj /crisp/webhook w marketplace"
    }

@app.get("/health")
async def health():
    return {"status": "ok", "time": datetime.now().isoformat()}

# ============================================
# STARTUP
# ============================================

@app.on_event("startup")
async def startup():
    print("\n" + "="*60)
    print("🚀 CRISP BOT Z RAG I COHERE")
    print("="*60)
    print(f"📊 Panel: http://localhost:8000/panel")
    print(f"🌍 Webhook: {BASE_URL}/crisp/webhook")
    print(f"🔍 Test: {BASE_URL}/crisp/test")
    print("="*60)

    if RAG_AVAILABLE:
        print("✅ RAG dostępny")
    else:
        print("⚠️ RAG niedostępny - używa dummy")

    if os.getenv("COHERE_API_KEY"):
        print("✅ Cohere skonfigurowany")
    else:
        print("⚠️ Cohere brak - fallback responses")

    print("="*60 + "\n")

# ============================================
# MAIN
# ============================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)