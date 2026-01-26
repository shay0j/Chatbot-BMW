"""
Test integracji RAG z Cohere - NAPRAWIONA WERSJA BEZ PREAMBLE
"""
import asyncio
import sys
from pathlib import Path

# Dodaj ścieżkę do projektu
sys.path.append(str(Path(__file__).parent))

import cohere
from app.services.rag_service import get_rag_service
from app.core.config import settings

class BMWChatbotFixed:
    def __init__(self):
        self.rag_service = None
        self.cohere_client = None
        self.model_name = "command-r7b-12-2024"
        
    async def initialize(self):
        """Inicjalizuje RAG i Cohere"""
        print("🚀 Inicjalizacja BMW Chatbot Fixed...")
        
        # 1. Inicjalizuj RAG service
        self.rag_service = await get_rag_service()
        print("✅ RAG Service zainicjalizowany")
        
        # 2. Inicjalizuj Cohere client
        try:
            self.cohere_client = cohere.Client(api_key=settings.COHERE_API_KEY)
            print("✅ Cohere Client (Chat API) zainicjalizowany")
            print(f"🔄 Używam modelu: {self.model_name}")
            
        except Exception as e:
            print(f"❌ Błąd Cohere: {e}")
            return False
        
        return True
    
    async def get_rag_context(self, query: str) -> dict:
        """Pobiera kontekst z RAG"""
        return await self.rag_service.retrieve_with_intent_check(query)
    
    async def generate_with_cohere_chat(self, query: str, context: str, intent: str) -> str:
        """Używa TYLKO Cohere Chat API (bez preamble)"""
        
        # Buduj pełną wiadomość
        if intent == "price":
            system_rules = """ZASADY:
1. Odpowiadaj WYŁĄCZNIE po POLSKU
2. Używaj TYLKO informacji z kontekstu
3. Jeśli nie ma ceny, powiedz że potrzebny jest kontakt z dealerem
4. Jeśli jest zakres cen, podaj go
5. Bądź pomocny i profesjonalny"""
        
        elif intent == "technical":
            system_rules = """ZASADY:
1. Odpowiadaj WYŁĄCZNIE po POLSKU
2. Używaj TYLKO informacji z kontekstu
3. Podawaj konkretne liczby (KM, kW, zasięg, przyspieszenie)
4. Jeśli nie ma danych, powiedz o tym
5. Bądź precyzyjny"""
        
        else:
            system_rules = """ZASADY:
1. Odpowiadaj WYŁĄCZNIE po POLSKU
2. Używaj TYLKO informacji z kontekstu
3. Jeśli nie masz informacji, powiedz "Nie mam tej informacji w bazie, skontaktuj się z wybranym salonem ZK Motors."
4. Nigdy nie odpowiadaj na pytania o inne marki samochodów niż BMW i Mini
5. Nigdy nie odpowiadaj na pytania niezwiązane z samochodami BMW i Mini
6. Nigdy nie wymyślaj informacji
7. Bądź pomocny i konkretny"""

        # Buduj pełną wiadomość (system + kontekst + pytanie)
        full_message = f"""Jesteś asystentem BMW w Polsce.

{system_rules}

KONTEKST Z BMW.PL:
{context}

PYTANIE KLIENTA: {query}

ODPOWIEDŹ ASYSTENTA:"""
        
        # Użyj Cohere Chat API (tylko message)
        try:
            response = self.cohere_client.chat(
                model=self.model_name,
                message=full_message,
                temperature=0.3,
                max_tokens=300
            )
            return response.text.strip()
                
        except Exception as e:
            print(f"❌ Błąd Chat API: {e}")
            raise
    
    async def generate_response(self, query: str) -> str:
        """Generuje odpowiedź używając RAG + Cohere Chat API"""
        
        # 1. Pobierz kontekst z RAG
        rag_result = await self.get_rag_context(query)
        
        # Loguj info
        print(f"\n🔍 RAG dla '{query[:30]}...':")
        print(f"   - Has data: {rag_result['has_data']}")
        print(f"   - Intent: {rag_result['intent']}")
        print(f"   - Confidence: {rag_result['confidence']:.3f}")
        print(f"   - Documents: {rag_result.get('documents_retrieved', 0)}")
        
        # 2. Jeśli brak danych
        if not rag_result["has_data"]:
            return "Przepraszam, nie znalazłem informacji na ten temat w bazie danych BMW."
        
        # 3. Przygotuj kontekst (lepiej sformatowany)
        context_parts = []
        for i, doc in enumerate(rag_result["documents"][:3]):
            # Oczyść tekst - usuń nadmiarowe białe znaki
            content = ' '.join(doc["content"].split())
            # Skróć ale zachowaj zdania
            sentences = content.split('. ')
            if len(sentences) > 3:
                content = '. '.join(sentences[:3]) + '.'
            
            score = doc.get("score", 0)
            context_parts.append(f"[Fragment dokumentu {i+1}]: {content}")
        
        context = "\n\n".join(context_parts)
        
        # 4. Generuj odpowiedź z Cohere Chat API
        try:
            response = await self.generate_with_cohere_chat(
                query=query,
                context=context,
                intent=rag_result["intent"]
            )
            return response
            
        except Exception as e:
            print(f"❌ Błąd Chat API, używam fallback: {e}")
            return await self.fallback_response(rag_result)
    
    async def fallback_response(self, rag_result: dict) -> str:
        """Fallback gdy Cohere nie działa"""
        if not rag_result["has_data"]:
            return "Przepraszam, nie znalazłem informacji na ten temat."
        
        # Wybierz najlepszy dokument
        best_doc = max(rag_result["documents"], key=lambda x: x.get("score", 0))
        content = best_doc["content"]
        
        # Znajdź pierwsze pełne zdania
        sentences = content.split('. ')
        if len(sentences) > 3:
            preview = '. '.join(sentences[:3]) + '.'
        else:
            preview = '. '.join(sentences)
        
        intent = rag_result["intent"]
        
        if intent == "price":
            return f"Z informacji dostępnych: {preview}\n\nAby poznać dokładną cenę, skontaktuj się z dealerem BMW."
        elif intent == "technical":
            return f"Specyfikacja techniczna: {preview}"
        elif rag_result.get("detected_models"):
            model = rag_result["detected_models"][0]
            return f"Informacje o BMW {model}: {preview}"
        else:
            return f"Z bazy danych BMW: {preview}"
    
    async def chat_loop(self):
        """Interaktywna pętla chat"""
        print("\n" + "="*60)
        print("🤖 BMW CHATBOT FIXED - BEZ PREAMBLE")
        print("="*60)
        print("Zadawaj pytania o BMW!")
        print("'stats' - statystyki RAG")
        print("'test' - szybki test")
        print("'exit' - zakończ")
        print("="*60)
        
        while True:
            try:
                user_input = input("\n👤 Ty: ").strip()
                
                if user_input.lower() == 'exit':
                    print("\n🚗 BMW Assistant: Do widzenia! Dziękuję za rozmowę.")
                    break
                
                elif user_input.lower() == 'stats':
                    stats = await self.rag_service.get_stats()
                    print(f"\n📊 STATYSTYKI RAG:")
                    print(f"   Zapytania: {stats['queries_processed']}")
                    print(f"   Dokumenty: {stats['documents_in_store']}")
                    print(f"   Zwrócone: {stats['documents_retrieved']}")
                    print(f"   Cache hit rate: {stats['cache_hit_rate']:.1%}")
                    continue
                
                elif user_input.lower() == 'test':
                    await self.run_quick_test()
                    continue
                
                print("⏳ BMW Assistant: Myślę...")
                response = await self.generate_response(user_input)
                print(f"\n🤖 BMW Assistant: {response}")
                
            except KeyboardInterrupt:
                print("\n\nPrzerwano przez użytkownika.")
                break
            except Exception as e:
                print(f"\n❌ Błąd: {e}")

    async def run_quick_test(self):
        """Szybki test"""
        print("\n🧪 Szybki test...")
        
        test_queries = [
            "BMW i4",
            "Ile kosztuje BMW X3?",
            "Moc silnika BMW X5",
            "Zasięg BMW iX3",
        ]
        
        for query in test_queries:
            print(f"\n{'='*40}")
            print(f"📝 {query}")
            response = await self.generate_response(query)
            print(f"🤖 {response}")
            await asyncio.sleep(1)

async def main():
    """Główna funkcja"""
    chatbot = BMWChatbotFixed()
    
    # Inicjalizacja
    if not await chatbot.initialize():
        print("\n❌ Nie udało się zainicjalizować.")
        return
    
    # Szybkie testy
    print("\n🧪 Szybki test Chat API (bez preamble)...")
    await chatbot.run_quick_test()
    
    print("\n✅ Test zakończony! Rozpoczynam interaktywny chat...")
    await chatbot.chat_loop()

if __name__ == "__main__":
    asyncio.run(main())