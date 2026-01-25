"""
Test integracji RAG z Cohere - ZAKTUALIZOWANA WERSJA
Używa modelu command-r7b-12-2024 zamiast command
"""
import asyncio
import sys
import os
from pathlib import Path

# Dodaj ścieżkę do projektu
sys.path.append(str(Path(__file__).parent))

import cohere
from app.services.rag_service import get_rag_service
from app.core.config import settings

class BMWChatbotV2:
    def __init__(self):
        self.rag_service = None
        self.cohere_client = None
        self.model_name = "command-r7b-12-2024"  # Użyj modelu z configu
        
    async def initialize(self):
        """Inicjalizuje RAG i Cohere"""
        print("🚀 Inicjalizacja BMW Chatbot v2...")
        
        # 1. Inicjalizuj RAG service
        self.rag_service = await get_rag_service()
        print("✅ RAG Service zainicjalizowany")
        
        # 2. Inicjalizuj Cohere client
        try:
            self.cohere_client = cohere.Client(api_key=settings.COHERE_API_KEY)
            print("✅ Cohere Client zainicjalizowany")
            
            # Sprawdź dostępne modele
            print(f"🔄 Używam modelu: {self.model_name}")
            
        except Exception as e:
            print(f"❌ Błąd Cohere: {e}")
            return False
        
        return True
    
    async def get_rag_context(self, query: str) -> dict:
        """Pobiera kontekst z RAG"""
        return await self.rag_service.retrieve_with_intent_check(query)
    
    def build_prompt(self, query: str, rag_result: dict) -> str:
        """Buduje prompt dla Cohere"""
        
        if not rag_result["has_data"]:
            return f'''Jesteś asystentem BMW. Klient zadał pytanie, ale nie masz informacji w bazie.

Pytanie: {query}

Odpowiedz: "Przepraszam, nie znalazłem informacji na ten temat w bazie danych BMW. Czy mogę pomóc w czymś innym?"'''
        
        # Przygotuj kontekst
        context_parts = ["INFORMACJE Z BMW.PL:"]
        for i, doc in enumerate(rag_result["documents"][:3]):
            content = ' '.join(doc["content"].split()[:80])  # Pierwsze 80 słów
            context_parts.append(f"{i+1}. {content}")
        
        context = "\n\n".join(context_parts)
        
        # Detekcja intencji dla lepszej odpowiedzi
        intent = rag_result["intent"]
        intent_hint = ""
        
        if intent == "price":
            intent_hint = "Jeśli pytasz o cenę, podaj zakres cenowy jeśli jest w informacjach."
        elif intent == "technical":
            intent_hint = "Jeśli pytasz o specyfikację, podaj konkretne liczby jeśli są w informacjach."
        
        prompt = f'''Jesteś asystentem BMW w Polsce. Twoim zadaniem jest odpowiadanie na pytania klientów używając TYLKO poniższych informacji z oficjalnej strony bmw.pl.

{context}

WAŻNE ZASADY:
1. Odpowiadaj WYŁĄCZNIE po POLSKU
2. Używaj TYLKO informacji z powyższego kontekstu
3. Jeśli odpowiedzi nie ma w kontekście, powiedz "Nie mam tej informacji w bazie danych"
4. Bądź konkretny i pomocny
5. {intent_hint}

PYTANIE KLIENTA: {query}

ODPOWIEDŹ ASYSTENTA BMW (krótko i na temat):'''
        
        return prompt
    
    async def generate_response(self, query: str) -> str:
        """Generuje odpowiedź używając RAG + Cohere"""
        
        # 1. Pobierz kontekst z RAG
        rag_result = await self.get_rag_context(query)
        
        # Loguj info
        print(f"\n🔍 RAG dla '{query[:30]}...':")
        print(f"   - Has data: {rag_result['has_data']}")
        print(f"   - Intent: {rag_result['intent']}")
        print(f"   - Confidence: {rag_result['confidence']:.3f}")
        
        # 2. Zbuduj prompt
        prompt = self.build_prompt(query, rag_result)
        
        # 3. Wywołaj Cohere API
        try:
            # SPRAWDŹ CZY TO DZIAŁA - różne podejścia
            
            # Metoda 1: generate() - może jeszcze działać
            try:
                response = self.cohere_client.generate(
                    model=self.model_name,
                    prompt=prompt,
                    max_tokens=250,
                    temperature=0.3,
                    truncate='END'
                )
                return response.generations[0].text.strip()
            except Exception as e1:
                print(f"⚠️  Generate nie działa: {e1}")
                
                # Metoda 2: chat() bez chat_history
                try:
                    response = self.cohere_client.chat(
                        model=self.model_name,
                        message=prompt,
                        temperature=0.3,
                        max_tokens=250
                    )
                    return response.text.strip()
                except Exception as e2:
                    print(f"⚠️  Chat nie działa: {e2}")
                    
                    # Metoda 3: chat() z message tylko jako query
                    try:
                        system_msg = prompt.split("PYTANIE KLIENTA:")[0]
                        user_query = query
                        
                        response = self.cohere_client.chat(
                            model=self.model_name,
                            message=user_query,
                            preamble=system_msg,
                            temperature=0.3,
                            max_tokens=250
                        )
                        return response.text.strip()
                    except Exception as e3:
                        print(f"⚠️  Chat z preamble nie działa: {e3}")
                        
                        # Fallback
                        return await self.fallback_response(rag_result)
                        
        except Exception as e:
            print(f"❌ Wszystkie metody Cohere zawiodły: {e}")
            return await self.fallback_response(rag_result)
    
    async def fallback_response(self, rag_result: dict) -> str:
        """Fallback gdy Cohere nie działa"""
        if not rag_result["has_data"]:
            return "Przepraszam, nie znalazłem informacji na ten temat."
        
        first_doc = rag_result["documents"][0]["content"]
        words = first_doc.split()[:40]
        preview = " ".join(words) + ("..." if len(words) == 40 else "")
        
        intent = rag_result["intent"]
        
        if intent == "price":
            return f"Z informacji dostępnych: {preview} Aby poznać dokładną cenę, skontaktuj się z dealerem BMW."
        elif intent == "technical":
            return f"Specyfikacja: {preview}"
        else:
            return f"Informacje: {preview}"
    
    async def chat_loop(self):
        """Interaktywna pętla chat"""
        print("\n" + "="*60)
        print("🤖 BMW CHATBOT v2 - command-r7b-12-2024")
        print("="*60)
        print("Zadawaj pytania o BMW!")
        print("'stats' - statystyki RAG")
        print("'exit' - zakończ")
        print("="*60)
        
        while True:
            try:
                user_input = input("\n👤 Ty: ").strip()
                
                if user_input.lower() == 'exit':
                    print("\n🚗 Do widzenia!")
                    break
                
                elif user_input.lower() == 'stats':
                    stats = await self.rag_service.get_stats()
                    print(f"\n📊 Statystyki: {stats['queries_processed']} zapytań, {stats['documents_in_store']} dokumentów")
                    continue
                
                print("⏳ Myślę...")
                response = await self.generate_response(user_input)
                print(f"\n🤖 BMW Assistant: {response}")
                
            except KeyboardInterrupt:
                print("\n\nDo widzenia!")
                break
            except Exception as e:
                print(f"\n❌ Błąd: {e}")

async def main():
    """Główna funkcja"""
    chatbot = BMWChatbotV2()
    
    # Inicjalizacja
    if not await chatbot.initialize():
        print("\n❌ Nie udało się zainicjalizować.")
        print("Sprawdź:")
        print("1. Klucz API Cohere w .env lub config.py")
        print("2. Czy model 'command-r7b-12-2024' jest dostępny")
        print("3. Czy masz aktualną wersję biblioteki cohere")
        return
    
    # Szybkie testy
    print("\n🧪 Testuję podstawowe zapytania...")
    
    test_queries = [
        "BMW X3",
        "Ile kosztuje BMW X5?",
        "Moc silnika Seria 3",
    ]
    
    for query in test_queries:
        print(f"\n{'='*40}")
        print(f"📝 Zapytanie: {query}")
        
        response = await chatbot.generate_response(query)
        print(f"🤖 Odpowiedź: {response}")
        
        await asyncio.sleep(1)
    
    print("\n✅ Testy zakończone! Rozpoczynam interaktywny chat...")
    await chatbot.chat_loop()

if __name__ == "__main__":
    asyncio.run(main())