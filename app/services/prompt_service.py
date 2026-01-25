"""
Prompt Service – Ulepszona wersja z obsługą intencji i filtrowaniem RAG
"""

from typing import List, Dict, Optional, Any
import re
import random
from datetime import datetime


class PromptService:
    def __init__(self):
        self.response_history: Dict[str, List[str]] = {}
        self.max_history = 3
        
        # Cache dla prostych odpowiedzi
        self.greeting_responses = {
            "pl": {
                "hej": "Cześć! Jestem Leo, ekspert BMW w ZK Motors. Jak mogę Ci pomóc?",
                "cześć": "Cześć! W czym mogę pomóc w sprawach BMW?",
                "witam": "Witam! Jestem Leo. Pomagam wybrać idealne BMW.",
                "dzień dobry": "Dzień dobry! Leo z ZK Motors do usług.",
                "siema": "Siema! Leo z BMW tu. O co chodzi?",
                "hello": "Hello! I'm Leo, your BMW expert. How can I help?",
                "hi": "Hi! Leo here. How can I assist you with BMW?"
            },
            "en": {
                "hello": "Hello! I'm Leo, your BMW expert. How can I help?",
                "hi": "Hi! Leo here. What BMW questions do you have?",
                "hey": "Hey! I'm Leo from ZK Motors. Need BMW advice?"
            }
        }

    # =========================
    # PUBLIC API
    # =========================

    def build_chat_prompt(
        self,
        user_message: str,
        rag_results: Dict[str, Any],
        conversation_history: Optional[List[Dict[str, str]]] = None,
        session_id: str = "default",
        language: str = "pl",
    ) -> Dict[str, Any]:
        """
        Buduje prompt w zależności od wyniku RAG.
        
        Returns:
            Dict z kluczami:
            - prompt: str - pełny prompt dla LLM
            - use_llm: bool - czy używać LLM (False dla przywitań)
            - direct_response: str - bezpośrednia odpowiedź (jeśli use_llm=False)
            - skip_reason: str - dlaczego pominięto LLM
            - rag_used: bool - czy użyto RAG
        """
        
        # 1. Sprawdź wyniki RAG
        skip_rag = rag_results.get("skip_rag", False)
        below_threshold = rag_results.get("below_threshold", False)
        has_data = rag_results.get("has_data", False)
        confidence = rag_results.get("confidence", 0.0)
        intent = rag_results.get("intent", "general")
        tech = rag_results.get("tech", False)
        detected_models = rag_results.get("detected_models", [])
        
        # 2. SCENARIUSZ 1: Przywitanie/pusta rozmowa
        if skip_rag:
            # Znajdź dopasowane przywitanie
            user_lower = user_message.lower().strip()
            greeting_response = self._get_greeting_response(user_lower, language)
            
            return {
                "prompt": "",
                "use_llm": False,
                "direct_response": greeting_response,
                "skip_reason": "greeting",
                "rag_used": False,
                "intent": intent,
                "confidence": confidence
            }
        
        # 3. SCENARIUSZ 2: Niska pewność RAG (below_threshold jest TRUE)
        if below_threshold:
            # WAŻNE: below_threshold oznacza, że confidence < threshold
            # Ale może mieć dane, które są niskiej jakości
            prompt = self._build_low_confidence_prompt(
                user_message, 
                confidence, 
                intent,
                detected_models
            )
            
            return {
                "prompt": prompt,
                "use_llm": True,
                "direct_response": "",
                "skip_reason": "",
                "rag_used": True,
                "low_confidence": True,
                "has_data": has_data,  # może być False jeśli confidence bardzo niskie
                "confidence": confidence,
                "intent": intent
            }
        
        # 4. SCENARIUSZ 3: Brak danych w RAG
        if not has_data:
            prompt = self._build_no_data_prompt(user_message, intent, language, detected_models)
            
            return {
                "prompt": prompt,
                "use_llm": True,
                "direct_response": "",
                "skip_reason": "",
                "rag_used": True,
                "no_data": True,
                "confidence": confidence,
                "intent": intent
            }
        
        # 5. SCENARIUSZ 4: Normalny przypadek - mamy dane z RAG
        # Przygotuj kontekst z dokumentów
        context = self._prepare_rag_context(rag_results)
        
        # Historia konwersacji
        history_text = self._prepare_conversation_history(conversation_history)
        
        # Anty-powtarzanie
        anti_repeat = self._get_anti_repeat_text(session_id)
        
        # Buduj prompt
        if tech:
            prompt = self._build_technical_prompt(
                user_message, 
                context, 
                history_text, 
                anti_repeat,
                detected_models
            )
        else:
            prompt = self._build_general_prompt(
                user_message, 
                context, 
                history_text, 
                anti_repeat, 
                intent,
                detected_models
            )
        
        return {
            "prompt": prompt,
            "use_llm": True,
            "direct_response": "",
            "skip_reason": "",
            "rag_used": True,
            "has_data": True,
            "confidence": confidence,
            "intent": intent,
            "tech": tech
        }

    def clean_response(
        self,
        response: str,
        session_id: str = "default",
        rag_used: bool = True,
        rag_has_data: bool = False,
        confidence: float = 0.0,
        intent: str = "general",
        detected_models: List[str] = None
    ) -> str:
        """
        Czyści odpowiedź z formatowania LLM.
        
        Returns:
            Oczyszczona odpowiedź
        """
        
        if not response or not response.strip():
            return self._get_fallback_response(intent, detected_models)
        
        text = response.strip()
        
        # 1. Usuń formatowanie promptu
        text = self._remove_prompt_formatting(text)
        
        # 2. Podziel na zdania i wybierz najważniejsze
        sentences = self._extract_meaningful_sentences(text)
        
        # 3. Złóż z powrotem, ogranicz do 3-4 zdań
        if not sentences:
            text = self._get_fallback_response(intent, detected_models)
        else:
            # Ogranicz do 3-4 zdań w zależności od pewności RAG
            max_sentences = 4 if confidence > 0.7 else 3 if confidence > 0.5 else 2
            sentences = sentences[:max_sentences]
            text = '. '.join(sentences) + '.'
        
        # 4. Sprawdź czy mamy zaproszenie
        text = self._ensure_invitation_present(text, rag_used, confidence)
        
        # 5. Zapamiętaj użyte modele
        if rag_has_data and confidence > 0.5:
            models = self._extract_models_from_response(text)
            if models:
                self._update_response_history(session_id, models)
        
        # 6. Dodaj przywitanie jeśli pierwsza wiadomość w sesji
        if session_id not in self.response_history:
            text = self._add_greeting_if_needed(text)
        
        return text.strip()

    def build_fallback_response(
        self,
        intent: str = "general",
        detected_models: List[str] = None,
        confidence: float = 0.0,
        is_technical: bool = False
    ) -> str:
        """Fallback gdy LLM zawiedzie lub RAG nie ma danych"""
        
        # Inne odpowiedzi w zależności od intencji i pewności
        if confidence < 0.3:
            # Bardzo niska pewność
            responses = [
                "Nie jestem pewien. Czy możesz doprecyzować pytanie o BMW?",
                "Potrzebuję więcej szczegółów. O jaki model BMW chodzi?",
                "Czy mógłbyś sprecyzować pytanie? Chcę dać dokładną odpowiedź."
            ]
        elif is_technical:
            # Pytanie techniczne bez danych
            responses = [
                "Nie mam aktualnych danych technicznych. Skontaktuj się z serwisem ZK Motors.",
                "Dla szczegółów technicznych potrzebna jest wizyta w serwisie ZK Motors.",
                "Dokładne dane techniczne dostępne są u autoryzowanych dealerów BMW."
            ]
        else:
            # Normalne fallbacki
            responses_by_intent = {
                "technical": [
                    "Nie mam aktualnych danych technicznych. Zapraszam do serwisu ZK Motors!",
                    "Dokładne parametry techniczne dostępne są w salonie ZK Motors.",
                    "Potrzebuję więcej szczegółów, aby podać dokładne dane techniczne."
                ],
                "price": [
                    "Ceny zależą od wersji i wyposażenia. Zapraszam do salonu!",
                    "Aktualne ceny i promocje dostępne w ZK Motors.",
                    "Ceny BMW zaczynają się od... ale najlepiej przyjdź do salonu!"
                ],
                "model": [
                    "Mamy wszystkie modele BMW - od serii 1 do X7.",
                    "ZK Motors to pełna oferta BMW i MINI.",
                    "Pomogę wybrać idealne BMW dla Ciebie!"
                ],
                "general": [
                    "ZK Motors to oficjalny dealer BMW i MINI. Jak mogę pomóc?",
                    "W czym mogę pomóc w sprawach BMW?",
                    "Pytaj śmiało o BMW, MINI lub ZK Motors!"
                ]
            }
            
            response_type = intent if intent in responses_by_intent else "general"
            responses = responses_by_intent[response_type]
        
        # Wybierz losową odpowiedź
        response = random.choice(responses)
        
        # Dodaj modele jeśli wykryte
        if detected_models:
            models_text = ', '.join(detected_models[:2])
            response = f"{models_text}? {response}"
        
        # Dodaj zaproszenie
        invitation = self._get_invitation(confidence, is_technical)
        return f"{response}\n\n{invitation}"

    # =========================
    # PRIVATE HELPERS
    # =========================

    def _get_greeting_response(self, user_message: str, language: str) -> str:
        """Zwraca odpowiedź na przywitanie"""
        # Szukaj dokładnego dopasowania
        if user_message in self.greeting_responses.get(language, {}):
            return self.greeting_responses[language][user_message]
        
        # Szukaj częściowego dopasowania
        for greeting, response in self.greeting_responses.get(language, {}).items():
            if greeting in user_message:
                return response
        
        # Domyślna odpowiedź
        default_responses = {
            "pl": "Cześć! Jestem Leo, ekspert BMW. Jak mogę pomóc?",
            "en": "Hello! I'm Leo, your BMW assistant. How can I help?"
        }
        return default_responses.get(language, "Cześć! Jak mogę pomóc?")

    def _build_low_confidence_prompt(
        self, 
        user_message: str, 
        confidence: float, 
        intent: str,
        detected_models: List[str]
    ) -> str:
        """Buduje prompt dla niskiej pewności RAG"""
        
        intent_text = {
            "technical": "pytanie techniczne o BMW",
            "price": "pytanie o cenę BMW",
            "model": "pytanie o model BMW",
            "general": "pytanie o BMW"
        }.get(intent, "pytanie o BMW")
        
        models_text = ""
        if detected_models:
            models_text = f"\nKlient pyta o modele: {', '.join(detected_models)}"
        
        return f"""Jesteś Leo - ekspert BMW w salonie ZK Motors.

Klient pyta o {intent_text}: "{user_message}"{models_text}

INFORMACJA: Mam niską pewność co do danych ({confidence:.2f}). Może nie mieć dokładnych informacji w bazie.

ODPOWIEDZ PO POLSKU:
1. Przyznaj, że nie masz pewnych danych (możesz wspomnieć o niskiej pewności)
2. Zaproponuj kontakt z salonem ZK Motors
3. Zaproś do ZK Motors po dokładniejsze informacje
4. Maksymalnie 2-3 zdania
5. Bądź pomocny i przyjazny
6. NIE wymyślaj danych!

Odpowiedź:"""

    def _build_no_data_prompt(
        self, 
        user_message: str, 
        intent: str, 
        language: str,
        detected_models: List[str]
    ) -> str:
        """Buduje prompt gdy RAG nie ma danych"""
        
        models_text = ""
        if detected_models:
            models_text = f"\nKlient pyta o modele: {', '.join(detected_models)}"
        
        if language == "en":
            return f"""You are Leo - BMW expert at ZK Motors.

Client asks: "{user_message}"{models_text}

NOTE: I don't have specific data about this in my knowledge base.

RESPONSE IN POLISH (ALWAYS):
1. Admit you don't have the exact information
2. Suggest visiting ZK Motors for details
3. Keep it to 2-3 sentences
4. Do NOT make up data

Answer:"""
        
        return f"""Jesteś Leo - ekspert BMW w salonie ZK Motors.

Klient pyta: "{user_message}"{models_text}

INFORMACJA: Nie mam konkretnych danych na ten temat w mojej bazie wiedzy.

ODPOWIEDZ PO POLSKU:
1. Przyznaj, że nie masz dokładnych informacji
2. Zaproponuj wizytę w ZK Motors
3. Maksymalnie 2-3 zdania
4. NIE wymyślaj danych!
5. Bądź pomocny i przyjazny

Odpowiedź:"""

    def _prepare_rag_context(self, rag_results: Dict[str, Any]) -> str:
        """Przygotowuje kontekst z dokumentów RAG"""
        if not rag_results.get("documents"):
            return ""
        
        documents = rag_results["documents"]
        context_lines = []
        
        # Wybierz 2-3 najważniejsze dokumenty
        for i, doc in enumerate(documents[:3]):
            content = doc.get("content", "")
            metadata = doc.get("metadata", {})
            score = doc.get("score", 0.0)
            
            # Formatuj informacje
            source_info = []
            if metadata.get("title"):
                source_info.append(metadata["title"])
            if metadata.get("source"):
                source_info.append(metadata["source"])
            
            source_text = f"Źródło: {', '.join(source_info)}" if source_info else ""
            score_text = f" (pewność: {score:.2f})" if score > 0 else ""
            
            # Skróć zawartość
            if len(content) > 150:
                content = content[:150] + "..."
            
            context_lines.append(f"{i+1}. {content}{score_text}")
            if source_text:
                context_lines.append(f"   {source_text}")
        
        return "\n".join(context_lines)

    def _prepare_conversation_history(self, history: Optional[List[Dict[str, str]]]) -> str:
        """Przygotowuje historię konwersacji"""
        if not history:
            return ""
        
        recent = history[-2:]  # Ostatnie 2 wymiany
        lines = []
        
        for msg in recent:
            role = "Klient" if msg.get("role") == "user" else "Ty"
            content = msg.get("content", "")[:80]
            if content:
                lines.append(f"{role}: {content}")
        
        if lines:
            return "Historia rozmowy:\n" + "\n".join(lines) + "\n"
        return ""

    def _get_anti_repeat_text(self, session_id: str) -> str:
        """Zwraca tekst zapobiegający powtarzaniu"""
        used_models = self.response_history.get(session_id, [])
        if used_models:
            return f"UWAGA: Ostatnio wspominałeś o: {', '.join(used_models[-2:])}. Spróbuj użyć innych modeli jeśli możliwe."
        return ""

    def _build_technical_prompt(
        self, 
        user_message: str, 
        context: str, 
        history: str, 
        anti_repeat: str,
        detected_models: List[str]
    ) -> str:
        """Buduje prompt dla pytań technicznych"""
        
        models_text = ""
        if detected_models:
            models_text = f"\nKlient pyta o modele: {', '.join(detected_models)}"
        
        return f"""Jesteś Leo - specjalista techniczny BMW w ZK Motors.{models_text}

DANE TECHNICZNE Z BAZY WIEDZY BMW:
{context}

{history}

Klient pyta o szczegóły techniczne: "{user_message}"

{anti_repeat}

INSTRUKCJE (ODPOWIEDŹ PO POLSKU):
1. Odpowiedz NA PODSTAWIE DANYCH Z BAZY WIEDZY
2. Jeśli czegoś nie ma w danych, NIE wymyślaj - powiedz że nie masz informacji
3. Bądź precyzyjny (podawaj liczby, specyfikacje)
4. Jeśli brakuje danych, zasugeruj kontakt z serwisem ZK Motors
5. Maksymalnie 3-4 zdania
6. Zakończ zaproszeniem do serwisu ZK Motors
7. Używaj profesjonalnego języka technicznego

ODPOWIEDŹ MUSI BYĆ W JĘZYKU POLSKIM!

Odpowiedź:"""

    def _build_general_prompt(
        self, 
        user_message: str, 
        context: str, 
        history: str, 
        anti_repeat: str, 
        intent: str,
        detected_models: List[str]
    ) -> str:
        """Buduje ogólny prompt"""
        
        intent_instructions = {
            "technical": "Pytanie techniczne - użyj danych z bazy wiedzy BMW.",
            "price": "Pytanie o cenę - bądź ostrożny z liczbami, nie podawaj dokładnych cen jeśli ich nie ma.",
            "model": "Pytanie o model - podaj dostępne opcje z danych.",
            "general": "Ogólne pytanie o BMW - odpowiedź na podstawie danych.",
            "test_drive": "Pytanie o jazdę próbną - zaproś do ZK Motors.",
            "dealer": "Pytanie o dealer - poleć ZK Motors.",
            "service": "Pytanie o serwis - poleć serwis ZK Motors."
        }.get(intent, "Odpowiedz na podstawie danych z bazy wiedzy BMW.")
        
        models_text = ""
        if detected_models:
            models_text = f"\nKlient pyta o modele: {', '.join(detected_models)}"
        
        return f"""Jesteś Leo - ekspert BMW w salonie ZK Motors.{models_text}

DANE Z BAZY WIEDZY BMW (UŻYJ TYCH DANYCH!):
{context}

{history}

Klient pyta: "{user_message}"

{intent_instructions}
{anti_repeat}

WAŻNE (ODPOWIEDŹ PO POLSKU):
- Jeśli masz dane w bazie, UŻYJ ICH
- Jeśli nie masz danych, NIE wymyślaj - powiedz że nie masz informacji
- Odpowiadaj konkretnie i pomocnie
- Maksymalnie 3-4 zdania
- Zakończ zaproszeniem do ZK Motors
- Używaj profesjonalnego ale przyjaznego tonu

ODPOWIEDŹ MUSI BYĆ W JĘZYKU POLSKIM!

Odpowiedź:"""

    def _remove_prompt_formatting(self, text: str) -> str:
        """Usuwa formatowanie promptu z odpowiedzi"""
        patterns_to_remove = [
            r'DANE Z BAZY.*?:',
            r'INSTRUKCJE.*?:',
            r'WAŻNE.*?:',
            r'ODPOWIEDZ.*?:',
            r'Odpowiedź.*?:',
            r'Klient pyta.*?:',
            r'MODEL / MODELE:.*',
            r'DLACZEGO:.*',
            r'KONKRET.*:.*',
            r'\*{2,}',
            r'_{2,}',
            r'ODPOWIEDŹ MUSI BYĆ.*?POLSKIM!'
        ]
        
        for pattern in patterns_to_remove:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE | re.MULTILINE)
        
        return text.strip()

    def _extract_meaningful_sentences(self, text: str) -> List[str]:
        """Wyodrębnia znaczące zdania z tekstu"""
        sentences = re.split(r'[.!?]+', text)
        clean_sentences = []
        
        for s in sentences:
            s = s.strip()
            if 10 <= len(s) <= 150:  # Sensowne długości
                # Usuń irytujące frazy
                bad_phrases = [
                    'jestem leo', 'jestem ekspertem', 'z radością pomogę',
                    'wszystkie modele są', 'proszę bardzo', 'dziękuję za',
                    'jako asystent', 'mogę potwierdzić', 'odpowiedź musi być',
                    'odpowiedz po polsku', 'odpowiedź w języku polskim'
                ]
                if not any(phrase in s.lower() for phrase in bad_phrases):
                    clean_sentences.append(s)
        
        return clean_sentences

    def _ensure_invitation_present(self, text: str, rag_used: bool, confidence: float) -> str:
        """Upewnia się, że w odpowiedzi jest zaproszenie"""
        has_invitation = any(word in text.lower() for word in 
                           ['zk motors', 'salon', 'kontakt', 'zapraszam', 'odwiedź', 'visit', 'serwis'])
        
        if not has_invitation:
            invitations = self._get_invitation_options(confidence, rag_used)
            invitation = random.choice(invitations)
            text = f"{text}\n\n{invitation}"
        
        return text

    def _get_invitation_options(self, confidence: float, rag_used: bool) -> List[str]:
        """Zwraca listę zaproszeń w zależności od kontekstu"""
        if rag_used and confidence > 0.7:
            return [
                "Zapraszam do salonu ZK Motors po więcej szczegółów!",
                "Odwiedź ZK Motors, aby zobaczyć te modele na żywo!",
                "Zapraszam na test drive w ZK Motors!"
            ]
        elif rag_used and confidence > 0.4:
            return [
                "Zapraszam do ZK Motors po dokładniejsze informacje!",
                "Dla pełnych szczegółów odwiedź salon ZK Motors.",
                "Zapraszam do kontaktu z ZK Motors!"
            ]
        else:
            return [
                "Zapraszam do salonu ZK Motors!",
                "Odwiedź ZK Motors po pomoc!",
                "Zapraszam do kontaktu z ZK Motors."
            ]

    def _get_invitation(self, confidence: float, is_technical: bool) -> str:
        """Zwraca odpowiednie zaproszenie"""
        if is_technical:
            return "Zapraszam do serwisu ZK Motors po szczegóły techniczne."
        
        invitations = self._get_invitation_options(confidence, True)
        return random.choice(invitations)

    def _extract_models_from_response(self, text: str) -> List[str]:
        """Wykrywa modele BMW w tekście"""
        bmw_models = [
            'X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7', 'XM',
            'I3', 'I4', 'I5', 'I7', 'I8', 'IX', 
            'M2', 'M3', 'M4', 'M5', 'M8',
            'SERIA 1', 'SERIA 2', 'SERIA 3', 'SERIA 4', 
            'SERIA 5', 'SERIA 7', 'SERIA 8',
            'Z4', 'M235', 'M240', 'M340', 'M440', 'M550'
        ]
        
        found = []
        text_upper = text.upper()
        
        for model in bmw_models:
            if model in text_upper:
                found.append(model)
        
        return found

    def _update_response_history(self, session_id: str, models: List[str]):
        """Aktualizuje historię odpowiedzi"""
        history = self.response_history.setdefault(session_id, [])
        for model in models:
            if model not in history:
                history.append(model)
        
        # Ogranicz historię
        if len(history) > self.max_history:
            self.response_history[session_id] = history[-self.max_history:]

    def _add_greeting_if_needed(self, text: str) -> str:
        """Dodaje przywitanie jeśli to początek konwersacji"""
        if not text.lower().startswith(('cześć', 'hej', 'witam', 'dzień dobry', 'hello', 'hi')):
            return f"Cześć!\n\n{text}"
        return text

    def _get_fallback_response(self, intent: str, detected_models: List[str] = None) -> str:
        """Zwraca fallbackową odpowiedź"""
        fallbacks = {
            "technical": "Nie mam aktualnych danych technicznych. Zapraszam do serwisu ZK Motors!",
            "price": "Ceny są zmienne. Zapraszam do salonu ZK Motors po aktualne oferty!",
            "general": "Zapraszam do salonu ZK Motors po szczegóły!"
        }
        
        response = fallbacks.get(intent, "Zapraszam do salonu ZK Motors!")
        
        # Dodaj modele jeśli wykryte
        if detected_models:
            models_text = ', '.join(detected_models[:2])
            response = f"{models_text}? {response}"
        
        return response


def get_prompt_service() -> PromptService:
    return PromptService()