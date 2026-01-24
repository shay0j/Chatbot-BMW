import faiss
import numpy as np
import pickle
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from sentence_transformers import SentenceTransformer
from datetime import datetime
import logging
import re

# Konfiguracja logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RAGSystem:
    """Kompletny system RAG dla BMW z inteligentnym filtrowaniem po modelach"""
    
    def __init__(self, vector_db_path: Optional[Path] = None, model_name: str = None):
        """
        Inicjalizuje system RAG
        """
        self.vector_db_path = vector_db_path
        self.model_name = model_name or 'paraphrase-multilingual-mpnet-base-v2'
        self.index = None
        self.metadata = []
        self.chunks = []
        self.model = None
        self.embedding_dim = None
        
        # Ładowanie modelu embeddingów
        self._load_embedding_model()
        
        # Ładowanie bazy danych
        if vector_db_path:
            self.load_vector_database(vector_db_path)
    
    def _load_embedding_model(self):
        """Ładuje model do embeddingów"""
        logger.info(f"🔄 Ładuję model embeddingów: {self.model_name}")
        try:
            self.model = SentenceTransformer(self.model_name)
            # Test embedding dimension
            test_embedding = self.model.encode(["test"])
            self.embedding_dim = test_embedding.shape[1]
            logger.info(f"✅ Model załadowany. Wymiar: {self.embedding_dim}")
        except Exception as e:
            logger.error(f"❌ Błąd ładowania modelu: {e}")
            raise
    
    def load_vector_database(self, db_path: Path):
        """Ładuje wektorową bazę danych"""
        logger.info(f"📂 Ładuję bazę danych z: {db_path}")
        
        # Szukaj plików
        faiss_file = None
        metadata_file = None
        
        for suffix in ['.faiss', '.index']:
            potential_file = db_path.with_suffix(suffix)
            if potential_file.exists():
                faiss_file = potential_file
                break
        
        for suffix in ['.metadata.pkl', '.pkl']:
            potential_file = db_path.with_suffix(suffix)
            if potential_file.exists():
                metadata_file = potential_file
                break
        
        if not faiss_file or not metadata_file:
            # Spróbuj znaleźć w folderze
            folder = db_path if db_path.is_dir() else db_path.parent
            faiss_files = list(folder.glob("*.faiss"))
            pkl_files = list(folder.glob("*.pkl"))
            
            if faiss_files:
                faiss_file = faiss_files[0]
            if pkl_files:
                metadata_file = pkl_files[0]
        
        if not faiss_file:
            raise FileNotFoundError(f"Nie znaleziono pliku FAISS w: {db_path}")
        
        if not metadata_file:
            raise FileNotFoundError(f"Nie znaleziono pliku metadanych w: {db_path}")
        
        # Ładuj indeks FAISS
        logger.info(f"📊 Ładuję indeks FAISS: {faiss_file.name}")
        self.index = faiss.read_index(str(faiss_file))
        
        # Ładuj metadane
        logger.info(f"📋 Ładuję metadane: {metadata_file.name}")
        with open(metadata_file, 'rb') as f:
            metadata_data = pickle.load(f)
        
        self.chunks = metadata_data.get('chunks', [])
        self.metadata = metadata_data.get('metadata', [])
        
        logger.info(f"✅ Załadowano bazę: {self.index.ntotal} wektorów, {len(self.chunks)} chunk-ów")
        
        return True
    
    def _extract_models_from_query(self, query: str) -> List[str]:
        """Wydobywa modele BMW z pytania"""
        query_lower = query.lower()
        models_found = []
        
        # Lista modeli do wykrywania
        bmw_models = {
            'x1': 'X1', 'x2': 'X2', 'x3': 'X3', 'x4': 'X4', 'x5': 'X5', 
            'x6': 'X6', 'x7': 'X7', 'xm': 'XM',
            'i3': 'i3', 'i4': 'i4', 'i5': 'i5', 'i7': 'i7', 'i8': 'i8', 
            'ix': 'iX', 'ix3': 'iX3', 'ix1': 'iX1',
            'm2': 'M2', 'm3': 'M3', 'm4': 'M4', 'm5': 'M5', 'm8': 'M8',
            'm135': 'M135', 'm235': 'M235', 'm340': 'M340',
            'z4': 'Z4', '2 series': '2', '1 series': '1', '4 series': '4',
            'serii 3': '3', 'seria 3': '3', 'serii 5': '5', 'seria 5': '5',
            'serii 7': '7', 'seria 7': '7', 'serii 8': '8', 'seria 8': '8'
        }
        
        for model_key, model_code in bmw_models.items():
            if model_key in query_lower:
                # Unikaj duplikatów
                if model_code not in models_found:
                    models_found.append(model_code)
        
        # Specjalne przypadki
        if 'serii' in query_lower or 'seria' in query_lower:
            # Wykryj numer serii
            series_match = re.search(r'seri[ia]\s*(\d+)', query_lower)
            if series_match:
                series_num = series_match.group(1)
                if series_num not in models_found:
                    models_found.append(series_num)
        
        return models_found
    
    def _chunk_matches_models(self, chunk_models: List[str], target_models: List[str]) -> bool:
        """Sprawdza czy chunk pasuje do docelowych modeli"""
        if not target_models or not chunk_models:
            return False
        
        chunk_models_upper = [str(m).upper().strip() for m in chunk_models]
        
        for target_model in target_models:
            target_model_upper = target_model.upper().strip()
            
            # Proste dopasowanie
            if target_model_upper in chunk_models_upper:
                return True
            
            # Dopasowanie dla serii (np. "3" w "330i" lub "M3")
            if target_model_upper in ['3', '5', '7', '8']:
                for chunk_model in chunk_models_upper:
                    # Sprawdź czy numer serii jest w nazwie modelu
                    if (target_model_upper == chunk_model or  # "3" == "3"
                        target_model_upper in chunk_model):   # "3" w "330i"
                        return True
            
            # Dopasowanie dla iX (case-insensitive)
            if target_model_upper == 'IX':
                if 'IX' in chunk_models_upper or 'iX' in chunk_models_upper:
                    return True
        
        return False
    
    def _clean_chunk_text(self, text: str) -> str:
        """Czyści tekst chunka - usuwa śmieci z początku"""
        if not text:
            return text
        
        # 1. Usuń znaki interpunkcyjne z początku
        text = re.sub(r'^[,\-;:\s\.…]+', '', text)
        
        # 2. Znajdź pierwszą literę/digit jeśli wciąż zaczyna się źle
        if text and text[0] in ',.;:!?-…':
            # Znajdź pierwszą sensowną pozycję
            for i, char in enumerate(text):
                if char.isalnum():
                    text = text[i:]
                    break
        
        # 3. Usuń marketingowe frazy
        marketing_phrases = [
            r'^dowiedz się więcej\s*[:;]?\s*',
            r'^skonfiguruj\s*[:;]?\s*',
            r'^wi[aą]cej informacji\s*[:;]?\s*',
            r'^sprawd[zź]\s*[:;]?\s*',
            r'^przejd[zź]\s*[:;]?\s*',
            r'^odkryj\s*[:;]?\s*'
        ]
        
        for phrase in marketing_phrases:
            text = re.sub(phrase, '', text, flags=re.IGNORECASE)
        
        return text.strip()
    
    def _extract_specific_info(self, question: str, context: str) -> Optional[str]:
        """Wyodrębnia konkretne informacje z kontekstu na podstawie pytania"""
        question_lower = question.lower()
        lines = [line.strip() for line in context.split('\n') if line.strip()]
        
        # 1. DLA X5 MOC
        if 'x5' in question_lower and ('moc' in question_lower or 'silnik' in question_lower):
            for line in lines:
                if 'x5' in line.lower():
                    # Szukaj: "250 kW (340 KM)"
                    match = re.search(r'(\d+[.,]?\d*)\s*kW\s*\((\d+[.,]?\d*)\s*KM\)', line, re.IGNORECASE)
                    if match:
                        return f"BMW X5 ma moc {match.group(1)} kW ({match.group(2)} KM)"
                    
                    # Szukaj: "moc: 250 kW"
                    match = re.search(r'moc[:\s]+(\d+[.,]?\d*)\s*kW', line, re.IGNORECASE)
                    if match:
                        return f"BMW X5: moc {match.group(1)} kW"
                    
                    # Szukaj: "340 KM"
                    match = re.search(r'(\d+[.,]?\d*)\s*KM', line, re.IGNORECASE)
                    if match:
                        try:
                            power = float(match.group(1).replace(',', '.'))
                            if power > 100:  # Sensowna moc
                                return f"BMW X5: {match.group(1)} KM"
                        except:
                            pass
        
        # 2. DLA i4 PRZYSPIESZENIE
        elif 'i4' in question_lower and ('przyspieszenie' in question_lower or '0-100' in question_lower):
            for line in lines:
                if 'i4' in line.lower():
                    # Szukaj: "3.7 s" lub "3,7 sekundy"
                    match = re.search(r'(\d+[.,]\d+)\s*(?:s|sekund)', line, re.IGNORECASE)
                    if match:
                        return f"BMW i4 przyspiesza 0-100 km/h w {match.group(1)} sekundy"
        
        # 3. DLA X3 CENA
        elif 'x3' in question_lower and ('cena' in question_lower or 'koszt' in question_lower):
            for line in lines:
                if 'x3' in line.lower() and 'zł' in line.lower():
                    # Wyodrębnij cenę
                    price_match = re.search(r'(\d[\d\s]*[.,]?\d*)\s*zł', line, re.IGNORECASE)
                    if price_match:
                        price = price_match.group(1).replace(' ', '')
                        return f"BMW X3 kosztuje od {price} zł"
        
        # 4. DLA iX ZASIĘG
        elif 'ix' in question_lower and 'zasięg' in question_lower:
            for line in lines:
                if 'ix' in line.lower() and 'km' in line.lower():
                    # Szukaj: "zasięg: 500 km"
                    match = re.search(r'zasi[ae]g[:\s]+(\d+[.,]?\d*)\s*km', line, re.IGNORECASE)
                    if match:
                        return f"BMW iX ma zasięg do {match.group(1)} km"
                    
                    # Szukaj: "do 500 km"
                    match = re.search(r'do\s+(\d+[.,]?\d*)\s*km', line, re.IGNORECASE)
                    if match:
                        return f"BMW iX: zasięg do {match.group(1)} km"
        
        # 5. DLA X1 HYBRYDA
        elif 'x1' in question_lower and 'hybryd' in question_lower:
            for line in lines:
                if 'x1' in line.lower() and ('hybryd' in line.lower() or 'elektr' in line.lower() or 'plug' in line.lower()):
                    if 'tak' in line.lower() or 'dostępn' in line.lower() or 'jest' in line.lower():
                        return "BMW X1 jest dostępne jako hybryda plug-in"
                    elif 'nie' in line.lower():
                        return "BMW X1 nie jest dostępne jako hybryda"
        
        # 6. DLA SERII 3 CENA
        elif ('serii 3' in question_lower or 'seria 3' in question_lower) and ('cena' in question_lower or 'koszt' in question_lower):
            for line in lines:
                line_lower = line.lower()
                if ('3' in line_lower or 'trójki' in line_lower) and 'zł' in line_lower:
                    # Wyodrębnij cenę
                    price_match = re.search(r'(\d[\d\s]*[.,]?\d*)\s*zł', line, re.IGNORECASE)
                    if price_match:
                        price = price_match.group(1).replace(' ', '')
                        return f"BMW serii 3 kosztuje od {price} zł"
        
        # 7. DLA M3 vs SERIA 3
        elif 'm3' in question_lower and ('różni' in question_lower or 'różnica' in question_lower or 'vs' in question_lower):
            m3_info = []
            series3_info = []
            
            for line in lines:
                line_lower = line.lower()
                if 'm3' in line_lower:
                    # Szukaj specyfikacji M3
                    if any(word in line_lower for word in ['moc', 'silnik', 'km', 'kw', 'przyspieszenie']):
                        m3_info.append(line.strip())
                elif ('serii 3' in line_lower or 'seria 3' in line_lower or '330' in line_lower):
                    # Szukaj specyfikacji serii 3
                    if any(word in line_lower for word in ['moc', 'silnik', 'km', 'kw', 'przyspieszenie']):
                        series3_info.append(line.strip())
            
            if m3_info or series3_info:
                response = "Porównanie BMW M3 vs seria 3:\n\n"
                if m3_info:
                    response += "M3:\n"
                    for info in m3_info[:2]:
                        response += f"- {info}\n"
                if series3_info:
                    response += "\nSeria 3:\n"
                    for info in series3_info[:2]:
                        response += f"- {info}\n"
                return response
        
        # 8. DLA MODELE ELEKTRYCZNE
        elif 'elektryczn' in question_lower and 'model' in question_lower:
            electric_models = []
            for line in lines:
                line_lower = line.lower()
                if any(model in line_lower for model in ['i3', 'i4', 'i5', 'i7', 'i8', 'ix', 'ix3', 'ix1']):
                    # Wyodrębnij nazwę modelu
                    for model in ['i3', 'i4', 'i5', 'i7', 'i8', 'iX', 'iX3', 'iX1']:
                        if model.lower() in line_lower and model not in electric_models:
                            electric_models.append(model)
            
            if electric_models:
                return f"Modele elektryczne BMW: {', '.join(electric_models)}"
        
        # 9. DLA FINANSOWANIE
        elif 'finansowan' in question_lower or 'leasing' in question_lower or 'rata' in question_lower:
            financing_info = []
            for line in lines:
                if any(word in line.lower() for word in ['leasing', 'finansowanie', 'rata', 'oprocentowanie', 'okres']):
                    financing_info.append(line.strip())
            
            if financing_info:
                return "Opcje finansowania BMW:\n\n" + "\n".join(f"- {info}" for info in financing_info[:3])
        
        # 10. DLA SERWIS
        elif 'serwis' in question_lower or 'warsztat' in question_lower:
            service_info = []
            for line in lines:
                if any(word in line.lower() for word in ['serwis', 'warsztat', 'dealer', 'salon', 'naprawa']):
                    service_info.append(line.strip())
            
            if service_info:
                return "Informacje o serwisie BMW:\n\n" + "\n".join(f"- {info}" for info in service_info[:3])
        
        return None
    
    def query(self, question: str, k: int = 5, filters: Optional[Dict] = None, 
              use_model_filter: bool = True, use_priority: bool = True) -> List[Dict]:
        """
        Wykonuje zapytanie z inteligentnym filtrowaniem po modelach
        
        Args:
            question: Pytanie użytkownika
            k: Liczba wyników do zwrócenia
            filters: Dodatkowe filtry
            use_model_filter: Czy filtrować po wykrytych modelach
            use_priority: Czy używać priorytetów z metadata
        
        Returns:
            Lista wyników z podobieństwami i metadanymi
        """
        logger.info(f"🔍 Przetwarzam zapytanie: '{question}'")
        
        # 1. Wydobyj modele z pytania
        target_models = []
        if use_model_filter:
            target_models = self._extract_models_from_query(question)
            if target_models:
                logger.info(f"   🎯 Wykryte modele w pytaniu: {target_models}")
        
        # 2. Generuj embedding dla pytania
        question_embedding = self.model.encode([question])
        
        # 3. Szukaj więcej wyników niż potrzebujemy (do filtrowania)
        search_k = min(k * 10, self.index.ntotal)  # Szukaj dużo, potem filtruj
        if isinstance(self.index, faiss.IndexFlatIP):
            faiss.normalize_L2(question_embedding)
        
        distances, indices = self.index.search(question_embedding, search_k)
        
        # 4. Filtruj, priorytetyzuj i sortuj wyniki
        scored_results = []
        seen_chunk_ids = set()
        
        for distance, idx in zip(distances[0], indices[0]):
            if idx >= len(self.metadata):
                continue
                
            chunk_meta = self.metadata[idx]
            chunk_id = chunk_meta['id']
            
            # Unikaj duplikatów
            if chunk_id in seen_chunk_ids:
                continue
            seen_chunk_ids.add(chunk_id)
            
            # Zastosuj podstawowe filtry
            skip_chunk = False
            if filters:
                for key, value in filters.items():
                    if key in chunk_meta['metadata']:
                        if chunk_meta['metadata'][key] != value:
                            skip_chunk = True
                            break
            if skip_chunk:
                continue
            
            # OCZYŚĆ TEKST CHUNKA
            cleaned_text = self._clean_chunk_text(chunk_meta['text'])
            if len(cleaned_text) < 30:  # Pomijaj za krótkie po oczyszczeniu
                continue
            
            # Oblicz score dla tego chunka
            score = float(distance)  # Podstawowy score z embedding
            
            chunk_models = chunk_meta['metadata'].get('models', [])
            chunk_has_target_model = False
            
            # Bonus za match modelu
            if target_models and chunk_models:
                if self._chunk_matches_models(chunk_models, target_models):
                    chunk_has_target_model = True
                    score += 0.3  # Duży bonus za pasujący model
            
            # Bonus za priorytet retrieval
            if use_priority:
                priority = chunk_meta['metadata'].get('retrieval_priority', 1)
                if priority >= 4:  # Wysoki priorytet (specyfikacje)
                    score += 0.2
                elif priority >= 3:  # Średni priorytet (modele)
                    score += 0.1
            
            # Bonus za specyfikacje w treści pytania
            question_lower = question.lower()
            chunk_text_lower = cleaned_text.lower()
            
            if any(word in question_lower for word in ['cena', 'koszt', 'zł']):
                if any(word in chunk_text_lower for word in ['zł', 'cena', 'od', 'pln']):
                    score += 0.15
            
            if any(word in question_lower for word in ['moc', 'silnik', 'km', 'koni']):
                if any(word in chunk_text_lower for word in ['km', 'koni', 'moc', 'silnik', 'kw', 'hp']):
                    score += 0.15
            
            if any(word in question_lower for word in ['przyspieszenie', '0-100']):
                if any(word in chunk_text_lower for word in ['0-100', 'przyspiesza', 'sekund']):
                    score += 0.15
            
            if any(word in question_lower for word in ['zasięg', 'zasięgu']):
                if any(word in chunk_text_lower for word in ['km', 'kilometr', 'zasięg']):
                    score += 0.15
            
            # Bonus za krótsze, konkretne chunki
            chunk_length = len(cleaned_text)
            if 100 < chunk_length < 400:  # Optymalny rozmiar
                score += 0.05
            
            scored_results.append({
                'distance': float(distance),
                'score': score,
                'index': idx,
                'chunk_meta': chunk_meta,
                'cleaned_text': cleaned_text,
                'has_target_model': chunk_has_target_model,
                'chunk_length': chunk_length
            })
        
        # 5. Sortuj po score (najwyższy pierwszy)
        scored_results.sort(key=lambda x: x['score'], reverse=True)
        
        # 6. Przygotuj finalne wyniki
        results = []
        for i, result in enumerate(scored_results[:k]):
            chunk_meta = result['chunk_meta']
            
            results.append({
                'rank': i + 1,
                'similarity_score': result['distance'],
                'relevance_score': result['score'],
                'chunk_id': chunk_meta['id'],
                'text': result['cleaned_text'],  # Użyj OCZYSZCZONEGO tekstu!
                'original_text': chunk_meta['text'],  # Zachowaj oryginał
                'metadata': chunk_meta['metadata'],
                'source_info': {
                    'title': chunk_meta['metadata'].get('title', ''),
                    'url': chunk_meta['metadata'].get('source_url', ''),
                    'models': chunk_meta['metadata'].get('models', []),
                    'categories': chunk_meta['metadata'].get('categories', []),
                    'has_target_model': result['has_target_model'],
                    'retrieval_priority': chunk_meta['metadata'].get('retrieval_priority', 1),
                    'chunk_length': result['chunk_length']
                }
            })
        
        # Loguj statystyki
        if results:
            matched_models = sum(1 for r in results if r['source_info']['has_target_model'])
            logger.info(f"✅ Znaleziono {len(results)} wyników ({matched_models} z pasującym modelem)")
            if results[0]['source_info']['has_target_model']:
                logger.info(f"   🎯 Najlepszy wynik: model PASUJE")
        
        return results
    
    def generate_answer(self, question: str, context_results: List[Dict], 
                       max_context_length: int = 2000) -> Dict:
        """
        Generuje odpowiedź na podstawie znalezionych kontekstów
        
        Args:
            question: Pytanie użytkownika
            context_results: Wyniki z query()
            max_context_length: Maksymalna długość kontekstu
            
        Returns:
            Słownik z odpowiedzią i źródłami
        """
        if not context_results:
            return {
                'answer': "Przepraszam, nie znalazłem odpowiednich informacji w bazie danych.",
                'sources': [],
                'confidence': 0.0
            }
        
        # Sortuj wyniki po relevance_score
        sorted_results = sorted(context_results, key=lambda x: x['relevance_score'], reverse=True)
        
        # Przygotuj kontekst z najlepszych wyników
        context_parts = []
        total_length = 0
        sources = []
        
        # Najpierw bierz wyniki z pasującym modelem
        model_matched_results = [r for r in sorted_results if r['source_info']['has_target_model']]
        other_results = [r for r in sorted_results if not r['source_info']['has_target_model']]
        
        # Wybieraj w kolejności: pasujące modele → wysoki priorytet → reszta
        all_results_sorted = model_matched_results + other_results
        
        for result in all_results_sorted:
            chunk_text = result['text']  # Użyj już oczyszczonego tekstu
            
            # Pomijaj bardzo krótkie fragmenty (chyba że mają specyfikacje)
            if len(chunk_text) < 30 and 'zł' not in chunk_text and 'km' not in chunk_text:
                continue
                
            if total_length + len(chunk_text) <= max_context_length:
                context_parts.append(chunk_text)
                total_length += len(chunk_text)
                
                # Dodaj źródło
                source_info = {
                    'title': result['metadata'].get('title', ''),
                    'url': result['metadata'].get('source_url', ''),
                    'similarity': result['similarity_score'],
                    'relevance': result['relevance_score'],
                    'models': result['metadata'].get('models', []),
                    'has_target_model': result['source_info']['has_target_model'],
                    'retrieval_priority': result['metadata'].get('retrieval_priority', 1)
                }
                sources.append(source_info)
            else:
                break
        
        context = "\n\n".join(context_parts)
        
        # Generuj INTELIGENTNĄ odpowiedź
        answer = self._generate_intelligent_answer(question, context, sources)
        
        return {
            'answer': answer,
            'sources': sources,
            'context_length': total_length,
            'results_used': len(context_parts),
            'confidence': np.mean([r['relevance_score'] for r in all_results_sorted[:len(context_parts)]]) if context_parts else 0.0
        }
    
    def _generate_intelligent_answer(self, question: str, context: str, sources: List[Dict]) -> str:
        """
        INTELIGENTNA generacja odpowiedzi z wyodrębnieniem konkretów
        """
        # 1. SPRÓBUJ WYODRĘBNIĆ KONKRETNĄ INFORMACJĘ
        specific_info = self._extract_specific_info(question, context)
        if specific_info:
            return specific_info
        
        # 2. Jeśli nie udało się wyodrębnić konkretów, użyj bardziej ogólnej metody
        question_lower = question.lower()
        lines = [line.strip() for line in context.split('\n') if line.strip()]
        
        # 3. DLA PYTAŃ O CENY - szukaj linii z cenami
        if any(word in question_lower for word in ['cena', 'koszt', 'ile kosztuje']):
            price_lines = []
            for line in lines:
                if 'zł' in line or 'PLN' in line:
                    # Sprawdź czy linia zawiera model z pytania
                    target_models = self._extract_models_from_query(question)
                    if target_models:
                        for model in target_models:
                            if model.lower() in line.lower():
                                price_lines.append(line)
                                break
                    else:
                        price_lines.append(line)
            
            if price_lines:
                unique_prices = []
                seen = set()
                for price in price_lines:
                    if price not in seen:
                        seen.add(price)
                        unique_prices.append(price)
                
                if unique_prices:
                    answer = "Znalezione ceny:\n\n"
                    for i, price in enumerate(unique_prices[:4], 1):
                        answer += f"{i}. {price}\n"
                    return answer
        
        # 4. DLA PYTAŃ O SPECYFIKACJE - szukaj linii z liczbami
        elif any(word in question_lower for word in ['moc', 'silnik', 'przyspieszenie', 'zasięg', 'km', 'kw']):
            spec_lines = []
            for line in lines:
                # Szukaj linii z liczbami (specyfikacje)
                if re.search(r'\d+[.,]?\d*\s*(?:kW|KM|km|s|zł)', line, re.IGNORECASE):
                    # Sprawdź czy linia zawiera model z pytania
                    target_models = self._extract_models_from_query(question)
                    if target_models:
                        for model in target_models:
                            if model.lower() in line.lower():
                                spec_lines.append(line)
                                break
                    else:
                        spec_lines.append(line)
            
            if spec_lines:
                unique_specs = []
                seen = set()
                for spec in spec_lines:
                    if spec not in seen and len(spec) > 15:
                        seen.add(spec)
                        unique_specs.append(spec)
                
                if unique_specs:
                    answer = "Znalezione specyfikacje:\n\n"
                    for i, spec in enumerate(unique_specs[:4], 1):
                        answer += f"{i}. {spec}\n"
                    return answer
        
        # 5. DLA PYTAŃ OGÓLNYCH - wybierz najważniejsze linie
        meaningful_lines = []
        for line in lines:
            # Filtruj marketing i bardzo krótkie linie
            if (len(line) > 30 and 
                not any(marketing in line.lower() for marketing in 
                       ['dowiedz się więcej', 'skonfiguruj', 'więcej informacji',
                        'sprawdź', 'przejdź', 'odkryj', '...'])):
                meaningful_lines.append(line)
        
        if meaningful_lines:
            # Sortuj: najpierw linie z liczbami (specyfikacje)
            prioritized_lines = []
            other_lines = []
            
            for line in meaningful_lines:
                if re.search(r'\d+', line):  # Linia z liczbami
                    prioritized_lines.append(line)
                else:
                    other_lines.append(line)
            
            selected_lines = prioritized_lines[:2] + other_lines[:2]
            
            answer = "Znalezione informacje:\n\n"
            for i, line in enumerate(selected_lines, 1):
                answer += f"{i}. {line[:120]}{'...' if len(line) > 120 else ''}\n"
            return answer
        
        # 6. OSTATECZNY FALLBACK - pierwsze 3 linie z kontekstu
        if lines:
            return "Informacje:\n\n" + "\n".join(f"- {line[:100]}..." if len(line) > 100 else f"- {line}" 
                                                for line in lines[:3])
        
        # 7. BRAK INFORMACJI
        return "Nie znaleziono konkretnych informacji na to pytanie w dostępnych danych."
    
    def get_database_info(self) -> Dict:
        """Zwraca informacje o bazie danych"""
        return {
            'total_vectors': self.index.ntotal if self.index else 0,
            'total_chunks': len(self.chunks),
            'embedding_dim': self.embedding_dim,
            'index_type': str(type(self.index).__name__) if self.index else 'None',
            'model_name': self.model_name,
            'loaded_at': datetime.now().isoformat()
        }

def find_latest_vector_db():
    """Znajduje najnowszą bazę wektorową"""
    output_base = Path(r"C:\Users\hellb\Documents\Chatbot_BMW\RAG\output")
    
    # Szukaj plików .faiss
    faiss_files = []
    for folder in output_base.iterdir():
        if folder.is_dir():
            for file in folder.glob("*.faiss"):
                faiss_files.append((folder, file))
    
    if not faiss_files:
        logger.error("❌ Nie znaleziono baz danych FAISS")
        return None
    
    # Sortuj po czasie modyfikacji
    faiss_files.sort(key=lambda x: x[1].stat().st_mtime, reverse=True)
    
    latest_folder, latest_file = faiss_files[0]
    
    # Znajdź odpowiadający plik metadanych
    metadata_file = None
    for suffix in ['.metadata.pkl', '.pkl']:
        potential_file = latest_file.with_suffix(suffix)
        if potential_file.exists():
            metadata_file = potential_file
            break
    
    if not metadata_file:
        # Szukaj w folderze
        pkl_files = list(latest_folder.glob("*.pkl"))
        if pkl_files:
            metadata_file = pkl_files[0]
    
    logger.info(f"📁 Znaleziono bazę: {latest_file.name}")
    logger.info(f"   Folder: {latest_folder.name}")
    logger.info(f"   Metadata: {metadata_file.name if metadata_file else 'Nie znaleziono'}")
    
    return latest_file

def test_rag_system():
    """Testuje system RAG"""
    print("=" * 70)
    print("🧪 TEST SYSTEMU RAG - BMW CHATBOT (INTELIGENTNY v2)")
    print("=" * 70)
    
    # Znajdź bazę danych
    db_file = find_latest_vector_db()
    if not db_file:
        return
    
    # Utwórz system RAG
    print(f"\n🚀 Inicjalizuję system RAG z bazą: {db_file.name}")
    
    try:
        rag = RAGSystem(vector_db_path=db_file)
    except Exception as e:
        print(f"❌ Błąd inicjalizacji: {e}")
        return
    
    # Pokaż informacje o bazie
    db_info = rag.get_database_info()
    print(f"\n📊 INFORMACJE O BAZIE:")
    print(f"   Wektory: {db_info['total_vectors']}")
    print(f"   Chunk-i: {db_info['total_chunks']}")
    print(f"   Wymiar: {db_info['embedding_dim']}")
    print(f"   Model: {db_info['model_name']}")
    
    # Testowe pytania - SKRÓCONA LISTA DLA TESTOWANIA
    test_questions = [
        "Jaka jest moc silnika w BMW X5?",
        "Ile wynosi przyspieszenie 0-100 km/h w BMW i4?",
        "Ile kosztuje BMW X3?",
        "Jaki jest zasięg BMW iX?",
        "Czy BMW X1 jest dostępne jako hybryda?",
        "Czym się różni BMW M3 od zwykłej serii 3?"
    ]
    
    print(f"\n🎯 TESTUJĘ {len(test_questions)} PYTAŃ:")
    print("=" * 70)
    
    for i, question in enumerate(test_questions, 1):
        print(f"\n{i}. ❓ PYTANIE: {question}")
        
        try:
            # Wyszukaj z filtrowaniem po modelach
            results = rag.query(question, k=3, use_model_filter=True, use_priority=True)
            
            if results:
                # Wygeneruj odpowiedź
                answer_data = rag.generate_answer(question, results)
                
                # Pokaż odpowiedź
                print(f"   ✅ ODPOWIEDŹ: {answer_data['answer'][:200]}...")
                print(f"   📊 Trafność: {answer_data['confidence']:.3f}")
                print(f"   📚 Źródła: {len(answer_data['sources'])}")
                
                # Pokaż statystyki match modelu
                matched_sources = sum(1 for s in answer_data['sources'] if s.get('has_target_model', False))
                print(f"   🎯 Źródła z pasującym modelem: {matched_sources}")
                
                # Pokaż najlepsze źródło
                if answer_data['sources']:
                    best_source = answer_data['sources'][0]
                    print(f"   🏆 Najlepsze źródło: {best_source['title'][:40]}...")
                    if best_source['models']:
                        print(f"      Modele: {', '.join(best_source['models'][:3])}")
            else:
                print(f"   ⚠️  Brak wyników")
                
        except Exception as e:
            print(f"   ❌ Błąd: {e}")
            import traceback
            traceback.print_exc()
    
    # Interaktywny tryb
    print(f"\n" + "=" * 70)
    print("💬 TRYB INTERAKTYWNY")
    print("=" * 70)
    
    while True:
        question = input("\n🎯 Twoje pytanie o BMW (lub 'exit' aby zakończyć): ")
        
        if question.lower() in ['exit', 'quit', 'q']:
            print("👋 Do widzenia!")
            break
        
        if not question.strip():
            continue
        
        print(f"\n🔍 Szukam odpowiedzi...")
        
        try:
            # Wyszukaj z zaawansowanym filtrowaniem
            results = rag.query(
                question, 
                k=5, 
                use_model_filter=True,
                use_priority=True
            )
            
            if results:
                answer_data = rag.generate_answer(question, results)
                
                print(f"\n✅ ODPOWIEDŹ:")
                print(f"   {answer_data['answer']}")
                
                print(f"\n📊 STATYSTYKI:")
                print(f"   Trafność ogólna: {answer_data['confidence']:.3f}")
                print(f"   Użyte źródła: {answer_data['results_used']}")
                print(f"   Długość kontekstu: {answer_data['context_length']} znaków")
                
                # Liczba źródeł z pasującym modelem
                matched_sources = sum(1 for s in answer_data['sources'] if s.get('has_target_model', False))
                print(f"   Źródeł z pasującym modelem: {matched_sources}/{len(answer_data['sources'])}")
                
                # Debug: pokaż wykryte modele z pytania
                detected_models = rag._extract_models_from_query(question)
                if detected_models:
                    print(f"\n🔍 Wykryte modele w pytaniu: {detected_models}")
            else:
                print(f"❌ Nie znalazłem odpowiednich informacji.")
                
        except Exception as e:
            print(f"❌ Błąd: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    test_rag_system()