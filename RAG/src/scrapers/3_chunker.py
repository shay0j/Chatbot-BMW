from langchain_text_splitters import RecursiveCharacterTextSplitter
import json
import pandas as pd
from pathlib import Path
from datetime import datetime
import re
import hashlib
from typing import List, Dict, Any, Optional
from collections import Counter

class RAGPreprocessor:
    """Przygotowuje dane dla RAG z inteligentnym chunkowaniem i post-processingiem dla BMW"""
    
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 100):
        """
        Inicjalizuje preprocessor z optymalnymi ustawieniami dla danych BMW
        
        Args:
            chunk_size: Rozmiar chunka w znakach (500 ≈ 125-150 tokenów)
            chunk_overlap: Przekrycie między chunkami
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Lista złych fraz do filtrowania
        self.bad_phrases = [
            'lorem ipsum', 'skip to main content', 'dummy text',
            'example text', 'placeholder', 'breadcrumb', 'navbar',
            'footer', 'header', 'cookie policy', 'privacy policy',
            'terms of use', 'regulamin', 'polityka prywatności',
            'cookies', 'menu', 'nawigacja'
        ]
        
        # Optymalne separatory dla treści BMW
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=[
                "\n\n## ",     # Nagłówki H2
                "\n\n### ",    # Nagłówki H3
                "\n\n• ",      # Listy z punktorami
                "\n\n- ",      # Listy z myślnikami
                "\n\n",        # Podwójna nowa linia (nowy akapit)
                ".\n",         # Koniec zdania + nowa linia
                ". ",          # Kropka + spacja
                "\n",          # Nowa linia
                "; ",          # Koniec frazy
                ", ",          # Przecinek
                " ",           # Spacja
                ""             # Ostateczność
            ],
            keep_separator=False  # NIE zachowuj separatora na początku chunka!
        )
    
    # ==================== METODY CZYSZCZENIA TEKSTU ====================
    
    def _clean_text_before_chunking(self, text: str) -> str:
        """Czyści tekst przed chunkowaniem - naprawia typowe problemy scrapingu"""
        if not text:
            return ""
            
        # 1. Napraw zlepione słowa z liczbami
        text = re.sub(r'([a-zA-Z])(\d)', r'\1 \2', text)  # za2500 -> za 2500
        text = re.sub(r'(\d)([a-zA-Z])', r'\1 \2', text)  # 250KM -> 250 KM
        
        # 2. Napraw zlepione słowa z CamelCase
        text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)  # BMWX3 -> BMW X3
        
        # 3. Zamień nbsp i inne specjalne znaki na zwykłe spacje
        text = text.replace('\xa0', ' ')
        text = text.replace('\u200b', ' ')  # zero-width space
        text = text.replace('\r', ' ')
        
        # 4. Usuń nadmiarowe białe znaki
        text = ' '.join(text.split())
        
        # 5. Zapewnij spacje po kropkach (jeśli brak)
        text = re.sub(r'\.(?=[a-zA-Z])', '. ', text)
        
        # 6. Usuń ciągi powtarzających się znaków interpunkcyjnych
        text = re.sub(r'([.!?])\1+', r'\1', text)
        
        return text.strip()
    
    def _clean_chunk_start(self, chunk_text: str) -> str:
        """Czyści początek chunka - usuwa interpunkcję na początku"""
        if not chunk_text:
            return ""
            
        # Usuń początkowe przecinki, kropki, średniki, spacje
        while chunk_text and chunk_text[0] in ',.;:!?-\t ':
            chunk_text = chunk_text[1:]
            
        # Jeśli chunek zaczyna się małą literą, a to długi tekst, zmień na dużą
        if (chunk_text and chunk_text[0].islower() and 
            len(chunk_text) > 50 and 
            not any(chunk_text[:3].lower() in prefix for prefix in ['bmw', 'x1', 'x3', 'x5', 'i3', 'i4', 'i7', 'ix'])):
            chunk_text = chunk_text[0].upper() + chunk_text[1:]
            
        return chunk_text.strip()
    
    def _clean_chunk_content(self, text: str) -> str:
        """Dodatkowe czyszczenie treści chunka"""
        if not text:
            return ""
        
        # Napraw pozostałe zlepione słowa
        text = re.sub(r'([a-zA-Z])(\d)', r'\1 \2', text)
        text = re.sub(r'(\d)([a-zA-Z])', r'\1 \2', text)
        
        # Usuń nadmiarowe spacje
        text = ' '.join(text.split())
        
        # Zapewnij spacje po kropkach
        text = re.sub(r'\.(?=[a-zA-Z])', '. ', text)
        
        return text.strip()
    
    # ==================== METODY DETEKCJI I WALIDACJI ====================
    
    def _is_specification_line(self, text: str) -> bool:
        """Czy linia zawiera specyfikację techniczną?"""
        spec_indicators = [
            r'\b\d+(?:[.,]\d+)?\s*zł\b',
            r'\b\d+(?:[.,]\d+)?\s*(?:kW|KM)\b',
            r'\b\d+[.,]\d+\s*s\b.*0-100',
            r'\b\d+\s*km\b.*zasięg',
            r'\b\d+(?:[.,]\d+)?\s*l/100km\b',
            r'\bV\d+\b',  # V6, V8
            r'\b\d+\s*Nm\b',  # Moment obrotowy
            r'przyspieszenie.*\d+[.,]\d+\s*s',
            r'\bcena\s*[:=]',
            r'\bmoc\s*[:=]',
            r'\bsilnik\s*[:=]',
            r'\bzasięg\s*[:=]',
            r'\bspalanie\s*[:=]',
        ]
        
        text_lower = text.lower()
        return any(re.search(pattern, text_lower) for pattern in spec_indicators)
    
    def _is_bad_chunk(self, text: str) -> bool:
        """Czy chunek jest bezużyteczny?"""
        if not text or len(text.strip()) < 10:
            return True
            
        text_lower = text.lower()
        
        # 1. Odrzuć placeholdery
        if any(phrase in text_lower for phrase in self.bad_phrases):
            return True
        
        # 2. Odrzuć za krótkie bez wartości (chyba że to specyfikacja)
        if len(text) < 30 and not self._has_valuable_content(text):
            return True
        
        # 3. Odrzuć tylko listy modeli bez kontekstu
        if self._is_just_model_list(text):
            return True
        
        # 4. Odrzuć chunki które są głównie cyframi/symbolami
        if len(text) > 50:
            letter_count = sum(1 for c in text if c.isalpha())
            if letter_count / len(text) < 0.3:  # Mniej niż 30% liter
                return True
        
        # 5. Odrzuć chunki z dużą ilością powtórzeń
        words = text.split()
        if len(words) > 10:
            unique_words = set(words)
            if len(unique_words) / len(words) < 0.4:  # Za dużo powtórzeń
                return True
        
        return False
    
    def _has_valuable_content(self, text: str) -> bool:
        """Czy krótki chunek ma wartościową treść?"""
        # Sprawdź czy zawiera specyfikacje
        if self._is_specification_line(text):
            return True
        
        # Sprawdź czy zawiera modele
        if re.search(r'\b(?:bmw\s+)?[xmi]\d+\b', text, re.IGNORECASE):
            return True
        
        # Sprawdź czy zawiera ceny
        if re.search(r'\b\d+(?:\s?\d{3})*[.,]\d{2}\s*zł\b', text, re.IGNORECASE):
            return True
        
        # Sprawdź czy zawiera kluczowe słowa
        keywords = ['cena', 'silnik', 'moc', 'przyspieszenie', 'zasięg', 'leasing', 'finansowanie']
        if any(keyword in text.lower() for keyword in keywords):
            return True
        
        return False
    
    def _is_just_model_list(self, text: str) -> bool:
        """Czy to tylko lista modeli bez kontekstu?"""
        lines = [l.strip() for l in text.split('\n') if l.strip()]
        
        if len(lines) > 3:
            model_line_count = 0
            for line in lines:
                # Sprawdź czy linia to tylko nazwa modelu
                if (re.match(r'^(BMW\s+)?[XMI]\d+', line, re.IGNORECASE) or
                    re.match(r'^Seria\s+\d+', line, re.IGNORECASE) or
                    re.match(r'^\d+\s*[-–]\s*', line)):
                    model_line_count += 1
            
            # Jeśli ponad 70% linii to tylko nazwy modeli
            if model_line_count / len(lines) > 0.7:
                return True
        
        return False
    
    # ==================== METODY PODZIAŁU TEKSTU ====================
    
    def _estimate_tokens(self, text: str) -> int:
        """Szacuje liczbę tokenów dla tekstu"""
        return len(text) // 3  # Konserwatywne oszacowanie dla polskiego
    
    def _split_by_sentences(self, text: str, max_chars: int = 600) -> List[str]:
        """Dzieli tekst na zdania i łączy je w chunki nie przekraczające max_chars"""
        if not text or len(text) < 50:
            return [text] if text else []
        
        # Podziel na zdania (zachowując znaki interpunkcyjne)
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
                
            # Jeśli dodanie tego zdania przekroczy limit i mamy już coś w chunk-u
            if current_chunk and len(current_chunk) + len(sentence) + 1 > max_chars:
                chunks.append(current_chunk)
                current_chunk = sentence
            else:
                current_chunk = f"{current_chunk} {sentence}".strip() if current_chunk else sentence
        
        if current_chunk:
            chunks.append(current_chunk)
            
        return chunks
    
    def _split_very_large_chunk(self, text: str, metadata: Dict) -> List[Dict]:
        """Dzieli bardzo duże chunki na mniejsze"""
        if len(text) <= 800:
            return [{'text': text, 'metadata': metadata}]
        
        chunks = []
        
        # 1. Spróbuj podzielić na akapity
        paragraphs = re.split(r'\n\s*\n', text)
        
        current_chunk = ""
        
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
                
            # Jeśli dodanie tego akapitu przekroczy 800 znaków
            if current_chunk and len(current_chunk) + len(para) > 800:
                if current_chunk:
                    # Przygotuj nowe metadata dla podzielonego chunka
                    new_metadata = metadata.copy()
                    new_metadata['chunk_char_count'] = len(current_chunk)
                    new_metadata['chunk_token_estimate'] = self._estimate_tokens(current_chunk)
                    new_metadata['is_split_chunk'] = True
                    
                    chunks.append({
                        'id': f"{metadata.get('unique_id', 'chunk')}_part{len(chunks)}",
                        'text': self._clean_chunk_start(current_chunk),
                        'metadata': new_metadata
                    })
                current_chunk = para
            else:
                current_chunk = f"{current_chunk}\n\n{para}".strip() if current_chunk else para
        
        if current_chunk:
            new_metadata = metadata.copy()
            new_metadata['chunk_char_count'] = len(current_chunk)
            new_metadata['chunk_token_estimate'] = self._estimate_tokens(current_chunk)
            new_metadata['is_split_chunk'] = True
            
            chunks.append({
                'id': f"{metadata.get('unique_id', 'chunk')}_part{len(chunks)}",
                'text': self._clean_chunk_start(current_chunk),
                'metadata': new_metadata
            })
        
        return chunks
    
    # ==================== METODY POST-PROCESSINGU ====================
    
    def _normalize_for_deduplication(self, text: str) -> str:
        """Normalizuje tekst do wykrywania duplikatów"""
        if not text:
            return ""
        
        # 1. Zamień na małe litery
        text = text.lower()
        
        # 2. Usuń wszystkie białe znaki (znormalizuj spacje)
        text = re.sub(r'\s+', ' ', text)
        
        # 3. Usuń interpunkcję
        text = re.sub(r'[^\w\s]', '', text)
        
        # 4. Usuń słowa-krzaki (krótkie słowa)
        words = text.split()
        words = [w for w in words if len(w) > 2]
        
        # 5. Posortuj słowa (dla niezależności od kolejności)
        words.sort()
        
        return ' '.join(words)
    
    def _chunks_can_be_merged(self, chunk1: Dict, chunk2: Dict) -> bool:
        """Czy dwa chunki można połączyć?"""
        # 1. Sprawdź czy dotyczą tych samych modeli
        models1 = set(chunk1['metadata'].get('models', []))
        models2 = set(chunk2['metadata'].get('models', []))
        
        if models1 and models2 and not models1.intersection(models2):
            return False
        
        # 2. Sprawdź czy łączna długość nie przekroczy limitu
        total_length = len(chunk1['text']) + len(chunk2['text'])
        if total_length > 600:
            return False
        
        # 3. Sprawdź czy tagi są podobne
        tags1 = set(chunk1['metadata'].get('content_tags', []))
        tags2 = set(chunk2['metadata'].get('content_tags', []))
        
        if tags1 and tags2 and not tags1.intersection(tags2):
            return False
        
        # 4. Sprawdź czy chunki dotyczą podobnych tematów
        text1_lower = chunk1['text'].lower()
        text2_lower = chunk2['text'].lower()
        
        # Wspólne słowa kluczowe
        common_keywords = ['cena', 'silnik', 'moc', 'przyspieszenie', 'zasięg', 'leasing', 'finansowanie', 'serwis']
        has_common_keyword = any(
            keyword in text1_lower and keyword in text2_lower 
            for keyword in common_keywords
        )
        
        if not has_common_keyword and total_length > 300:
            # Jeśli nie mają wspólnych słów kluczowych, lepiej nie łączyć
            return False
        
        return True
    
    def post_process_chunks(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Czyści i optymalizuje chunki po podziale"""
        if not chunks:
            return []
        
        processed_chunks = []
        seen_hashes = set()
        
        i = 0
        while i < len(chunks):
            chunk = chunks[i]
            text = chunk['text']
            
            # 1. Odrzuć złe chunki
            if self._is_bad_chunk(text):
                i += 1
                continue
            
            # 2. Sprawdź duplikaty
            text_hash = hash(self._normalize_for_deduplication(text))
            if text_hash in seen_hashes:
                i += 1
                continue
            seen_hashes.add(text_hash)
            
            # 3. Podziel bardzo duże chunki
            if len(text) > 1000:
                sub_chunks = self._split_very_large_chunk(text, chunk['metadata'])
                for sub_chunk in sub_chunks:
                    sub_text = sub_chunk['text']
                    if not self._is_bad_chunk(sub_text):
                        sub_hash = hash(self._normalize_for_deduplication(sub_text))
                        if sub_hash not in seen_hashes:
                            processed_chunks.append(sub_chunk)
                            seen_hashes.add(sub_hash)
                i += 1
                continue
            
            # 4. Spróbuj połączyć z poprzednim jeśli krótki
            if len(text) < 80 and processed_chunks:
                last_chunk = processed_chunks[-1]
                if self._chunks_can_be_merged(last_chunk, chunk):
                    # Połącz chunki
                    last_chunk['text'] += " " + text
                    # Zaktualizuj metadata
                    last_chunk['metadata']['chunk_char_count'] = len(last_chunk['text'])
                    last_chunk['metadata']['chunk_token_estimate'] = self._estimate_tokens(last_chunk['text'])
                    # Zachowaj unię tagów
                    tags1 = set(last_chunk['metadata'].get('content_tags', []))
                    tags2 = set(chunk['metadata'].get('content_tags', []))
                    if tags1 or tags2:
                        last_chunk['metadata']['content_tags'] = list(tags1.union(tags2))
                    
                    i += 1
                    continue
            
            # 5. Dodaj normalny chunek (z oczyszczeniem)
            chunk['text'] = self._clean_chunk_start(text)
            chunk['text'] = self._clean_chunk_content(chunk['text'])
            processed_chunks.append(chunk)
            i += 1
        
        return processed_chunks
    
    # ==================== METODY TWORZENIA METADATA ====================
    
    def _create_rich_metadata(self, item: Dict[str, Any], chunk_index: int, 
                             total_chunks: int, chunk_text: str) -> Dict[str, Any]:
        """
        Tworzy bogate metadata z priorytetami dla retrieval
        """
        metadata = {
            'unique_id': item.get('unique_id', f"doc_{abs(hash(item.get('url', '')))}"),
            'source_url': item.get('url', ''),
            'title': item.get('title', '')[:150],
            'categories': item.get('categories', []),
            'detected_models': item.get('detected_models', []),
            'is_model_page': item.get('is_model_page', False),
            'original_word_count': item.get('word_count', 0),
            'scraped_at': item.get('scraped_at', ''),
            'normalized_at': item.get('normalized_at', ''),
            
            # Informacje o chunk-u
            'chunk_index': chunk_index,
            'total_chunks': total_chunks,
            'chunk_char_count': len(chunk_text),
            'chunk_token_estimate': self._estimate_tokens(chunk_text),
            'chunk_has_specs': self._is_specification_line(chunk_text),
            
            # Priorytet dla retrieval
            'retrieval_priority': self._calculate_priority(item, chunk_text),
        }
        
        # Wzbogać o specyfikacje
        specs = item.get('specifications', {})
        if specs:
            # Modele z specifications
            if 'model_info' in specs:
                models = specs['model_info'].get('models', [])
                if models:
                    metadata['models'] = models
                    metadata['primary_model'] = models[0] if models else ''
                
                if 'model_from_url' in specs['model_info']:
                    metadata['model_from_url'] = specs['model_info']['model_from_url']
            
            # Silnik - skompresowane do stringa dla łatwiejszego filtrowania
            if 'engine' in specs and specs['engine']:
                engine_str_parts = []
                for key, value in specs['engine'].items():
                    if value and value != 0:
                        engine_str_parts.append(f"{key}:{value}")
                if engine_str_parts:
                    metadata['engine_specs_compact'] = "|".join(engine_str_parts)
            
            # Ceny - uproszczone
            if 'prices' in specs and specs['prices']:
                prices = []
                for price in specs['prices'][:3]:  # Max 3 ceny
                    if isinstance(price, dict):
                        amount = price.get('amount', 0)
                        if amount > 0:
                            prices.append(amount)
                
                if prices:
                    metadata['min_price'] = min(prices)
                    metadata['max_price'] = max(prices)
                    metadata['has_prices'] = True
            
            # Dodatkowe flagi
            metadata['has_engine_specs'] = 'engine' in specs and bool(specs['engine'])
            metadata['has_prices'] = 'prices' in specs and bool(specs['prices'])
        
        # DODANE: Jeśli nie ma models z specifications, weź z detected_models
        if item.get('detected_models') and not metadata.get('models'):
            metadata['models'] = item.get('detected_models', [])
        
        return metadata
    
    def _calculate_priority(self, item: Dict[str, Any], chunk_text: str) -> int:
        """Oblicza priorytet retrieval dla chunka (1-5, 5 najwyższy)"""
        priority = 1
        
        # Priorytet 5: Chunk-i z cenami
        if self._is_specification_line(chunk_text) and any(word in chunk_text.lower() for word in ['zł', 'cena', 'od']):
            priority = 5
        
        # Priorytet 4: Specyfikacje techniczne
        elif self._is_specification_line(chunk_text):
            priority = 4
        
        # Priorytet 3: Strony z modelami
        elif item.get('is_model_page', False):
            priority = 3
        
        # Priorytet 2: Strony z kategoriami electric/configurator
        elif any(cat in item.get('categories', []) for cat in ['electric', 'configurator']):
            priority = 2
        
        return priority
    
    def _enrich_with_content_tags(self, metadata: Dict[str, Any], text: str):
        """Wzbogaca metadata o tagi z treści"""
        text_lower = text.lower()
        
        # Tagi tematyczne
        tags = []
        
        if any(word in text_lower for word in ['cena', 'zł', 'koszt', 'od']):
            tags.append('cena')
        
        if any(word in text_lower for word in ['silnik', 'moc', 'km', 'kw', 'nm', 'v6', 'v8']):
            tags.append('silnik')
        
        if any(word in text_lower for word in ['przyspieszenie', '0-100', '0 do 100']):
            tags.append('przyspieszenie')
        
        if any(word in text_lower for word in ['zasięg', 'bateria', 'ładowanie', 'elektryczny']):
            tags.append('zasięg')
        
        if any(word in text_lower for word in ['leasing', 'finansowanie', 'rata', 'kredyt']):
            tags.append('finansowanie')
        
        if any(word in text_lower for word in ['serwis', 'gwarancja', 'przegląd', 'warsztat']):
            tags.append('serwis')
        
        if any(word in text_lower for word in ['bmw x', 'x1', 'x3', 'x5', 'x7']):
            tags.append('suv')
        
        if any(word in text_lower for word in ['seria 3', 'seria 5', 'seria 7']):
            tags.append('sedan')
        
        if any(word in text_lower for word in ['i3', 'i4', 'i7', 'ix', 'elektryczny']):
            tags.append('elektryczny')
        
        if any(word in text_lower for word in ['kombi', 'touring', 'wagon']):
            tags.append('kombi')
        
        if any(word in text_lower for word in ['kabriolet', 'cabrio', 'convertible']):
            tags.append('kabriolet')
        
        if any(word in text_lower for word in ['coupe', 'coupé']):
            tags.append('coupe')
        
        if tags:
            metadata['content_tags'] = tags
    
    # ==================== GŁÓWNA METODA CHUNKOWANIA ====================
    
    def create_intelligent_chunks(self, data_items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Tworzy inteligentne chunk-i zoptymalizowane pod RAG dla BMW
        
        Strategia:
        1. Najpierw czyść tekst
        2. Dla treści ze specyfikacjami - używaj dzielenia po zdaniach
        3. Dla innych - użyj standardowego splittera
        4. Na końcu post-processing
        """
        all_chunks = []
        
        print(f"📝 Przetwarzam {len(data_items)} stron...")
        
        for item_idx, item in enumerate(data_items):
            content = item.get('content', '')
            if not content or len(content.strip()) < 50:
                continue
            
            # 1. OCZYŚĆ TEKST PRZED CHUNKOWANIEM
            cleaned_content = self._clean_text_before_chunking(content)
            
            # Określ typ treści
            is_spec_heavy = item.get('is_model_page', False) and any(
                spec in item.get('specifications', {}) 
                for spec in ['prices', 'engine']
            )
            
            # 2. PODZIEL TEKST W ZALEŻNOŚCI OD TYPU
            if is_spec_heavy:
                # Dla specyfikacji - dziel po zdaniach dla lepszej czytelności
                content_chunks = self._split_by_sentences(cleaned_content, max_chars=400)
            else:
                # Standardowe treści - użyj splittera
                content_chunks = self.text_splitter.split_text(cleaned_content)
            
            # 3. OCZYŚĆ I PRZETWÓRZ KAŻDY CHUNK
            final_chunks = []
            for chunk in content_chunks:
                # Oczyść początek chunka
                chunk = self._clean_chunk_start(chunk)
                
                if not chunk or len(chunk.strip()) < 30:
                    continue
                
                # Jeśli chunek jest wciąż za duży, podziel go
                if len(chunk) > 600:
                    sub_chunks = self._split_by_sentences(chunk, max_chars=400)
                    final_chunks.extend(sub_chunks)
                else:
                    final_chunks.append(chunk)
            
            # 4. STWÓRZ DOKUMENTY Z METADATA
            for i, chunk_text in enumerate(final_chunks):
                # Pomijaj za krótkie chunki (chyba że zawierają specyfikacje)
                if len(chunk_text) < 50 and not self._is_specification_line(chunk_text):
                    continue
                
                chunk_metadata = self._create_rich_metadata(
                    item, i, len(final_chunks), chunk_text
                )
                
                # Dodaj dodatkowe tagi na podstawie treści
                self._enrich_with_content_tags(chunk_metadata, chunk_text)
                
                all_chunks.append({
                    'id': f"{chunk_metadata['unique_id']}_ch{i:03d}_{hash(chunk_text[:100]) % 10000:04d}",
                    'text': chunk_text,
                    'metadata': chunk_metadata
                })
            
            if (item_idx + 1) % 10 == 0:
                print(f"   Przetworzono {item_idx + 1}/{len(data_items)} stron...")
        
        # 5. POST-PROCESSING: wyczyść i zoptymalizuj chunki
        print(f"\n🧹 Post-processing {len(all_chunks)} chunków...")
        all_chunks = self.post_process_chunks(all_chunks)
        
        print(f"✅ Po post-processingu: {len(all_chunks)} chunków")
        
        return all_chunks
    
    # ==================== METODY ZAPISU ====================
    
    def save_for_rag(self, chunks: List[Dict[str, Any]], output_path: Path) -> None:
        """Zapisuje chunk-i w formacie dla RAG ze szczegółowymi statystykami"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        print(f"\n📊 STATYSTYKI CHUNK-ÓW:")
        print(f"   Łącznie chunk-ów: {len(chunks):,}")
        
        if not chunks:
            print("⚠️  Brak chunków do zapisania!")
            return
        
        # Analiza rozkładu rozmiarów
        sizes = [len(c['text']) for c in chunks]
        print(f"   Średni rozmiar: {sum(sizes)/len(sizes):.0f} znaków")
        print(f"   Min: {min(sizes)} znaków, Max: {max(sizes)} znaków")
        
        # Liczba chunk-ów w zakresach
        tiny = sum(1 for s in sizes if s < 100)
        small = sum(1 for s in sizes if 100 <= s < 200)
        medium = sum(1 for s in sizes if 200 <= s < 400)
        large = sum(1 for s in sizes if 400 <= s < 600)
        huge = sum(1 for s in sizes if s >= 600)
        
        print(f"   Bardzo małe (<100): {tiny} ({tiny/len(chunks)*100:.1f}%)")
        print(f"   Małe (100-200): {small} ({small/len(chunks)*100:.1f}%)")
        print(f"   Średnie (200-400): {medium} ({medium/len(chunks)*100:.1f}%)")
        print(f"   Duże (400-600): {large} ({large/len(chunks)*100:.1f}%)")
        print(f"   Bardzo duże (≥600): {huge} ({huge/len(chunks)*100:.1f}%)")
        
        # Statystyki tematyczne
        spec_chunks = sum(1 for c in chunks if c['metadata'].get('chunk_has_specs', False))
        model_chunks = sum(1 for c in chunks if c['metadata'].get('is_model_page', False))
        price_chunks = sum(1 for c in chunks if c['metadata'].get('has_prices', False))
        
        print(f"   Chunk-i ze specyfikacjami: {spec_chunks} ({spec_chunks/len(chunks)*100:.1f}%)")
        print(f"   Chunk-i z modelami: {model_chunks} ({model_chunks/len(chunks)*100:.1f}%)")
        print(f"   Chunk-i z cenami: {price_chunks} ({price_chunks/len(chunks)*100:.1f}%)")
        
        # Analiza jakości - ile chunków zaczyna się od interpunkcji
        bad_starts = sum(1 for c in chunks if c['text'].startswith(',') or c['text'].startswith('.') or c['text'].startswith(';'))
        print(f"   Chunk-i z złym początkiem: {bad_starts} ({bad_starts/len(chunks)*100:.1f}%)")
        
        # Nowe statystyki jakości
        bad_phrase_chunks = sum(1 for c in chunks if any(
            phrase in c['text'].lower() 
            for phrase in self.bad_phrases
        ))
        print(f"   Chunki z placeholderami: {bad_phrase_chunks}")
        
        very_large_chunks = sum(1 for c in chunks if len(c['text']) > 1000)
        print(f"   Chunki >1000 znaków: {very_large_chunks}")
        
        # Sprawdź duplikaty
        text_hashes = [hash(self._normalize_for_deduplication(c['text'])) for c in chunks]
        unique_hashes = set(text_hashes)
        duplicate_count = len(chunks) - len(unique_hashes)
        print(f"   Prawdopodobne duplikaty: {duplicate_count} ({duplicate_count/len(chunks)*100:.1f}%)")
        
        # 1. Zapisz JSONL (główne dane)
        jsonl_file = output_path.with_suffix('.jsonl')
        print(f"\n💾 Zapisuję JSONL: {jsonl_file}")
        
        with open(jsonl_file, 'w', encoding='utf-8') as f:
            for chunk in chunks:
                json.dump(chunk, f, ensure_ascii=False)
                f.write('\n')
        
        # 2. Zapisz statystyki szczegółowe
        stats_file = output_path.with_suffix('.stats.json')
        stats = {
            'total_chunks': len(chunks),
            'size_distribution': {
                'tiny': tiny,
                'small': small,
                'medium': medium,
                'large': large,
                'huge': huge
            },
            'thematic_distribution': {
                'spec_chunks': spec_chunks,
                'model_chunks': model_chunks,
                'price_chunks': price_chunks
            },
            'quality_metrics': {
                'bad_starts': bad_starts,
                'bad_starts_percentage': bad_starts/len(chunks)*100 if chunks else 0,
                'bad_phrase_chunks': bad_phrase_chunks,
                'very_large_chunks': very_large_chunks,
                'duplicate_chunks': duplicate_count,
                'unique_chunks': len(unique_hashes),
                'unique_percentage': len(unique_hashes)/len(chunks)*100 if chunks else 0
            },
            'size_stats': {
                'avg_chars': sum(sizes) / len(sizes) if sizes else 0,
                'min_chars': min(sizes) if sizes else 0,
                'max_chars': max(sizes) if sizes else 0,
                'median_chars': sorted(sizes)[len(sizes)//2] if sizes else 0
            },
            'priority_distribution': {
                str(i): sum(1 for c in chunks if c['metadata'].get('retrieval_priority', 1) == i)
                for i in range(1, 6)
            },
            'saved_at': datetime.now().isoformat(),
            'chunker_settings': {
                'chunk_size': self.chunk_size,
                'chunk_overlap': self.chunk_overlap
            }
        }
        
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
        
        # 3. Zapisz CSV z podglądem
        csv_file = output_path.with_suffix('.preview.csv')
        print(f"💾 Zapisuję CSV preview: {csv_file}")
        
        try:
            csv_data = []
            for chunk in chunks[:1000]:  # Max 1000 wierszy dla czytelności
                csv_data.append({
                    'id': chunk.get('id', ''),
                    'text_preview': (chunk['text'][:200] + '...') if len(chunk['text']) > 200 else chunk['text'],
                    'models': ', '.join(chunk['metadata'].get('models', [])[:3]),
                    'title': chunk['metadata'].get('title', '')[:80],
                    'priority': chunk['metadata'].get('retrieval_priority', 1),
                    'has_specs': chunk['metadata'].get('chunk_has_specs', False),
                    'has_prices': chunk['metadata'].get('has_prices', False),
                    'size_chars': chunk['metadata'].get('chunk_char_count', 0),
                    'tags': ', '.join(chunk['metadata'].get('content_tags', [])[:3])
                })
            
            df = pd.DataFrame(csv_data)
            df.to_csv(csv_file, index=False, encoding='utf-8')
        except Exception as e:
            print(f"   ⚠️ Błąd przy zapisie CSV: {e}")
        
        print(f"\n✅ ZAPISANO:")
        print(f"   📄 JSONL: {jsonl_file} ({jsonl_file.stat().st_size / 1024:.1f} KB)")
        print(f"   📈 Statystyki: {stats_file}")
        print(f"   📊 Podgląd: {csv_file}")
        
        # Pokaż przykładowe chunki
        print(f"\n🎯 PRZYKŁADOWE CHUNK-I (pierwsze 3):")
        for i, chunk in enumerate(chunks[:3]):
            print(f"\n{'='*60}")
            print(f"CHUNK #{i+1}:")
            print(f"ID: {chunk['id']}")
            print(f"Modele: {chunk['metadata'].get('models', [])}")
            print(f"Priorytet: {chunk['metadata'].get('retrieval_priority', 1)}")
            print(f"Tagi: {chunk['metadata'].get('content_tags', [])}")
            print(f"Rozmiar: {chunk['metadata'].get('chunk_char_count', 0)} znaków")
            print(f"Tekst: {chunk['text'][:150]}...")
        
        return chunks


def find_latest_normalized_folder():
    """Znajduje najnowszy znormalizowany folder"""
    output_base = Path(r"C:\Users\hellb\Documents\Chatbot_BMW\RAG\output")
    
    # Szukaj folderów z "_normalized_filtered" (najpierw) lub "_normalized"
    filtered_folders = [f for f in output_base.iterdir() 
                       if f.is_dir() and "_normalized_filtered" in f.name]
    
    if filtered_folders:
        latest = sorted(filtered_folders)[-1]
        print(f"📁 Znaleziono przefiltrowane dane: {latest.name}")
        return latest
    
    # Jeśli nie ma przefiltrowanych, szukaj zwykłych znormalizowanych
    normalized_folders = [f for f in output_base.iterdir() 
                         if f.is_dir() and "_normalized" in f.name]
    
    if not normalized_folders:
        print("❌ Nie znaleziono znormalizowanych danych!")
        print("   Najpierw uruchom normalizację (2_normalizer.py).")
        return None
    
    latest = sorted(normalized_folders)[-1]
    print(f"📁 Znaleziono znormalizowane dane: {latest.name}")
    return latest


def prepare_for_rag():
    """Przygotowuje dane dla RAG z inteligentnym chunkowaniem"""
    # Znajdź znormalizowane dane
    latest_normalized = find_latest_normalized_folder()
    
    if not latest_normalized:
        return None
    
    # Wczytaj znormalizowane dane
    all_file = latest_normalized / "all_normalized.jsonl"
    if not all_file.exists():
        print(f"❌ Brak pliku: {all_file}")
        # Spróbuj znaleźć inny plik
        jsonl_files = list(latest_normalized.glob("*.jsonl"))
        if jsonl_files:
            all_file = jsonl_files[0]
            print(f"   Użyto: {all_file.name}")
        else:
            return None
    
    # Wczytaj dane
    data_items = []
    print(f"\n📖 Wczytuję dane z: {all_file}")
    
    with open(all_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            try:
                data_items.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"   ⚠️ Błąd w linii {line_num}: {e}")
                continue
    
    print(f"✅ Wczytano {len(data_items)} znormalizowanych stron")
    
    # Statystyki wczytanych danych
    model_pages = sum(1 for item in data_items if item.get('is_model_page', False))
    pages_with_prices = sum(1 for item in data_items if item.get('specifications', {}).get('prices'))
    pages_with_engine = sum(1 for item in data_items if item.get('specifications', {}).get('engine'))
    
    print(f"   🚗 Strony z modelami: {model_pages}")
    print(f"   💰 Strony z cenami: {pages_with_prices}")
    print(f"   🔧 Strony z silnikiem: {pages_with_engine}")
    print(f"   📄 Łącznie słów: {sum(item.get('word_count', 0) for item in data_items):,}")
    
    # Stwórz chunk-i z inteligentnym preprocessorem
    print("\n🤖 Tworzę inteligentne chunk-i...")
    preprocessor = RAGPreprocessor(chunk_size=500, chunk_overlap=100)
    
    # Użyj nowej metody z post-processingiem
    all_chunks = preprocessor.create_intelligent_chunks(data_items)
    
    if not all_chunks:
        print("❌ Nie udało się utworzyć chunk-ów!")
        return None
    
    # Utwórz folder na dane RAG
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    rag_output = latest_normalized.parent / f"rag_ready_final_{timestamp}"
    rag_output.mkdir(exist_ok=True)
    
    # Zapisz wszystkie chunk-i
    chunks_file = rag_output / "all_chunks.jsonl"
    print(f"\n💾 Zapisuję chunk-i do: {rag_output}")
    preprocessor.save_for_rag(all_chunks, chunks_file)
    
    # Dodatkowo: zapisz chunk-i z modelami osobno dla łatwiejszego debugowania
    model_chunks = [c for c in all_chunks if c['metadata'].get('is_model_page', False)]
    if model_chunks:
        models_file = rag_output / "model_chunks.jsonl"
        print(f"\n🚗 Zapisuję osobno chunk-i z modelami...")
        preprocessor.save_for_rag(model_chunks, models_file)
    
    # Zapisz pozostałe chunk-i
    other_chunks = [c for c in all_chunks if not c['metadata'].get('is_model_page', False)]
    if other_chunks:
        others_file = rag_output / "other_chunks.jsonl"
        print(f"\n📰 Zapisuję pozostałe chunk-i...")
        preprocessor.save_for_rag(other_chunks, others_file)
    
    # Utwórz plik konfiguracyjny
    config = {
        'created_at': datetime.now().isoformat(),
        'source_normalized_data': str(latest_normalized),
        'chunker_settings': {
            'chunk_size': 500,
            'chunk_overlap': 100
        },
        'statistics': {
            'total_chunks': len(all_chunks),
            'model_chunks': len(model_chunks),
            'other_chunks': len(other_chunks),
            'avg_chunk_size': sum(len(c['text']) for c in all_chunks) / len(all_chunks) if all_chunks else 0
        },
        'files': {
            'all_chunks': 'all_chunks.jsonl',
            'model_chunks': 'model_chunks.jsonl' if model_chunks else None,
            'other_chunks': 'other_chunks.jsonl' if other_chunks else None
        }
    }
    
    config_file = rag_output / "config.json"
    with open(config_file, 'w', encoding='utf-8') as f:
        json.dump(config, f, ensure_ascii=False, indent=2)
    
    print(f"\n🎉 PRZYGOTOWANIE DANYCH ZAKOŃCZONE!")
    print(f"📁 Folder z danymi: {rag_output}")
    
    return all_chunks


def debug_sample_chunks():
    """Pokazuje przykładowe chunki do debugowania"""
    output_base = Path(r"C:\Users\hellb\Documents\Chatbot_BMW\RAG\output")
    
    # Znajdź najnowsze dane RAG
    rag_folders = [f for f in output_base.iterdir() 
                  if f.is_dir() and "rag_ready_final" in f.name]
    
    if not rag_folders:
        print("❌ Nie znaleziono danych RAG!")
        # Sprawdź czy są inne wersje
        other_rag = [f for f in output_base.iterdir() 
                    if f.is_dir() and "rag_ready" in f.name]
        if other_rag:
            latest_other = sorted(other_rag)[-1]
            print(f"📁 Znaleziono inne dane RAG: {latest_other.name}")
            rag_folders = [latest_other]
        else:
            return
    
    latest_rag = sorted(rag_folders)[-1]
    chunks_file = latest_rag / "all_chunks.jsonl"
    
    if not chunks_file.exists():
        print(f"❌ Nie znaleziono pliku: {chunks_file}")
        return
    
    print(f"\n🔍 ANALIZA CHUNK-ÓW z: {latest_rag.name}")
    
    # Wczytaj pierwsze 10 chunk-ów
    chunks = []
    with open(chunks_file, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= 10:
                break
            chunks.append(json.loads(line))
    
    print(f"\n📊 Przeanalizowano {len(chunks)} chunk-ów:")
    
    for i, chunk in enumerate(chunks):
        starts_with_bad_char = chunk['text'].startswith(',') or chunk['text'].startswith('.') or chunk['text'].startswith(';')
        has_bad_phrase = any(phrase in chunk['text'].lower() for phrase in ['lorem ipsum', 'dummy text', 'placeholder'])
        marker = "⚠️ " if (starts_with_bad_char or has_bad_phrase) else "✓ "
        
        print(f"\n{marker}CHUNK #{i+1}:")
        print(f"ID: {chunk['id']}")
        print(f"Modele: {chunk['metadata'].get('models', [])}")
        print(f"Tytuł: {chunk['metadata'].get('title', '')}")
        print(f"Priorytet: {chunk['metadata'].get('retrieval_priority', 1)}")
        print(f"Rozmiar: {chunk['metadata'].get('chunk_char_count', 0)} znaków")
        print(f"Początek: '{chunk['text'][:50]}...'")
        
        if starts_with_bad_char:
            print("🚨 PROBLEM: Chunk zaczyna się od interpunkcji!")
        if has_bad_phrase:
            print("🚨 PROBLEM: Chunk zawiera placeholder!")
        
        # Sprawdź czy to dobry chunek
        is_good = (not starts_with_bad_char and 
                  not has_bad_phrase and 
                  len(chunk['text']) > 50 and 
                  len(chunk['text']) < 1000)
        
        if is_good:
            print("✅ DOBRY CHUNK")


if __name__ == "__main__":
    print("=" * 70)
    print("🤖 FINALNY CHUNKER DLA RAG - BMW")
    print("=" * 70)
    print("\n🔧 Funkcje:")
    print("   • Inteligentne chunkowanie zależne od typu treści")
    print("   • Post-processing: usuwanie złych chunków, duplikatów, łączenie krótkich")
    print("   • Filtrowanie placeholderów (Lorem Ipsum, menu, etc.)")
    print("   • Bogate statystyki jakości")
    print()
    
    # Menu
    print("📋 MENU:")
    print("1. Przygotuj dane dla RAG (chunkowanie)")
    print("2. Debuguj istniejące chunki")
    print("3. Sprawdź dostępne znormalizowane dane")
    print()
    
    try:
        choice = input("Wybierz opcję (1-3): ").strip()
        
        if choice == "1":
            print("\n🚀 Rozpoczynam przygotowanie danych dla RAG...")
            chunks = prepare_for_rag()
            
            if chunks:
                print(f"\n✅ Sukces! Utworzono {len(chunks)} wysokiej jakości chunków.")
                print("   Możesz teraz przejść do tworzenia embeddingów (4_embeddings.py).")
            
        elif choice == "2":
            debug_sample_chunks()
            
        elif choice == "3":
            folder = find_latest_normalized_folder()
            if folder:
                print(f"\n📁 Najnowszy folder: {folder}")
                files = list(folder.glob("*"))
                print(f"📄 Pliki ({len(files)}):")
                for file in files:
                    size_kb = file.stat().st_size / 1024
                    print(f"   - {file.name} ({size_kb:.1f} KB)")
            else:
                print("❌ Nie znaleziono znormalizowanych danych!")
                
        else:
            print("❌ Nieznana opcja. Kończę.")
            
    except KeyboardInterrupt:
        print("\n\n⏹️  Przerwano przez użytkownika.")
    except Exception as e:
        print(f"\n❌ Wystąpił błąd: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*70)
    print("🤖 Chunker zakończył pracę")
    print("="*70)