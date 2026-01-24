from langchain.text_splitter import RecursiveCharacterTextSplitter
import json
import pandas as pd
from pathlib import Path
from datetime import datetime
import re
from typing import List, Dict, Any, Optional

class RAGPreprocessor:
    """Przygotowuje dane dla RAG z optymalnym chunkowaniem dla specyfikacji BMW"""
    
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 100):
        """
        Inicjalizuje preprocessor z optymalnymi ustawieniami dla danych BMW
        
        Args:
            chunk_size: Rozmiar chunka w znakach (500 ≈ 125-150 tokenów)
            chunk_overlap: Przekrycie między chunkami
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
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
                ".\s",         # Koniec zdania + spacja
                "\n",          # Nowa linia
                "; ",          # Koniec frazy
                ", ",          # Przecinek
                " ",           # Spacja
                ""             # Ostateczność
            ],
            keep_separator=True  # Ważne dla zachowania struktury
        )
    
    def _estimate_tokens(self, text: str) -> int:
        """Szacuje liczbę tokenów dla tekstu"""
        return len(text) // 3  # Konserwatywne oszacowanie dla polskiego
    
    def _extract_specification_blocks(self, text: str) -> List[str]:
        """
        Wyodrębnia bloki specyfikacji z tekstu
        (Ceny, parametry techniczne, tabele danych)
        """
        spec_blocks = []
        
        # Wzorce dla specyfikacji BMW
        spec_patterns = [
            # Ceny
            r'(?:(?:cena|od|od\s+)\s*)?\d{1,3}(?:\s?\d{3})*(?:[.,]\d{2})?\s*zł.*?(?=\n\n|\n\S|$)',
            # Moc silnika
            r'\b\d+(?:[.,]\d+)?\s*(?:kW|KM|kon[i|e]).*?(?=\n\n|\n\S|$)',
            # Przyspieszenie
            r'\b0[-–]\s*100[^:]*:\s*\d+[.,]\d+\s*s(?:ek\.?)?.*?(?=\n\n|\n\S|$)',
            # Zasięg
            r'\bzasięg[^:]*:\s*\d+(?:[.,]\d+)?\s*(?:km|kilometrów).*?(?=\n\n|\n\S|$)',
            # Pojemność silnika
            r'\b\d+(?:[.,]\d+)?\s*(?:cm³|ccm|l|litr[ów]?).*?(?=\n\n|\n\S|$)',
            # Spalanie
            r'\b\d+(?:[.,]\d+)?\s*(?:l/100km|l\./100\s*km).*?(?=\n\n|\n\S|$)',
        ]
        
        # Szukaj wszystkich specyfikacji
        for pattern in spec_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE | re.MULTILINE)
            for match in matches:
                spec_block = match.group(0).strip()
                if len(spec_block) > 10 and len(spec_block) < 300:
                    spec_blocks.append(spec_block)
        
        return spec_blocks
    
    def _split_into_semantic_units(self, text: str) -> List[str]:
        """
        Dzieli tekst na semantyczne jednostki dla BMW
        """
        units = []
        
        # 1. Podziel na akapity
        paragraphs = re.split(r'\n\s*\n', text.strip())
        
        for paragraph in paragraphs:
            # Pomiń puste
            if not paragraph.strip():
                continue
            
            # 2. Sprawdź czy to lista specyfikacji
            lines = paragraph.split('\n')
            
            # Jeśli mamy listę punktowaną ze specyfikacjami
            if len(lines) > 1 and any(re.match(r'^[•\-*]\s+', line) for line in lines):
                # Każdy punkt jako osobny chunk jeśli zawiera specyfikacje
                for line in lines:
                    if line.strip() and self._is_specification_line(line):
                        units.append(line.strip())
                    elif line.strip():
                        units.append(line.strip())
            
            # 3. Długi akapit - sprawdź czy zawiera specyfikacje
            elif len(paragraph) > 300 and ':' in paragraph:
                # Spróbuj podzielić po średnikach/dwukropkach
                segments = re.split(r'[;:]', paragraph)
                current_segment = ""
                
                for segment in segments:
                    segment = segment.strip()
                    if not segment:
                        continue
                    
                    # Jeśli segment zawiera specyfikację, traktuj jako osobny
                    if self._is_specification_line(segment):
                        if current_segment:
                            units.append(current_segment)
                            current_segment = ""
                        units.append(segment)
                    else:
                        # Łącz z poprzednim jeśli nie za długi
                        if len(current_segment) + len(segment) + 2 < self.chunk_size:
                            current_segment = f"{current_segment}: {segment}" if current_segment else segment
                        else:
                            if current_segment:
                                units.append(current_segment)
                            current_segment = segment
                
                if current_segment:
                    units.append(current_segment)
            
            # 4. Standardowy akapit
            else:
                units.append(paragraph.strip())
        
        return units
    
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
        ]
        
        text_lower = text.lower()
        return any(re.search(pattern, text_lower) for pattern in spec_indicators)
    
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
            # Modele
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
    
    def create_intelligent_chunks(self, data_items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Tworzy inteligentne chunk-i zoptymalizowane pod RAG dla BMW
        
        Strategia:
        1. Specyfikacje → małe, precyzyjne chunki (200-400 znaków)
        2. Opisy modeli → średnie chunki (400-600 znaków)
        3. Treści ogólne → standardowe chunki (500 znaków)
        """
        all_chunks = []
        
        for item_idx, item in enumerate(data_items):
            content = item.get('content', '')
            if not content or len(content.strip()) < 30:
                continue
            
            # Określ typ treści
            is_spec_heavy = item.get('is_model_page', False) and any(
                spec in item.get('specifications', {}) 
                for spec in ['prices', 'engine']
            )
            
            # DOSTOSUJ chunk_size w zależności od typu
            if is_spec_heavy:
                # Specyfikacje → mniejsze chunki
                temp_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=400,
                    chunk_overlap=80,
                    length_function=len,
                    separators=self.text_splitter._separators,
                    keep_separator=True
                )
                content_chunks = temp_splitter.split_text(content)
            else:
                # Standardowe treści
                content_chunks = self.text_splitter.split_text(content)
            
            # Jeśli chunk-i są za duże, podziel dalej
            final_chunks = []
            for chunk in content_chunks:
                if len(chunk) > 600:  # Jeszcze za duży
                    # Podziel po zdaniach
                    sentences = re.split(r'(?<=[.!?])\s+', chunk)
                    current_chunk = ""
                    
                    for sentence in sentences:
                        if len(current_chunk) + len(sentence) < 400:
                            current_chunk += " " + sentence if current_chunk else sentence
                        else:
                            if current_chunk:
                                final_chunks.append(current_chunk.strip())
                            current_chunk = sentence
                    
                    if current_chunk:
                        final_chunks.append(current_chunk.strip())
                else:
                    final_chunks.append(chunk)
            
            # Stwórz dokumenty z metadata
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
        
        return all_chunks
    
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
        
        if tags:
            metadata['content_tags'] = tags
    
    def save_for_rag(self, chunks: List[Dict[str, Any]], output_path: Path) -> None:
        """Zapisuje chunk-i w formacie dla RAG ze statystykami"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        print(f"\n📊 STATYSTYKI CHUNK-ÓW:")
        print(f"   Łącznie chunk-ów: {len(chunks):,}")
        
        # Analiza rozkładu rozmiarów
        sizes = [len(c['text']) for c in chunks]
        print(f"   Średni rozmiar: {sum(sizes)/len(sizes):.0f} znaków")
        print(f"   Min: {min(sizes)} znaków, Max: {max(sizes)} znaków")
        
        # Liczba chunk-ów w zakresach
        small = sum(1 for s in sizes if s < 200)
        medium = sum(1 for s in sizes if 200 <= s < 400)
        large = sum(1 for s in sizes if s >= 400)
        
        print(f"   Małe (<200): {small} ({small/len(chunks)*100:.1f}%)")
        print(f"   Średnie (200-400): {medium} ({medium/len(chunks)*100:.1f}%)")
        print(f"   Duże (≥400): {large} ({large/len(chunks)*100:.1f}%)")
        
        # Statystyki tematyczne
        spec_chunks = sum(1 for c in chunks if c['metadata'].get('chunk_has_specs', False))
        model_chunks = sum(1 for c in chunks if c['metadata'].get('is_model_page', False))
        price_chunks = sum(1 for c in chunks if c['metadata'].get('has_prices', False))
        
        print(f"   Chunk-i ze specyfikacjami: {spec_chunks} ({spec_chunks/len(chunks)*100:.1f}%)")
        print(f"   Chunk-i z modelami: {model_chunks} ({model_chunks/len(chunks)*100:.1f}%)")
        print(f"   Chunk-i z cenami: {price_chunks} ({price_chunks/len(chunks)*100:.1f}%)")
        
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
                'small': small,
                'medium': medium,
                'large': large
            },
            'thematic_distribution': {
                'spec_chunks': spec_chunks,
                'model_chunks': model_chunks,
                'price_chunks': price_chunks
            },
            'size_stats': {
                'avg_chars': sum(sizes) / len(sizes),
                'min_chars': min(sizes),
                'max_chars': max(sizes),
                'median_chars': sorted(sizes)[len(sizes)//2]
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
        print(f"\n🎯 PRZYKŁADOWE CHUNK-I:")
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


def find_latest_data():
    """Znajduje najnowsze dane"""
    output_base = Path(r"C:\Users\hellb\Documents\Chatbot_BMW\RAG\output")
    
    # 1. Najpierw szukaj znormalizowanych danych
    normalized_folders = [f for f in output_base.iterdir() 
                         if f.is_dir() and "_normalized" in f.name]
    
    if normalized_folders:
        latest_normalized = sorted(normalized_folders)[-1]
        print(f"📁 Znaleziono znormalizowane dane: {latest_normalized.name}")
        return latest_normalized
    
    # 2. Jeśli nie ma znormalizowanych, szukaj surowych danych
    raw_folders = [f for f in output_base.iterdir() 
                  if f.is_dir() and f.name.startswith("bmw_complete")]
    
    if not raw_folders:
        print("❌ Nie znaleziono żadnych danych!")
        return None
    
    latest_raw = sorted(raw_folders)[-1]
    print(f"📁 Znaleziono surowe dane: {latest_raw.name}")
    print("⚠️  Najpierw uruchom normalizację danych!")
    return latest_raw


def analyze_output():
    """Analizuje dostępne dane"""
    latest_folder = find_latest_data()
    
    if latest_folder:
        # Sprawdź jakie pliki są dostępne
        files = list(latest_folder.glob("*.json*"))
        files.extend(latest_folder.glob("*.txt"))
        
        print(f"📊 Znaleziono {len(files)} plików:")
        for file in files[:10]:  # Pokaż tylko pierwsze 10
            size_kb = file.stat().st_size / 1024
            print(f"   - {file.name} ({size_kb:.1f} KB)")
        
        if len(files) > 10:
            print(f"   ... i {len(files) - 10} więcej")
        
        return True
    return False


def prepare_for_rag():
    """Przygotowuje dane dla RAG z inteligentnym chunkowaniem"""
    # Znajdź znormalizowane dane
    output_base = Path(r"C:\Users\hellb\Documents\Chatbot_BMW\RAG\output")
    
    # Szukaj folderów z "_normalized"
    normalized_folders = [f for f in output_base.iterdir() 
                         if f.is_dir() and "_normalized" in f.name]
    
    if not normalized_folders:
        print("❌ Nie znaleziono znormalizowanych danych!")
        print("   Najpierw uruchom normalizację (2_normalizer.py).")
        return None
    
    latest_normalized = sorted(normalized_folders)[-1]
    
    # Wczytaj znormalizowane dane
    all_file = latest_normalized / "all_normalized.jsonl"
    if not all_file.exists():
        print(f"❌ Brak pliku: {all_file}")
        print(f"   Sprawdzam inne pliki w {latest_normalized}:")
        for file in latest_normalized.iterdir():
            print(f"   - {file.name}")
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
    
    # Użyj nowej metody
    all_chunks = preprocessor.create_intelligent_chunks(data_items)
    
    if not all_chunks:
        print("❌ Nie udało się utworzyć chunk-ów!")
        return None
    
    # Utwórz folder na dane RAG
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    rag_output = latest_normalized.parent / f"rag_ready_intelligent_{timestamp}"
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
    
    # Utwórz plik konfiguracyjny dla embeddingów
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
            'avg_chunk_size': sum(len(c['text']) for c in all_chunks) / len(all_chunks)
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
    print(f"📄 Główne pliki:")
    print(f"   • all_chunks.jsonl - wszystkie chunk-i")
    if model_chunks:
        print(f"   • model_chunks.jsonl - tylko modele ({len(model_chunks)} chunk-i)")
    if other_chunks:
        print(f"   • other_chunks.jsonl - inne treści ({len(other_chunks)} chunk-i)")
    print(f"\n🚀 Następny krok: Uruchom 4_embeddings.py aby stworzyć embeddingi")
    
    return all_chunks


def debug_sample_chunks():
    """Pokazuje przykładowe chunki do debugowania"""
    output_base = Path(r"C:\Users\hellb\Documents\Chatbot_BMW\RAG\output")
    
    # Znajdź najnowsze dane RAG
    rag_folders = [f for f in output_base.iterdir() 
                  if f.is_dir() and "rag_ready_intelligent" in f.name]
    
    if not rag_folders:
        print("❌ Nie znaleziono danych RAG!")
        return
    
    latest_rag = sorted(rag_folders)[-1]
    chunks_file = latest_rag / "all_chunks.jsonl"
    
    if not chunks_file.exists():
        print(f"❌ Nie znaleziono pliku: {chunks_file}")
        return
    
    print(f"\n🔍 ANALIZA CHUNK-ÓW z: {latest_rag.name}")
    
    # Wczytaj pierwsze 5 chunk-ów
    chunks = []
    with open(chunks_file, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= 5:
                break
            chunks.append(json.loads(line))
    
    for i, chunk in enumerate(chunks):
        print(f"\n{'='*80}")
        print(f"CHUNK #{i+1}:")
        print(f"ID: {chunk['id']}")
        print(f"Modele: {chunk['metadata'].get('models', [])}")
        print(f"Tytuł: {chunk['metadata'].get('title', '')}")
        print(f"Priorytet: {chunk['metadata'].get('retrieval_priority', 1)}")
        print(f"Tagi: {chunk['metadata'].get('content_tags', [])}")
        print(f"Rozmiar: {chunk['metadata'].get('chunk_char_count', 0)} znaków")
        print(f"Czy ma specyfikacje: {chunk['metadata'].get('chunk_has_specs', False)}")
        print(f"Tekst:\n{chunk['text'][:300]}...")
    
    print(f"\n📊 Pokazano {len(chunks)} z wszystkich chunk-ów")


if __name__ == "__main__":
    print("=" * 70)
    print("🤖 INTELLIGENTNY CHUNKER DLA RAG - BMW")
    print("=" * 70)
    print("\n🔧 Ustawienia:")
    print("   • Chunk size: 500 znaków (~150 tokenów)")
    print("   • Chunk overlap: 100 znaków")
    print("   • Priorytety retrieval: 1-5 (5 najwyższy)")
    print("   • Specyfikacje → małe, precyzyjne chunki")
    print("   • Opisy → standardowe chunki")
    print()
    
    # Menu
    print("📋 MENU:")
    print("1. Przygotuj dane dla RAG (chunkowanie)")
    print("2. Debuguj istniejące chunki")
    print("3. Sprawdź dostępne dane")
    print()
    
    choice = input("Wybierz opcję (1-3): ").strip()
    
    if choice == "1":
        print("\n🚀 Rozpoczynam przygotowanie danych dla RAG...")
        chunks = prepare_for_rag()
        
    elif choice == "2":
        debug_sample_chunks()
        
    elif choice == "3":
        analyze_output()
        
    else:
        print("❌ Nieznana opcja. Kończę.")
    
    print("\n" + "="*70)
    print("🤖 Chunker zakończył pracę")
    print("="*70)