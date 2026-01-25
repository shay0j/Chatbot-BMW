import json
import re
from pathlib import Path
from datetime import datetime
from collections import Counter
from typing import Dict, List, Any

class DataNormalizer:
    """Normalizuje i czyści zebrane dane z filtrowaniem złych stron"""
    
    def __init__(self, input_folder):
        self.input_folder = Path(input_folder)
        self.output_folder = self.input_folder.parent / f"{self.input_folder.name}_normalized_filtered"
        self.output_folder.mkdir(exist_ok=True)
        
        # Lista złych fraz do filtrowania
        self.bad_phrases = [
            'lorem ipsum',
            'skip to main content',
            'dummy text',
            'example text',
            'placeholder',
            'jump to navigation',
            'breadcrumb',
            'footer',
            'header',
            'menu',
            'navbar',
            'cookie policy',
            'privacy policy',
            'terms of use',
            'regulamin',
            'polityka prywatności',
            'cookies'
        ]
        
        # Słowa kluczowe menu/nawigacji
        self.menu_keywords = [
            'home', 'strona główna', 'produkty', 'usługi', 'kontakt', 'o nas',
            'znajdź dealer', 'znajdź serwis', 'mapa serwisów', 'lokalizacja',
            'kariera', 'praca', 'aktualności', 'news', 'blog', 'media',
            'biuro prasowe', 'inwestorzy', 'dostępność', 'dostępność serwisu'
        ]
    
    def should_keep_item(self, item: Dict[str, Any]) -> bool:
        """
        Sprawdza czy strona powinna być zachowana
        Zwraca False dla placeholderów, menu, krótkich treści
        """
        content = item.get('content', '').lower()
        title = item.get('title', '').lower()
        url = item.get('url', '').lower()
        
        # 1. Sprawdź czy to placeholder/dummy content
        for phrase in self.bad_phrases:
            if phrase in content or phrase in title:
                print(f"  🗑️  FILTER: Bad phrase '{phrase}' in: {title[:60]}")
                return False
        
        # 2. Sprawdź czy to głównie menu/nawigacja
        if self._is_menu_page(content, title):
            print(f"  🗑️  FILTER: Menu/navigation page: {title[:60]}")
            return False
        
        # 3. Sprawdź minimalną długość treści
        content_length = len(content)
        if content_length < 300:  # Za krótkie strony są mało przydatne
            # Ale sprawdź czy to może być strona z cenami/specyfikacjami
            if not self._has_valuable_info(content):
                print(f"  🗑️  FILTER: Too short ({content_length} chars): {title[:60]}")
                return False
        
        # 4. Sprawdź czy to tylko lista modeli bez kontekstu
        if self._is_just_model_list(content):
            print(f"  🗑️  FILTER: Just model list: {title[:60]}")
            return False
        
        # 5. Sprawdź czy URL wskazuje na niechciane strony
        if self._is_bad_url(url):
            print(f"  🗑️  FILTER: Bad URL pattern: {url[:80]}")
            return False
        
        # 6. Sprawdź czy strona ma jakiekolwiek specyfikacje/ceny (dla stron modeli)
        if 'models' in item.get('categories', []) or 'models' in title:
            if not self._has_specifications(item, content):
                print(f"  🗑️  FILTER: Model page without specs: {title[:60]}")
                return False
        
        return True
    
    def _is_menu_page(self, content: str, title: str) -> bool:
        """Sprawdza czy strona to głównie menu/nawigacja"""
        words = content.split()
        if len(words) < 50:  # Bardzo krótkie strony są podejrzane
            return True
        
        # Sprawdź stosunek słów menu do całej treści
        menu_word_count = 0
        total_word_count = len(words)
        
        for word in words:
            if any(keyword in word for keyword in self.menu_keywords):
                menu_word_count += 1
        
        # Jeśli ponad 30% słów to słowa z menu - prawdopodobnie strona menu
        if total_word_count > 0 and (menu_word_count / total_word_count) > 0.3:
            return True
        
        # Sprawdź tytuł
        if any(keyword in title for keyword in self.menu_keywords):
            return True
        
        return False
    
    def _has_valuable_info(self, content: str) -> bool:
        """Sprawdza czy krótka treść zawiera wartościowe informacje"""
        # Szukaj cen, specyfikacji, modeli
        valuable_patterns = [
            r'\b\d{1,3}(?:\s?\d{3})*[,.]\d{2}\s*zł\b',
            r'\b\d+(?:[.,]\d+)?\s*(?:kW|KM|kon[i|e])\b',
            r'\b(?:bmw\s+)?[xi]\d+\b',
            r'\bseria\s+[1-8]\b',
            r'\bm\d+\b',
            r'\bcena\s*[=:]\s*\d',
            r'\bsilnik\s*[=:]\s*\d'
        ]
        
        content_lower = content.lower()
        for pattern in valuable_patterns:
            if re.search(pattern, content_lower):
                return True
        
        return False
    
    def _is_just_model_list(self, content: str) -> bool:
        """Sprawdza czy treść to tylko lista modeli bez kontekstu"""
        # Zlicz wystąpienia 'BMW'
        bmw_count = content.lower().count('bmw')
        words = content.split()
        
        # Jeśli dużo BMW, ale mało innych słów
        if bmw_count > 5 and len(words) < 30:
            # Sprawdź różnorodność słów
            unique_words = set(words)
            unique_ratio = len(unique_words) / len(words) if words else 0
            
            # Jeśli duża redundancja (wiele powtórzeń)
            if unique_ratio < 0.4:
                return True
        
        # Sprawdź czy to lista rozdzielona przecinkami/enterami
        lines = content.split('\n')
        if len(lines) > 3:
            # Jeśli większość linii to tylko nazwy modeli
            model_line_count = 0
            for line in lines:
                line = line.strip()
                if line and (
                    line.startswith('BMW') or 
                    re.match(r'^(X|i|M|Seria)\d+', line, re.IGNORECASE) or
                    re.match(r'^\d+\s*[-–]', line)
                ):
                    model_line_count += 1
            
            if model_line_count / len(lines) > 0.7:
                return True
        
        return False
    
    def _is_bad_url(self, url: str) -> bool:
        """Sprawdza czy URL wskazuje na niechcianą stronę"""
        bad_url_patterns = [
            r'privacy', r'policy', r'terms', r'regulamin', r'polityka',
            r'cookie', r'cookies', r'accessibility', r'dostępność',
            r'kontakt', r'contact', r'imprint', r'nota prawna',
            r'career', r'kariera', r'job', r'praca',
            r'media', r'press', r'newsroom', r'biuro-prasowe',
            r'sitemap', r'mapa-strony'
        ]
        
        for pattern in bad_url_patterns:
            if re.search(pattern, url, re.IGNORECASE):
                return True
        
        return False
    
    def _has_specifications(self, item: Dict, content: str) -> bool:
        """Sprawdza czy strona ma specyfikacje techniczne/ceny"""
        # 1. Sprawdź w specyfikacjach w item
        specs = item.get('specifications', {})
        if specs:
            if 'prices' in specs and specs['prices']:
                return True
            if 'engine' in specs and specs['engine']:
                return True
        
        # 2. Sprawdź w treści
        content_lower = content.lower()
        spec_patterns = [
            r'\b\d{1,3}(?:\s?\d{3})*[,.]\d{2}\s*zł\b',
            r'\b\d+(?:[.,]\d+)?\s*(?:kW|KM|kon[i|e])\b',
            r'\b\d+[.,]\d+\s*s\b.*0[-–]100',
            r'\bzasi[ęe]g[^\d]*\d+\s*km\b',
            r'\bsilnik[^\d]*\d+(?:[.,]\d+)?\s*(?:cm³|ccm|l)\b',
            r'\bmoc[^\d]*\d+\s*(?:kW|KM)\b',
            r'\bmoment[^\d]*\d+\s*Nm\b',
            r'\bprzyspieszenie[^\d]*\d+[.,]\d+\s*s\b'
        ]
        
        for pattern in spec_patterns:
            if re.search(pattern, content_lower):
                return True
        
        return False
    
    def normalize_text(self, text):
        """Normalizuje tekst z dodatkowym czyszczeniem"""
        if not text:
            return ""
        
        # 1. Napraw zlepione słowa z liczbami
        text = re.sub(r'([a-zA-Z])(\d)', r'\1 \2', text)  # za2500 -> za 2500
        text = re.sub(r'(\d)([a-zA-Z])', r'\1 \2', text)  # 250KM -> 250 KM
        
        # 2. Napraw CamelCase dla modeli
        text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)  # BMWX3 -> BMW X3
        
        # 3. Zamień nbsp na zwykłe spacje
        text = text.replace('\xa0', ' ')
        text = text.replace('\u200b', ' ')  # zero-width space
        
        # 4. Usuń nadmiarowe białe znaki
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[\r\n\t]+', ' ', text)
        
        # 5. Normalizuj cudzysłowy
        text = text.replace('"', "'")
        
        # 6. Zapewnij spacje po kropkach (jeśli brak)
        text = re.sub(r'\.(?=[a-zA-Z])', '. ', text)
        
        # 7. Usuń ciągi powtarzających się znaków
        text = re.sub(r'([.!?])\1+', r'\1', text)
        
        # 8. Usuń początkowe/końcowe spacje
        text = text.strip()
        
        return text
    
    def extract_structured_specs(self, specs_dict, content):
        """Wyciąga ustrukturyzowane specyfikacje"""
        structured = {
            'model_info': {},
            'engine': {},
            'performance': {},
            'dimensions': {},
            'prices': [],
            'electric': {},
            'other': {}
        }
        
        # Przenieś wykryte modele
        if 'detected_models' in specs_dict:
            structured['model_info']['models'] = specs_dict['detected_models']
        
        # Przenieś dane silnika
        if 'engine' in specs_dict:
            structured['engine'] = specs_dict['engine']
        
        # Przenieś osiągi
        if 'performance' in specs_dict:
            structured['performance'] = specs_dict['performance']
        
        # Przenieś ceny
        if 'prices' in specs_dict:
            structured['prices'] = specs_dict['prices']
        
        # Przenieś dane elektryczne
        if 'electric' in specs_dict:
            structured['electric'] = specs_dict['electric']
        
        # Dodatkowe parsowanie z treści
        self._parse_additional_specs(content, structured)
        
        # Dodaj model z URL jeśli istnieje
        if 'model_from_url' in specs_dict:
            if 'model_info' not in structured:
                structured['model_info'] = {}
            structured['model_info']['model_from_url'] = specs_dict['model_from_url']
        
        return structured
    
    def _parse_additional_specs(self, content, structured):
        """Parsuje dodatkowe specyfikacje z treści"""
        content_lower = content.lower()
        
        # Wymiary
        dim_patterns = {
            'length': r'd[uł]ugo[sś][ćc][:\s]+(\d{3,4}[,\d]*)\s*(?:mm|cm|m)',
            'width': r'szeroko[sś][ćc][:\s]+(\d{3,4}[,\d]*)\s*(?:mm|cm|m)',
            'height': r'wysoko[sś][ćc][:\s]+(\d{3,4}[,\d]*)\s*(?:mm|cm|m)',
            'weight': r'masa[:\s]+(\d{3,4}[,\d]*)\s*kg',
            'trunk': r'baga[żz]nik[:\s]+(\d{2,4}[,\d]*)\s*l',
        }
        
        for key, pattern in dim_patterns.items():
            match = re.search(pattern, content_lower)
            if match:
                structured['dimensions'][key] = match.group(1)
        
        # Skrzynia biegów
        gearbox_match = re.search(r'skrzynia[:\s]+([^\s,.]+)', content_lower)
        if gearbox_match:
            structured['other']['gearbox'] = gearbox_match.group(1)
        
        # Napęd
        drive_match = re.search(r'nap[ęe]d[:\s]+([^\s,.]+)', content_lower)
        if drive_match:
            structured['other']['drive'] = drive_match.group(1)
    
    def normalize_data(self):
        """Główna funkcja normalizacji z filtrowaniem"""
        input_files = {
            'all': self.input_folder / "all_data.jsonl",
            'models': self.input_folder / "models_data.json",
            'other': self.input_folder / "other_data.json"
        }
        
        # Sprawdź czy pliki istnieją
        for file_type, file_path in input_files.items():
            if not file_path.exists():
                print(f"⚠️ Brak pliku: {file_path}")
                # Spróbuj znaleźć alternatywny plik
                alt_files = list(self.input_folder.glob(f"*{file_type}*"))
                if alt_files:
                    input_files[file_type] = alt_files[0]
                    print(f"   Użyto alternatywy: {alt_files[0].name}")
        
        normalized_data = {
            'all': [],
            'models': [],
            'other': []
        }
        
        stats = {
            'total_loaded': 0,
            'filtered_out': 0,
            'kept': 0,
            'filter_reasons': Counter()
        }
        
        print(f"\n🔍 FILTROWANIE I NORMALIZACJA:")
        
        for data_type, input_file in input_files.items():
            if not input_file.exists():
                continue
            
            print(f"\n📄 Przetwarzam: {input_file.name}")
            
            items = []
            
            # Wczytaj dane w zależności od formatu
            try:
                if data_type == 'all':
                    # JSONL format
                    with open(input_file, 'r', encoding='utf-8') as f:
                        items = [json.loads(line) for line in f]
                else:
                    # JSON format
                    with open(input_file, 'r', encoding='utf-8') as f:
                        items = json.load(f)
            except Exception as e:
                print(f"   ❌ Błąd wczytywania {input_file}: {e}")
                continue
            
            stats['total_loaded'] += len(items)
            
            # Filtruj i normalizuj każdy item
            for i, item in enumerate(items):
                if (i + 1) % 10 == 0:
                    print(f"   Przetworzono {i + 1}/{len(items)}...")
                
                # Filtruj
                if not self.should_keep_item(item):
                    stats['filtered_out'] += 1
                    # Zapisz powód filtrowania (możesz dodać logikę śledzenia)
                    continue
                
                # Normalizuj
                normalized_item = self._normalize_item(item)
                if normalized_item:
                    normalized_data[data_type].append(normalized_item)
                    stats['kept'] += 1
        
        # Zapisz znormalizowane dane
        self._save_normalized_data(normalized_data, stats)
        
        print(f"\n✅ Znormalizowane dane zapisane w: {self.output_folder}")
        print(f"📊 Statystyki filtrowania:")
        print(f"   Załadowano: {stats['total_loaded']} stron")
        print(f"   Odrzucono: {stats['filtered_out']} stron ({stats['filtered_out']/max(stats['total_loaded'],1)*100:.1f}%)")
        print(f"   Zachowano: {stats['kept']} stron")
        
        return normalized_data
    
    def _normalize_item(self, item):
        """Normalizuje pojedynczy item"""
        normalized = item.copy()
        
        # Normalizuj teksty
        for field in ['title', 'description', 'content']:
            if field in normalized:
                normalized[field] = self.normalize_text(normalized[field])
        
        # Normalizuj specyfikacje
        if 'specifications' in normalized:
            normalized['specifications'] = self.extract_structured_specs(
                normalized['specifications'],
                normalized.get('content', '') + ' ' + normalized.get('title', '')
            )
        
        # Dodaj metadata
        normalized['normalized_at'] = datetime.now().isoformat()
        normalized['content_length'] = len(normalized.get('content', ''))
        
        # Dodaj unique_id
        normalized['unique_id'] = f"bmw_{abs(hash(normalized.get('url', ''))) % 1000000:06d}"
        
        # Dodaj flagę czy to strona modelu
        content_lower = normalized.get('content', '').lower()
        title_lower = normalized.get('title', '').lower()
        normalized['is_model_page'] = any(
            keyword in content_lower or keyword in title_lower 
            for keyword in ['seria', 'x1', 'x3', 'x5', 'x7', 'i3', 'i4', 'i7', 'ix', 'm3', 'm5']
        )
        
        # Dodaj liczbę słów (przydatne dla chunkera)
        normalized['word_count'] = len(normalized.get('content', '').split())
        
        # Usuń puste pola
        normalized = {k: v for k, v in normalized.items() if v not in [None, '', [], {}]}
        
        return normalized
    
    def _save_normalized_data(self, data, stats):
        """Zapisuje znormalizowane dane"""
        # 1. Wszystkie dane (JSONL)
        all_file = self.output_folder / "all_normalized.jsonl"
        with open(all_file, 'w', encoding='utf-8') as f:
            for item in data['all']:
                json.dump(item, f, ensure_ascii=False)
                f.write('\n')
        
        # 2. Modele (JSON)
        if data['models']:
            models_file = self.output_folder / "models_normalized.json"
            with open(models_file, 'w', encoding='utf-8') as f:
                json.dump(data['models'], f, ensure_ascii=False, indent=2)
        
        # 3. Inne (JSON)
        if data['other']:
            other_file = self.output_folder / "other_normalized.json"
            with open(other_file, 'w', encoding='utf-8') as f:
                json.dump(data['other'], f, ensure_ascii=False, indent=2)
        
        # 4. Statystyki szczegółowe
        stats_file = self.output_folder / "normalization_stats.json"
        detailed_stats = self._calculate_stats(data)
        detailed_stats['filtering_stats'] = {
            'total_loaded': stats['total_loaded'],
            'filtered_out': stats['filtered_out'],
            'kept': stats['kept'],
            'kept_percentage': (stats['kept'] / max(stats['total_loaded'], 1)) * 100
        }
        
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(detailed_stats, f, ensure_ascii=False, indent=2)
        
        # 5. Podsumowanie tekstowe
        summary_file = self.output_folder / "summary.txt"
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write(f"=== PODSUMOWANIE NORMALIZACJI Z FILTROWANIEM ===\n")
            f.write(f"Data: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Folder źródłowy: {self.input_folder.name}\n")
            f.write(f"\n📊 STATYSTYKI FILTROWANIA:\n")
            f.write(f"   Załadowano: {stats['total_loaded']} stron\n")
            f.write(f"   Odrzucono: {stats['filtered_out']} stron\n")
            f.write(f"   Zachowano: {stats['kept']} stron ({detailed_stats['filtering_stats']['kept_percentage']:.1f}%)\n")
            f.write(f"\n📄 ZNORMALIZOWANE DANE:\n")
            f.write(f"   Stron ogółem: {len(data['all'])}\n")
            f.write(f"   Stron z modelami: {len(data['models'])}\n")
            f.write(f"   Innych stron: {len(data['other'])}\n")
            
            if data['all']:
                avg_chars = sum(len(item.get('content', '')) for item in data['all']) / len(data['all'])
                f.write(f"   Średnio znaków/stronę: {avg_chars:.0f}\n")
            
            if 'models_found' in detailed_stats and detailed_stats['models_found']:
                f.write(f"\n🚗 WYKRYTE MODELE:\n")
                for model, count in list(detailed_stats['models_found'].items())[:15]:
                    f.write(f"   {model}: {count} stron\n")
            
            if 'price_stats' in detailed_stats:
                f.write(f"\n💰 STATYSTYKI CEN:\n")
                ps = detailed_stats['price_stats']
                f.write(f"   Znalezionych cen: {ps['total_found']}\n")
                f.write(f"   Średnia cena: {ps['avg_formatted']}\n")
                f.write(f"   Zakres: {ps['min_formatted']} - {ps['max_formatted']}\n")
            
            f.write(f"\n⚠️  ODRZUCONE TYPY STRON:\n")
            f.write(f"   • Lorem Ipsum / placeholder\n")
            f.write(f"   • Menu / nawigacja\n")
            f.write(f"   • Za krótkie (<300 znaków) bez wartościowych info\n")
            f.write(f"   • Tylko lista modeli bez kontekstu\n")
            f.write(f"   • Strony regulaminu / polityki prywatności\n")
    
    def _calculate_stats(self, data):
        """Oblicza statystyki normalizacji"""
        stats = {
            'total_pages': len(data['all']),
            'model_pages': len(data['models']),
            'other_pages': len(data['other']),
            'total_characters': sum(len(item.get('content', '')) for item in data['all']),
            'avg_characters_per_page': 0,
            'models_found': [],
            'normalized_at': datetime.now().isoformat()
        }
        
        if data['all']:
            stats['avg_characters_per_page'] = stats['total_characters'] / len(data['all'])
        
        # Zebrane modele
        all_models = []
        for item in data['all']:
            # Sprawdź w specyfikacjach
            specs = item.get('specifications', {})
            if 'model_info' in specs and 'models' in specs['model_info']:
                all_models.extend(specs['model_info']['models'])
            
            # Sprawdź w treści
            content = item.get('content', '').lower()
            # Proste wykrywanie modeli w treści
            model_patterns = [
                r'bmw\s+([xmi]\d+)',
                r'([xmi]\d+)\s+(?:bmw|model|wersja)',
                r'seria\s+(\d+)'
            ]
            
            for pattern in model_patterns:
                matches = re.findall(pattern, content, re.IGNORECASE)
                for match in matches:
                    model = match.upper() if isinstance(match, str) else match[0].upper()
                    if model not in all_models:
                        all_models.append(model)
        
        if all_models:
            model_counts = Counter(all_models)
            stats['models_found'] = dict(model_counts.most_common(20))
        
        # Statystyki cen
        all_prices = []
        for item in data['all']:
            specs = item.get('specifications', {})
            if 'prices' in specs:
                for price in specs['prices']:
                    if isinstance(price, dict) and 'amount' in price:
                        all_prices.append(price['amount'])
        
        if all_prices:
            stats['price_stats'] = {
                'total_found': len(all_prices),
                'min': min(all_prices),
                'max': max(all_prices),
                'avg': sum(all_prices) / len(all_prices),
                'min_formatted': f"{min(all_prices):,.0f} zł".replace(',', ' '),
                'max_formatted': f"{max(all_prices):,.0f} zł".replace(',', ' '),
                'avg_formatted': f"{sum(all_prices) / len(all_prices):,.0f} zł".replace(',', ' ')
            }
        
        return stats


def normalize_latest_data():
    """Normalizuje najnowsze dane z filtrowaniem"""
    output_base = Path(r"C:\Users\hellb\Documents\Chatbot_BMW\RAG\output")
    
    # Znajdź najnowszy folder z danymi (oprócz już znormalizowanych)
    all_folders = [f for f in output_base.iterdir() 
                   if f.is_dir() and f.name.startswith("bmw_complete")]
    
    if not all_folders:
        print("❌ Nie znaleziono folderów z danymi!")
        return None
    
    # Znajdź folder, który NIE ma jeszcze wersji znormalizowanej z filtrami
    latest_folder = None
    for folder in sorted(all_folders, reverse=True):
        normalized_name = f"{folder.name}_normalized_filtered"
        if not (output_base / normalized_name).exists():
            latest_folder = folder
            break
    
    if not latest_folder:
        # Jeśli wszystkie już mają znormalizowane wersje, użyj najnowszego
        latest_folder = sorted(all_folders)[-1]
        print(f"⚠️  Wszystkie foldery już mają znormalizowane wersje")
        print(f"   Używam najnowszego: {latest_folder.name}")
    
    print(f"📁 Normalizuję dane z: {latest_folder.name}")
    
    # Uruchom normalizację z filtrowaniem
    normalizer = DataNormalizer(latest_folder)
    normalized_data = normalizer.normalize_data()
    
    if normalized_data:
        print(f"\n🎯 WYNIK FILTROWANIA:")
        print(f"   📊 Zachowano {len(normalized_data['all'])} stron")
        
        # Pokaż przykłady zachowanych stron
        if normalized_data['all']:
            print(f"\n📋 PRZYKŁADOWE ZACHOWANE STRONY:")
            for i, item in enumerate(normalized_data['all'][:3]):
                print(f"\n{i+1}. {item.get('title', 'Brak tytułu')[:80]}")
                print(f"   URL: {item.get('url', '')[:80]}...")
                print(f"   Znaków: {len(item.get('content', ''))}")
                
                if item.get('is_model_page'):
                    print(f"   🚗 STRONA MODELU")
                
                specs = item.get('specifications', {})
                if 'prices' in specs and specs['prices']:
                    print(f"   💰 Ceny: {[p.get('formatted', '?') for p in specs['prices'][:2]]}")
    
    return normalized_data


if __name__ == "__main__":
    print("=" * 70)
    print("🔧 NORMALIZACJA DANYCH BMW Z FILTROWANIEM")
    print("=" * 70)
    print("\n📁 Szukam najnowszych danych...")
    
    normalized_data = normalize_latest_data()
    
    if normalized_data:
        print(f"\n" + "=" * 70)
        print("✅ NORMALIZACJA ZAKOŃCZONA")
        print("=" * 70)
        
        # Pokaż gdzie zapisano dane
        output_base = Path(r"C:\Users\hellb\Documents\Chatbot_BMW\RAG\output")
        latest_folders = [f for f in output_base.iterdir() 
                         if f.is_dir() and "_normalized_filtered" in f.name]
        
        if latest_folders:
            latest_output = sorted(latest_folders)[-1]
            print(f"\n📁 Znormalizowane dane zapisane w:")
            print(f"   {latest_output}")
            
            # Pokaż utworzone pliki
            output_files = list(latest_output.glob("*"))
            print(f"\n📄 Utworzone pliki:")
            for file in output_files:
                size_kb = file.stat().st_size / 1024
                print(f"   - {file.name} ({size_kb:.1f} KB)")