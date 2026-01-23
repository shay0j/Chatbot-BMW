import json
import re
from pathlib import Path
from datetime import datetime

class DataNormalizer:
    """Normalizuje i czyści zebrane dane"""
    
    def __init__(self, input_folder):
        self.input_folder = Path(input_folder)
        self.output_folder = self.input_folder.parent / f"{self.input_folder.name}_normalized"
        self.output_folder.mkdir(exist_ok=True)
        
    def normalize_text(self, text):
        """Normalizuje tekst"""
        if not text:
            return ""
        
        # 1. Usuń nadmiarowe białe znaki
        text = re.sub(r'\s+', ' ', text)
        
        # 2. Usuń specjalne znaki formatowania
        text = re.sub(r'[\r\n\t]+', ' ', text)
        
        # 3. Usuń ciągi powtarzających się znaków
        text = re.sub(r'([.!?])\1+', r'\1', text)
        
        # 4. Normalizuj cudzysłowy
        text = text.replace('"', "'")
        
        # 5. Usuń początkowe/końcowe spacje
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
        """Główna funkcja normalizacji"""
        input_files = {
            'all': self.input_folder / "all_data.jsonl",
            'models': self.input_folder / "models_data.json",
            'other': self.input_folder / "other_data.json"
        }
        
        normalized_data = {
            'all': [],
            'models': [],
            'other': []
        }
        
        for data_type, input_file in input_files.items():
            if not input_file.exists():
                print(f"⚠️ Brak pliku: {input_file}")
                continue
            
            print(f"📄 Normalizuję: {data_type}")
            
            if data_type == 'all':
                # JSONL format
                with open(input_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        item = json.loads(line)
                        normalized_item = self._normalize_item(item)
                        normalized_data['all'].append(normalized_item)
            else:
                # JSON format
                with open(input_file, 'r', encoding='utf-8') as f:
                    items = json.load(f)
                    for item in items:
                        normalized_item = self._normalize_item(item)
                        normalized_data[data_type].append(normalized_item)
        
        # Zapisz znormalizowane dane
        self._save_normalized_data(normalized_data)
        
        print(f"\n✅ Znormalizowane dane zapisane w: {self.output_folder}")
        
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
        
        # Dodaj unique_id dla łatwiejszego referencjonowania
        normalized['unique_id'] = f"bmw_{abs(hash(normalized.get('url', ''))) % 1000000:06d}"
        
        # Usuń puste pola
        normalized = {k: v for k, v in normalized.items() if v not in [None, '', [], {}]}
        
        return normalized
    
    def _save_normalized_data(self, data):
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
        
        # 4. Statystyki
        stats = self._calculate_stats(data)
        stats_file = self.output_folder / "normalization_stats.json"
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
        
        # 5. Podsumowanie
        summary_file = self.output_folder / "summary.txt"
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write(f"=== PODSUMOWANIE NORMALIZACJI ===\n")
            f.write(f"Data: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Folder źródłowy: {self.input_folder.name}\n")
            f.write(f"Stron ogółem: {stats['total_pages']}\n")
            f.write(f"Stron z modelami: {stats['model_pages']}\n")
            f.write(f"Innych stron: {stats['other_pages']}\n")
            f.write(f"Średnio znaków/stronę: {stats['avg_characters_per_page']:.0f}\n")
            
            if stats['models_found']:
                f.write("\nNajczęstsze modele:\n")
                for model, count in list(stats['models_found'].items())[:10]:
                    f.write(f"  {model}: {count} stron\n")
                    
            if 'price_stats' in stats:
                f.write("\nStatystyki cen:\n")
                price_stats = stats['price_stats']
                f.write(f"  Znalezionych cen: {price_stats['total_found']}\n")
                f.write(f"  Średnia cena: {price_stats['avg_formatted']}\n")
                f.write(f"  Zakres: {price_stats['min']:,.0f} - {price_stats['max']:,.0f} zł\n")
    
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
        for item in data['models']:
            # Sprawdź w specyfikacjach
            specs = item.get('specifications', {})
            if 'model_info' in specs and 'models' in specs['model_info']:
                all_models.extend(specs['model_info']['models'])
            # Sprawdź też w głównych polach
            if 'detected_models' in item:
                all_models.extend(item['detected_models'])
        
        if all_models:
            from collections import Counter
            model_counts = Counter(all_models)
            stats['models_found'] = dict(model_counts.most_common(20))
        
        # Statystyki cen
        all_prices = []
        for item in data['models']:
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
                'avg_formatted': f"{sum(all_prices) / len(all_prices):,.0f} zł".replace(',', ' ')
            }
        
        return stats

def find_latest_bmw_folder():
    """Znajduje najnowszy folder z danymi BMW"""
    output_base = Path(r"C:\Users\hellb\Documents\Chatbot_BMW\RAG\output")
    folders = sorted([f for f in output_base.iterdir() if f.is_dir() and f.name.startswith("bmw")])
    
    if not folders:
        print("❌ Nie znaleziono folderów z danymi!")
        return None
    
    latest_folder = folders[-1]
    print(f"📁 Znaleziono najnowszy folder: {latest_folder.name}")
    
    # Sprawdź jakie pliki są dostępne
    files = list(latest_folder.glob("*.json*"))
    files.extend(latest_folder.glob("*.txt"))
    
    print(f"📊 Znaleziono {len(files)} plików:")
    for file in files:
        size_kb = file.stat().st_size / 1024
        print(f"   - {file.name} ({size_kb:.1f} KB)")
    
    return latest_folder

def normalize_latest_data():
    """Normalizuje najnowsze dane"""
    latest_folder = find_latest_bmw_folder()
    
    if not latest_folder:
        return None
    
    print(f"\n📁 Normalizuję najnowsze dane z: {latest_folder.name}")
    
    normalizer = DataNormalizer(latest_folder)
    normalized_data = normalizer.normalize_data()
    
    print(f"\n🎯 PODSUMOWANIE NORMALIZACJI:")
    print(f"   Znormalizowano: {len(normalized_data['all'])} stron")
    print(f"   Modele: {len(normalized_data['models'])} stron")
    print(f"   Inne: {len(normalized_data['other'])} stron")
    
    return normalized_data

# Uruchom bezpośrednio
if __name__ == "__main__":
    print("=" * 60)
    print("🔧 NORMALIZACJA DANYCH BMW - START")
    print("=" * 60)
    
    # Znajdź najnowszy folder
    latest_folder = find_latest_bmw_folder()
    
    if latest_folder:
        # Zapytaj czy normalizować
        response = input("\n🎯 Czy chcesz znormalizować te dane? (t/n): ")
        
        if response.lower() == 't':
            normalized_data = normalize_latest_data()
            
            # Pokaż przykładowe znormalizowane dane
            if normalized_data and normalized_data['models']:
                print(f"\n" + "=" * 60)
                print("📋 PRZYKŁAD ZNORMALIZOWANYCH DANE MODELU:")
                print("=" * 60)
                sample = normalized_data['models'][0]
                print(f"   ID: {sample.get('unique_id', 'N/A')}")
                print(f"   Tytuł: {sample.get('title', '')[:80]}...")
                print(f"   URL: {sample.get('url', '')}")
                
                specs = sample.get('specifications', {})
                if 'model_info' in specs:
                    print(f"   Modele: {specs['model_info'].get('models', [])}")
                if 'engine' in specs and specs['engine']:
                    print(f"   Silnik: {specs['engine']}")
                if 'prices' in specs and specs['prices']:
                    print(f"   Ceny: {[p.get('formatted', 'N/A') for p in specs['prices'][:2]]}")
                
                print(f"\n📁 Znormalizowane dane zapisane w:")
                output_folder = latest_folder.parent / f"{latest_folder.name}_normalized"
                if output_folder.exists():
                    print(f"   {output_folder}")
                    
                    # Pokaż utworzone pliki
                    output_files = list(output_folder.glob("*"))
                    print(f"\n📄 Utworzone pliki:")
                    for file in output_files:
                        size_kb = file.stat().st_size / 1024
                        print(f"   - {file.name} ({size_kb:.1f} KB)")
        else:
            print("❌ Anulowano normalizację.")
    else:
        print("❌ Nie znaleziono danych do normalizacji.")