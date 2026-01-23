from pathlib import Path
import json
from bs4 import BeautifulSoup
import re

BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE_DIR / "RAG" / "output"

# Znajdź najnowszy folder z danymi
folders = [d for d in OUTPUT_DIR.iterdir() if d.is_dir()]
if not folders:
    print("❌ Brak folderów w output/")
    exit(1)

LATEST_FOLDER = max(folders, key=lambda d: d.stat().st_mtime)
print(f"📁 Analizuję: {LATEST_FOLDER.name}")

# Weź kilka przykładowych stron różnych typów
html_files = list(LATEST_FOLDER.rglob("*.html"))

print(f"📊 Znaleziono {len(html_files)} plików HTML")

# Kategorie stron (żeby zrozumieć strukturę)
categories = {
    'model_pages': [],      # Strony konkretnych modeli
    'overview_pages': [],   # Przeglądy
    'config_pages': [],     # Konfiguratory
    'other_pages': []       # Inne
}

for html_file in html_files[:20]:  # Analizuj pierwsze 20
    with open(html_file, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()
    
    soup = BeautifulSoup(content, 'html.parser')
    
    # Analizuj URL
    url = html_file.stem.replace('_', '/').replace('https:', 'https://').replace('www.bmw.pl', 'https://www.bmw.pl')
    
    # Analizuj tytuł
    title = soup.find('title')
    title_text = title.get_text(strip=True) if title else "Brak tytułu"
    
    # Sprawdź czy to strona modelu
    is_model_page = any(x in title_text.lower() for x in ['bmw', 'x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 
                                                          'seria 1', 'seria 2', 'seria 3', 'seria 4', 'seria 5',
                                                          'i3', 'i4', 'i5', 'i7', 'i8', 'ix'])
    
    # Sprawdź zawartość
    all_text = soup.get_text()[:2000]  # Pierwsze 2000 znaków
    
    print(f"\n{'='*80}")
    print(f"📄 Plik: {html_file.name[:50]}...")
    print(f"🌐 URL: {url[:100]}...")
    print(f"🏷️  Tytuł: {title_text}")
    
    # Szukaj kluczowych sekcji
    print(f"\n🔍 SZUKAM DANYCH TECHNICZNYCH:")
    
    # 1. Szukaj danych technicznych (common patterns)
    tech_patterns = [
        r'(\d{2,4})\s*(KM|kW|PS)',  # Moc
        r'zasi[ęe]g.*?(\d{2,4})\s*(km|kilometr)',  # Zasięg
        r'(SUV|Sedan|Coupe|Cabrio|Roadster|Kombi|Hatchback)',  # Typ nadwozia
        r'(xDrive|sDrive|4x4|AWD|RWD|FWD)',  # Napęd
        r'(\d+[\.,]?\d*)\s*(l/100km|km/l)',  # Spalanie
    ]
    
    for pattern in tech_patterns:
        matches = re.findall(pattern, all_text, re.IGNORECASE)
        if matches:
            print(f"   ✓ {pattern[:30]}...: {matches[:3]}")  # Pierwsze 3 wyniki
    
    # 2. Szukaj tabel z danymi
    tables = soup.find_all('table')
    print(f"   📊 Tabele: {len(tables)}")
    
    # 3. Szukaj list z danymi
    lists = soup.find_all(['ul', 'ol'])
    list_items = sum(len(list.find_all('li')) for list in lists)
    print(f"   📋 Listy: {len(lists)} ({list_items} items)")
    
    # 4. Szukaj sekcji z danymi technicznymi
    tech_sections = soup.find_all(['div', 'section'], 
                                  text=re.compile(r'dane techniczn|specyfikacja|technical data', re.IGNORECASE))
    print(f"   🔧 Sekcje techniczne: {len(tech_sections)}")
    
    # 5. Zobacz strukturę HTML (nagłówki)
    print(f"\n📑 STRUKTURA STRONY:")
    for i, header in enumerate(soup.find_all(['h1', 'h2', 'h3'])[:5]):
        print(f"   {header.name}: {header.get_text()[:80]}...")
    
    # 6. Znajdź meta tags z danymi
    print(f"\n🏷️  META DANE:")
    meta_tags = soup.find_all('meta')
    for meta in meta_tags[:10]:  # Pierwsze 10
        if meta.get('name') and any(kw in meta.get('name', '').lower() for kw in ['description', 'keywords', 'model']):
            print(f"   {meta.get('name')}: {meta.get('content', '')[:100]}...")
    
    # Klasyfikuj stronę
    if 'dane-techniczne' in url or 'technical-data' in url:
        categories['model_pages'].append((html_file, title_text))
    elif 'all-models' in url or 'przeglad' in url:
        categories['overview_pages'].append((html_file, title_text))
    elif 'konfigurator' in url or 'configure' in url:
        categories['config_pages'].append((html_file, title_text))
    else:
        categories['other_pages'].append((html_file, title_text))

# Podsumowanie
print(f"\n{'='*80}")
print("📊 PODSUMOWANIE KATEGORII:")
for cat, pages in categories.items():
    print(f"  {cat}: {len(pages)} stron")
    if pages:
        print(f"     Przykłady:")
        for page, title in pages[:3]:
            print(f"       - {title[:60]}...")

# Zapisz przykładowe strony do dalszej analizy
sample_dir = BASE_DIR / "RAG" / "sample_pages"
sample_dir.mkdir(exist_ok=True)

print(f"\n💾 Zapisuję przykładowe strony do: {sample_dir}")

for i, (html_file, title) in enumerate(categories['model_pages'][:3]):
    with open(html_file, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()
    
    sample_file = sample_dir / f"model_page_{i}.html"
    sample_file.write_text(content, encoding='utf-8')
    print(f"  ✅ {sample_file.name} - {title[:50]}...")

print(f"\n🎯 NASTĘPNE KROKI:")
print("1. Przeanalizujemy strukturę przykładowych stron")
print("2. Napiszemy scraper dla każdego typu strony")
print("3. Będziemy wyciągać: model_name, typ_nadwozia, moc_silnika, zasieg, naped, cena")