import asyncio
import aiohttp
import aiofiles
import json
import re
import hashlib
import time
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Set, Optional, Tuple, Any
from urllib.parse import urlparse, urljoin, urldefrag
from dataclasses import dataclass, field
from collections import deque, Counter, defaultdict
import ssl
import certifi

# Web imports
from bs4 import BeautifulSoup
import requests

# =========================
# GLOBAL CONFIG
# =========================

@dataclass
class UltraConfig:
    """Konfiguracja crawlera BMW"""
    base_dir: Path = Path(__file__).resolve().parent.parent.parent.parent
    output_dir: Path = base_dir / "RAG" / "output" / f"bmw_ultra_{datetime.now().strftime('%d_%m_%Y_%H%M%S')}"
    
    start_urls: List[str] = field(default_factory=lambda: [
        "https://www.bmw.pl",
        "https://www.bmw.pl/pl/all-models.html",
        "https://www.bmw.pl/pl/elektromobilnosc.html",
    ])
    
    max_pages: int = 50
    request_delay: float = 2.0
    
    min_quality_score: float = 0.2
    
    skip_extensions: Dict[str, str] = field(default_factory=lambda: {
        '.pdf': 'PDF Document',
        '.jpg': 'Image', '.jpeg': 'Image', '.png': 'Image',
        '.gif': 'Image', '.svg': 'Image', '.mp4': 'Video',
        '.zip': 'Archive', '.rar': 'Archive',
    })
    
    def __post_init__(self):
        self.output_dir.mkdir(parents=True, exist_ok=True)

# =========================
# SIMPLE CRAWLER
# =========================

class SimpleBMWCrawler:
    """Prosty crawler BMW z requests"""
    
    def __init__(self, config: UltraConfig):
        self.config = config
        self.visited_urls = set()
        self.to_visit = deque(config.start_urls)
        self.scraped_data = []
        
        # Stats
        self.pages_scraped = 0
        self.errors = 0
        self.start_time = time.time()
        
        # Setup session
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'pl,en-US;q=0.7,en;q=0.3',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
        })
        
        # SSL verification
        self.session.verify = True
        
    def run(self):
        """Główna funkcja crawlowania"""
        print("🚀 Uruchamianie crawlera BMW...")
        print(f"📁 Output directory: {self.config.output_dir}")
        print(f"🎯 Max pages: {self.config.max_pages}")
        print("=" * 60)
        
        while self.to_visit and self.pages_scraped < self.config.max_pages:
            url = self.to_visit.popleft()
            
            if url in self.visited_urls:
                continue
            
            try:
                print(f"📄 Pobieranie: {url}")
                
                # Dodaj timeout i retry
                response = self.session.get(
                    url, 
                    timeout=15,
                    allow_redirects=True,
                    verify=True
                )
                
                if response.status_code == 200:
                    html = response.text
                    
                    # Parsuj
                    soup = BeautifulSoup(html, 'html.parser')
                    
                    # Wyciągnij dane
                    data = self.extract_data(soup, url, html)
                    
                    if data:
                        self.scraped_data.append(data)
                        self.pages_scraped += 1
                        print(f"✅ Zebrano: {url} (strona {self.pages_scraped})")
                        
                        # Znajdź linki
                        self.extract_links(soup, url)
                    
                    self.visited_urls.add(url)
                    
                    # Rate limiting
                    time.sleep(self.config.request_delay)
                    
                else:
                    print(f"❌ Błąd {response.status_code}: {url}")
                    self.errors += 1
                    
            except requests.exceptions.SSLError as e:
                print(f"🔒 Błąd SSL: {url} - {e}")
                print("⚠️ Próbuję bez weryfikacji SSL...")
                try:
                    # Próbuj bez SSL verification
                    response = requests.get(
                        url,
                        timeout=15,
                        verify=False,
                        headers={
                            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
                        }
                    )
                    
                    if response.status_code == 200:
                        html = response.text
                        soup = BeautifulSoup(html, 'html.parser')
                        data = self.extract_data(soup, url, html)
                        
                        if data:
                            self.scraped_data.append(data)
                            self.pages_scraped += 1
                            print(f"✅ Zebrano (bez SSL): {url}")
                            self.extract_links(soup, url)
                        
                        self.visited_urls.add(url)
                        time.sleep(self.config.request_delay)
                    else:
                        print(f"❌ Błąd {response.status_code} bez SSL: {url}")
                        self.errors += 1
                        
                except Exception as e2:
                    print(f"❌ Błąd przy próbie bez SSL: {url} - {e2}")
                    self.errors += 1
                    
            except Exception as e:
                print(f"❌ Błąd: {url} - {type(e).__name__}: {e}")
                self.errors += 1
        
        # Zapisz dane
        self.save_data()
        self.print_summary()
    
    def extract_data(self, soup: BeautifulSoup, url: str, html: str) -> Optional[Dict]:
        """Wyciąga dane ze strony"""
        try:
            # Tytuł
            title = soup.title.string if soup.title else ""
            
            # Treść
            text_elements = soup.find_all(['p', 'h1', 'h2', 'h3', 'h4'])
            content = ' '.join([elem.get_text(strip=True) for elem in text_elements[:50]])
            
            # Sprawdź jakość
            word_count = len(content.split())
            if word_count < 50:
                print(f"⏭️ Za mało tekstu ({word_count} słów): {url}")
                return None
            
            # Wyciągnij linki do specyfikacji
            specs_links = []
            for link in soup.find_all('a', href=True)[:30]:
                href = link['href']
                if 'dane-techniczne' in href or 'technical-data' in href or 'specyfikacja' in href:
                    if href.startswith('/'):
                        href = urljoin(url, href)
                    if 'bmw.pl' in href:
                        specs_links.append(href)
            
            data = {
                'url': url,
                'title': title[:200],
                'content': content[:5000],
                'word_count': word_count,
                'specs_links': list(set(specs_links))[:10],
                'scraped_at': datetime.now().isoformat(),
            }
            
            # Sprawdź czy to strona z modelem
            if self.is_model_page(content, url):
                data['is_model_page'] = True
                data['model_info'] = self.extract_model_info(content, url)
            
            return data
            
        except Exception as e:
            print(f"⚠️ Błąd ekstrakcji: {e}")
            return None
    
    def extract_model_info(self, content: str, url: str) -> Dict:
        """Wyciąga informacje o modelu"""
        info = {}
        content_lower = content.lower()
        url_lower = url.lower()
        
        # Wykryj model z URL
        models = ['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 
                  'i3', 'i4', 'ix', 'i7', 'i5',
                  'm2', 'm3', 'm4', 'm5', 'm8',
                  'seria 1', 'seria 2', 'seria 3', 'seria 4', 
                  'seria 5', 'seria 6', 'seria 7', 'seria 8']
        
        for model in models:
            if model in url_lower or model in content_lower:
                info['detected_model'] = model
                break
        
        # Wykryj paliwo
        if 'diesel' in content_lower:
            info['fuel_type'] = 'diesel'
        elif 'benzyna' in content_lower or 'benzynowy' in content_lower:
            info['fuel_type'] = 'benzyna'
        elif 'elektryczny' in content_lower or 'ev' in content_lower:
            info['fuel_type'] = 'elektryczny'
        elif 'hybryda' in content_lower:
            info['fuel_type'] = 'hybryda'
        
        # Wykryj moc
        power_match = re.search(r'(\d{2,4})\s*(?:km|kon[ií])', content, re.IGNORECASE)
        if power_match:
            info['power_hp'] = int(power_match.group(1))
        
        # Wykryj cenę
        price_match = re.search(r'(\d{1,3}(?:\s?\d{3})*[,\d]*)\s*(z[łl]|pln)', content, re.IGNORECASE)
        if price_match:
            amount = price_match.group(1)
            clean_amount = re.sub(r'[^\d,]', '', amount).replace(',', '.')
            try:
                info['price'] = float(clean_amount)
                info['currency'] = price_match.group(2).upper()
            except:
                pass
        
        return info
    
    def is_model_page(self, content: str, url: str) -> bool:
        """Sprawdza czy to strona z modelem"""
        content_lower = content.lower()
        url_lower = url.lower()
        
        model_indicators = ['model', 'silnik', 'moc', 'przyspieszenie', 'wymiary', 'cena']
        indicators_count = sum(1 for indicator in model_indicators if indicator in content_lower)
        
        return indicators_count >= 3 or 'model' in url_lower or 'seria' in url_lower
    
    def extract_links(self, soup: BeautifulSoup, base_url: str):
        """Wyciąga linki ze strony"""
        new_links = 0
        
        for link in soup.find_all('a', href=True)[:50]:
            href = link['href']
            
            # Normalizuj URL
            if href.startswith('/'):
                href = urljoin(base_url, href)
            elif href.startswith('http'):
                pass  # już pełny URL
            else:
                continue
            
            # Sprawdź czy to BMW
            if 'bmw.pl' not in href:
                continue
            
            # Sprawdź rozszerzenia
            if any(href.lower().endswith(ext) for ext in self.config.skip_extensions.keys()):
                continue
            
            # Dodaj do kolejki jeśli nie odwiedzony i nie w kolejce
            if (href not in self.visited_urls and 
                href not in self.to_visit and
                len(href) < 200):  # Unikaj bardzo długich URL-i
                
                # Priorytet dla stron z modelami
                if any(keyword in href.lower() for keyword in ['model', 'seria', 'x', 'i', 'm']):
                    self.to_visit.appendleft(href)  # Wyższy priorytet
                else:
                    self.to_visit.append(href)
                
                new_links += 1
        
        if new_links > 0:
            print(f"🔗 Znaleziono {new_links} nowych linków")
    
    def save_data(self):
        """Zapisuje zebrane dane"""
        if not self.scraped_data:
            print("⚠️ Brak danych do zapisania!")
            return
        
        output_file = self.config.output_dir / "scraped_data.jsonl"
        
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                for item in self.scraped_data:
                    json.dump(item, f, ensure_ascii=False)
                    f.write('\n')
            
            print(f"💾 Dane zapisane do: {output_file}")
            print(f"📊 Zebrano {len(self.scraped_data)} stron")
            
        except Exception as e:
            print(f"❌ Błąd zapisu: {e}")
    
    def print_summary(self):
        """Wyświetla podsumowanie"""
        elapsed = time.time() - self.start_time
        elapsed_hours = elapsed / 3600
        
        print("\n" + "=" * 60)
        print("🎉 BMW CRAWLER - PODSUMOWANIE")
        print("=" * 60)
        
        print(f"✅ Stron zebranych: {self.pages_scraped}")
        print(f"❌ Błędy: {self.errors}")
        print(f"⏱️ Czas pracy: {elapsed_hours:.2f} godzin")
        
        if self.pages_scraped > 0 and elapsed_hours > 0:
            pages_per_hour = self.pages_scraped / elapsed_hours
            print(f"📈 Wydajność: {pages_per_hour:.1f} stron/godzinę")
        
        # Analiza zebranych danych
        model_pages = sum(1 for d in self.scraped_data if d.get('is_model_page', False))
        print(f"🚗 Stron z modelami: {model_pages}")
        
        # Wykryte modele
        models = []
        for data in self.scraped_data:
            if 'model_info' in data and 'detected_model' in data['model_info']:
                models.append(data['model_info']['detected_model'])
        
        if models:
            from collections import Counter
            model_counts = Counter(models)
            print("\n🚗 WYKRYTE MODELE:")
            for model, count in model_counts.most_common():
                print(f"  {model}: {count} stron")
        
        print("=" * 60)

# =========================
# TEST FUNCTION
# =========================

def test_connection():
    """Testuje połączenie z BMW"""
    print("🔧 Testowanie połączenia z BMW...")
    
    test_urls = [
        "https://www.bmw.pl",
        "https://www.google.com"  # Do porównania
    ]
    
    for url in test_urls:
        try:
            print(f"\n🔗 Testowanie: {url}")
            response = requests.get(url, timeout=10, verify=False)
            print(f"✅ Status: {response.status_code}")
            print(f"📏 Rozmiar: {len(response.text)} bajtów")
            
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')
                title = soup.title.string if soup.title else "Brak tytułu"
                print(f"📄 Tytuł: {title[:50]}...")
                
        except Exception as e:
            print(f"❌ Błąd: {type(e).__name__}: {e}")
    
    print("\n" + "=" * 60)

# =========================
# MAIN
# =========================

def main():
    """Główna funkcja"""
    print("=" * 60)
    print("🚗 BMW CRAWLER - ZBIERANIE SPECYFIKACJI")
    print("=" * 60)
    
    # Test połączenia
    test_connection()
    
    print("\n" + "=" * 60)
    print("🚀 URUCHAMIANIE CRAWLERA...")
    print("=" * 60)
    
    # Wyłącz warnings dla SSL
    import urllib3
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    
    # Stwórz konfigurację
    config = UltraConfig()
    
    # Mniejsze wartości na test
    config.max_pages = 20  # Tylko 20 stron na początek
    config.request_delay = 3.0  # 3 sekundy między requestami
    
    # Start z mniejszą liczbą URL-i
    config.start_urls = [
        "https://www.bmw.pl",
        "https://www.bmw.pl/pl/all-models.html",
    ]
    
    # Stwórz i uruchom crawlera
    crawler = SimpleBMWCrawler(config)
    
    try:
        crawler.run()
    except KeyboardInterrupt:
        print("\n\n🛑 Crawler zatrzymany przez użytkownika")
        crawler.save_data()
    except Exception as e:
        print(f"\n❌ Nieoczekiwany błąd: {e}")
        import traceback
        traceback.print_exc()
        crawler.save_data()

if __name__ == "__main__":
    # Sprawdź zależności
    try:
        import requests
        from bs4 import BeautifulSoup
    except ImportError:
        print("❌ Brakujące zależności!")
        print("📦 Zainstaluj: pip install requests beautifulsoup4 lxml")
        sys.exit(1)
    
    # Uruchom
    main()