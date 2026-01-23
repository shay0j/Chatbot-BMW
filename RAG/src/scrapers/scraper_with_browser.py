from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException
from bs4 import BeautifulSoup
import json
import time
import re
from urllib.parse import urljoin, urlparse
from pathlib import Path
from datetime import datetime
import logging
from collections import deque

class CompleteBMWCrawler:
    """Kompletny crawler BMW zbierający WSZYSTKIE treści z naciskiem na modele"""
    
    def __init__(self, headless=False):  # Dodajemy opcję headless
        # Setup logging
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)
        
        # STAŁA ŚCIEŻKA OUTPUTU
        self.output_dir = Path(r"C:\Users\hellb\Documents\Chatbot_BMW\RAG\output")
        
        # Stwórz timestamp folder
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = self.output_dir / f"bmw_complete_{timestamp}"
        
        self.logger.info(f"📁 Output directory: {self.output_dir}")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup Chrome - WYŁĄCZAMY HEADLESS dla BMW
        options = webdriver.ChromeOptions()
        options.add_argument('--disable-blink-features=AutomationControlled')
        options.add_experimental_option("excludeSwitches", ["enable-automation"])
        options.add_experimental_option('useAutomationExtension', False)
        options.add_argument('user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36')
        options.add_argument('--disable-dev-shm-usage')
        options.add_argument('--no-sandbox')
        options.add_argument('--window-size=1920,1080')
        options.add_argument('--start-maximized')
        
        # DODAJEMY WIĘCEJ OPCJI PRZEZPIERANIA WYKRYWANIA
        options.add_argument('--disable-web-security')
        options.add_argument('--allow-running-insecure-content')
        
        # Tylko jeśli headless=True - ale dla BMW polecam NORMALNĄ przeglądarkę
        if headless:
            self.logger.warning("⚠️ Headless mode dla BMW może nie działać!")
            options.add_argument('--headless=new')
        else:
            self.logger.info("✅ Używam normalnej przeglądarki (lepsza kompatybilność z BMW)")
        
        # Dodajemy argumenty żeby uniknąć blokowania przez cloudflare
        options.add_argument("--disable-blink-features=AutomationControlled") 
        options.add_experimental_option("excludeSwitches", ["enable-automation"])
        options.add_experimental_option('useAutomationExtension', False)
        
        self.driver = webdriver.Chrome(options=options)
        self.driver.set_page_load_timeout(30)
        
        # Wykonaj JavaScript żeby ominąć detekcję
        self.driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
        
        # Stats and data
        self.visited_urls = set()
        self.to_visit = deque()
        self.scraped_data = []
        self.model_pages = []
        self.other_pages = []
        
        # Priorytetowe URL-e - POPRAWIONE (część linków mogła się zmienić)
        self.priority_urls = [
            "https://www.bmw.pl/pl/index.html",  # Zaczynamy od głównej strony
            "https://www.bmw.pl/pl/ssl/models.html",  # Lista modeli
            "https://www.bmw.pl/pl/all-models.html",  # Wszystkie modele
            "https://www.bmw.pl/pl/topics/fascination-bmw/bmw-models.html",  # Modele BMW
            "https://www.bmw.pl/pl/elektromobilnosc/all-bmw-electric-cars.html",  # Elektryczne
            "https://www.bmw.pl/pl/bmw-m/m-models.html",  # M modele
            "https://www.bmw.pl/pl/bmw-ix.html",  # iX
            "https://www.bmw.pl/pl/bmw-i4.html",  # i4
            "https://www.bmw.pl/pl/nowosci.html",  # Nowości
            "https://www.bmw.pl/pl/serwis.html",  # Serwis
            "https://www.bmw.pl/pl/bmw-m.html",  # M
        ]
        
        # Wzorce dla modeli
        self.model_patterns = {
            # Seria 1-8 - tylko pełne nazwy
            'Seria 1': r'\b1\s*Seria\b|\b1\s*Series\b',
            'Seria 2': r'\b2\s*Seria\b|\b2\s*Series\b',
            'Seria 3': r'\b3\s*Seria\b|\b3\s*Series\b',
            'Seria 4': r'\b4\s*Seria\b|\b4\s*Series\b',
            'Seria 5': r'\b5\s*Seria\b|\b5\s*Series\b',
            'Seria 7': r'\b7\s*Seria\b|\b7\s*Series\b',
            'Seria 8': r'\b8\s*Seria\b|\b8\s*Series\b',
            
            # X Series - tylko jako osobne słowa
            'X1': r'\bX1\b',
            'X2': r'\bX2\b',
            'X3': r'\bX3\b',
            'X4': r'\bX4\b',
            'X5': r'\bX5\b',
            'X6': r'\bX6\b',
            'X7': r'\bX7\b',
            
            # i Series (elektryczne)
            'i3': r'\bi3\b',
            'i4': r'\bi4\b',
            'iX': r'\biX\b',
            'iX3': r'\biX3\b',
            'iX1': r'\biX1\b',
            'i7': r'\bi7\b',
            'i5': r'\bi5\b',
            
            # M Series
            'M2': r'\bM2\b',
            'M3': r'\bM3\b',
            'M4': r'\bM4\b',
            'M5': r'\bM5\b',
            'M8': r'\bM8\b',
        }
        
        # Keywords dla różnych kategorii
        self.category_keywords = {
            'models': ['model', 'silnik', 'moc', 'przyspieszenie', 'wymiary', 'cena', 'specyfikacja', 'dane techniczne'],
            'electric': ['elektryczny', 'ładowanie', 'bateria', 'zasięg', 'ev', 'phev', 'kwh', 'wltp'],
            'service': ['serwis', 'przegląd', 'naprawa', 'części', 'gwarancja', 'warsztat', 'oryginalne części'],
            'configurator': ['konfigurator', 'wyposażenie', 'opcje', 'pakiety', 'kolory', 'tapicerka'],
            'offers': ['oferta', 'promocja', 'leasing', 'finansowanie', 'raty', 'rabat'],
            'news': ['nowość', 'premiera', 'aktualność', 'informacja', 'wydarzenie', 'wiadomości'],
        }
        
    def normalize_url(self, url, base_url=None):
        """Normalizuje URL"""
        if not url or url.startswith('javascript:') or url.startswith('mailto:'):
            return None
            
        # Usuń fragment
        url = url.split('#')[0].strip()
        
        if not url:
            return None
            
        # Dodaj protokół jeśli brakuje
        if url.startswith('//'):
            url = 'https:' + url
        elif url.startswith('/'):
            if base_url:
                url = urljoin(base_url, url)
            else:
                url = 'https://www.bmw.pl' + url
        elif not url.startswith('http'):
            if base_url:
                url = urljoin(base_url, url)
            else:
                url = 'https://www.bmw.pl/' + url
        
        # Zawsze https
        if url.startswith('http://'):
            url = url.replace('http://', 'https://')
            
        # Usuń parametry śledzące
        url = re.sub(r'\?utm_.*', '', url)
        url = re.sub(r'\?gclid=.*', '', url)
        
        return url.strip('/')
    
    def is_valid_url(self, url):
        """Sprawdza czy URL jest poprawny do crawlowania"""
        if not url:
            return False
            
        # Musi być BMW
        if 'bmw.pl' not in url:
            return False
            
        # Nie może być już odwiedzony
        if url in self.visited_urls:
            return False
            
        # Unikaj niepotrzebnych rozszerzeń
        skip_extensions = ['.pdf', '.jpg', '.jpeg', '.png', '.gif', '.svg', 
                          '.mp4', '.avi', '.mov', '.mp3', '.wav', '.zip', 
                          '.rar', '.7z', '.exe', '.dmg', '.msi']
        
        if any(url.lower().endswith(ext) for ext in skip_extensions):
            return False
            
        # Unikaj social media i tracking
        skip_patterns = [
            r'facebook\.com', r'twitter\.com', r'instagram\.com', 
            r'linkedin\.com', r'youtube\.com', r'google\.com',
            r'/api/', r'/ajax/', r'/rest/', r'/graphql',
            r'/login', r'/register', r'/signin', r'/signup',
            r'/cart', r'/checkout', r'/basket', r'/order',
            r'\.xml$', r'\.json$', r'\.rss$',
        ]
        
        for pattern in skip_patterns:
            if re.search(pattern, url, re.IGNORECASE):
                return False
                
        return True
    
    def extract_model_from_url(self, url):
        """Wyciąga model z URL-a (najbardziej wiarygodne)"""
        url_lower = url.lower()
        
        # Mapowanie URL-i do modeli
        url_patterns = [
            (r'/x1[/-]', 'X1'),
            (r'/x2[/-]', 'X2'),
            (r'/x3[/-]', 'X3'),
            (r'/x4[/-]', 'X4'),
            (r'/x5[/-]', 'X5'),
            (r'/x6[/-]', 'X6'),
            (r'/x7[/-]', 'X7'),
            (r'/i3[/-]', 'i3'),
            (r'/i4[/-]', 'i4'),
            (r'/i5[/-]', 'i5'),
            (r'/i7[/-]', 'i7'),
            (r'/ix[/-]', 'iX'),
            (r'/m2[/-]', 'M2'),
            (r'/m3[/-]', 'M3'),
            (r'/m4[/-]', 'M4'),
            (r'/m5[/-]', 'M5'),
            (r'/m8[/-]', 'M8'),
            (r'/3-seria', 'Seria 3'),
            (r'/5-seria', 'Seria 5'),
            (r'/7-seria', 'Seria 7'),
            (r'/8-seria', 'Seria 8'),
            (r'/1-seria', 'Seria 1'),
            (r'/2-seria', 'Seria 2'),
            (r'/4-seria', 'Seria 4'),
            (r'/6-seria', 'Seria 6'),
            (r'/z4[/-]', 'Z4'),
        ]
        
        for pattern, model in url_patterns:
            if re.search(pattern, url_lower):
                return model
        
        return None
    
    def extract_content(self, soup, url):
        """Wyciąga treść ze strony"""
        try:
            # Tytuł
            title = soup.title.string if soup.title else ""
            
            # Meta description
            meta_desc = soup.find('meta', {'name': 'description'})
            description = meta_desc['content'] if meta_desc else ""
            
            # Główna treść - próbuj różnych selektorów
            content_selectors = [
                'main', 
                'article', 
                '.content', 
                '.main-content',
                '#content', 
                '.article-content',
                '.text-content',
                '.body-content',
                '[role="main"]',
                '.module-text',  # BMW często używa tego
                '.text-module',  # Inny typowy dla BMW
                '.richtext',  # Rich text content
            ]
            
            main_content = None
            for selector in content_selectors:
                try:
                    main_content = soup.select_one(selector)
                    if main_content:
                        break
                except:
                    continue
            
            # Zbierz tekst
            text_parts = []
            
            if main_content:
                # Pobierz tekst z głównego kontenera
                elements = main_content.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'li', 'dt', 'dd', 'span'])
                for elem in elements[:200]:  # Limit
                    text = elem.get_text(strip=True)
                    if 10 < len(text) < 2000:  # Minimum 10 znaków, max 2000
                        text_parts.append(text)
            else:
                # Fallback: wszystkie paragrafy i nagłówki
                elements = soup.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'span'])
                for elem in elements[:150]:
                    text = elem.get_text(strip=True)
                    if 15 < len(text) < 1000:
                        text_parts.append(text)
            
            content = ' '.join(text_parts)
            word_count = len(content.split())
            
            # Jeśli bardzo mało treści, spróbuj inaczej
            if word_count < 30:
                # Spróbuj pobrać cały tekst body
                body = soup.find('body')
                if body:
                    all_text = body.get_text()
                    # Podziel na zdania
                    sentences = re.split(r'[.!?]+', all_text)
                    content = ' '.join([s.strip() for s in sentences[:50] if len(s.strip()) > 20])
                    word_count = len(content.split())
            
            return {
                'title': title[:500],
                'description': description[:500],
                'content': content[:20000],  # Max 20k znaków
                'word_count': word_count
            }
            
        except Exception as e:
            self.logger.error(f"Błąd przy ekstrakcji treści: {e}")
            return {
                'title': '',
                'description': '',
                'content': '',
                'word_count': 0
            }
    
    def detect_models(self, text, url):
        """Wykrywa modele BMW w tekście"""
        detected_models = []
        text_lower = text.lower()
        
        # Najpierw sprawdź URL (najbardziej wiarygodne)
        model_from_url = self.extract_model_from_url(url)
        if model_from_url:
            detected_models.append(model_from_url)
            return detected_models  # Jeśli mamy model z URL, to wystarczy
        
        # Tylko jeśli nie ma modelu z URL, sprawdzaj w tekście
        model_context_patterns = {
            'Seria 1': r'(?:BMW\s+)?1\s*Seria|1\s*Series',
            'Seria 2': r'(?:BMW\s+)?2\s*Seria|2\s*Series',
            'Seria 3': r'(?:BMW\s+)?3\s*Seria|3\s*Series',
            'Seria 5': r'(?:BMW\s+)?5\s*Seria|5\s*Series',
            'Seria 7': r'(?:BMW\s+)?7\s*Seria|7\s*Series',
            'X1': r'(?:BMW\s+)?X1\b',
            'X3': r'(?:BMW\s+)?X3\b',
            'X5': r'(?:BMW\s+)?X5\b',
            'i4': r'(?:BMW\s+)?i4\b',
            'iX': r'(?:BMW\s+)?iX\b',
            'M3': r'(?:BMW\s+)?M3\b',
            'M5': r'(?:BMW\s+)?M5\b',
        }
        
        for model_name, pattern in model_context_patterns.items():
            if re.search(pattern, text, re.IGNORECASE):
                # Sprawdź czy to nie przypadkowe wystąpienie
                # Szukaj w kontekście (w pobliżu słów kluczowych)
                context_window = 50  # znaków do przodu i tyłu
                matches = list(re.finditer(pattern, text, re.IGNORECASE))
                
                for match in matches:
                    start = max(0, match.start() - context_window)
                    end = min(len(text), match.end() + context_window)
                    context = text[start:end].lower()
                    
                    # Sprawdź czy w kontekście są słowa kluczowe BMW
                    bmw_keywords = ['bmw', 'model', 'silnik', 'moc', 'cena', 'specyfikacja']
                    if any(keyword in context for keyword in bmw_keywords):
                        if model_name not in detected_models:
                            detected_models.append(model_name)
                        break  # Jedno wystąpienie wystarczy
        
        return list(set(detected_models))
    
    def categorize_content(self, text, url, detected_models):
        """Kategoryzuje treść"""
        categories = []
        text_lower = text.lower()
        url_lower = url.lower()
        
        # Priorytet: modele
        if detected_models:
            categories.append('models')
        
        # Sprawdź słowa kluczowe
        for category, keywords in self.category_keywords.items():
            if category == 'models' and detected_models:
                continue  # Już dodane
            
            keyword_count = sum(1 for keyword in keywords if keyword in text_lower)
            if keyword_count >= 2:
                categories.append(category)
        
        # Sprawdź URL
        url_categories = {
            'models': ['/models/', '/x', '/i', '/m', '/seria', '/series'],
            'electric': ['/elektro', '/electric', '/ev'],
            'service': ['/serwis', '/service', '/warsztat'],
            'configurator': ['/konfigurator', '/configure'],
            'news': ['/wiadomosci', '/news', '/aktualnosci'],
            'offers': ['/oferty', '/offers', '/promocje'],
        }
        
        for category, patterns in url_categories.items():
            if any(pattern in url_lower for pattern in patterns):
                if category not in categories:
                    categories.append(category)
        
        # Jeśli żadna kategoria, to 'other'
        if not categories:
            categories.append('other')
        
        return categories
    
    def extract_specifications(self, soup, text, url):
        """Wyciąga specyfikacje techniczne"""
        specs = {
            'detected_models': [],
            'engine': {},
            'performance': {},
            'dimensions': {},
            'prices': [],
            'electric': {}
        }
        
        try:
            # Wykryj modele
            specs['detected_models'] = self.detect_models(text, url)
            
            # DODAJ TE LINIE: definicja text_lower
            text_lower = text.lower()
            
            # Moc silnika
            power_patterns = [
                r'\b(\d{3,4})\s*(KM|kW|PS|hp)\b',  # 245 KM (min 3 cyfry)
                r'moc[:\s]+(\d{3,4})\s*(KM|kW)',   # moc: 245 KM
                r'(\d{3,4})\s*(?:koni|KM)\s*\(',   # 245 koni (
            ]
            
            for pattern in power_patterns:
                matches = re.findall(pattern, text, re.IGNORECASE)
                for value, unit in matches[:2]:  # Max 2 wartości
                    try:
                        power_value = int(value)
                        if 80 <= power_value <= 800:  # Sensowny zakres mocy
                            if unit.upper() in ['KM', 'PS', 'HP']:
                                specs['engine']['power_hp'] = power_value
                            elif unit.upper() == 'KW':
                                specs['engine']['power_kw'] = power_value
                    except:
                        continue
            
            # Moment obrotowy
            torque_matches = re.findall(r'\b(\d{3,4})\s*Nm\b', text, re.IGNORECASE)
            for match in torque_matches[:1]:
                try:
                    torque = int(match)
                    if 200 <= torque <= 1000:  # Sensowny zakres
                        specs['engine']['torque_nm'] = torque
                except:
                    continue
            
            # Przyspieszenie 0-100
            accel_matches = re.findall(r'(\d[\d,.]{1,4})\s*s\s*(?:0[-\s]*100|0[-\s]*100\s*km/h)', text)
            for match in accel_matches[:1]:
                try:
                    accel = float(match.replace(',', '.'))
                    if 2.0 <= accel <= 10.0:  # Sensowny zakres
                        specs['performance']['acceleration_0_100'] = accel
                except:
                    continue
            
            # Prędkość maksymalna
            speed_matches = re.findall(r'\b(\d{2,3})\s*km/h\b', text)
            valid_speeds = []
            for match in speed_matches:
                try:
                    speed = int(match)
                    if 150 <= speed <= 350:  # Sensowny zakres
                        valid_speeds.append(speed)
                except:
                    continue
            
            if valid_speeds:
                specs['performance']['top_speed'] = max(valid_speeds)
            
            # Zasięg (dla EV)
            range_matches = re.findall(r'\b(\d{3,4})\s*km\b', text)
            valid_ranges = []
            for match in range_matches:
                try:
                    range_val = int(match)
                    if 200 <= range_val <= 800:  # Sensowny zakres dla EV
                        valid_ranges.append(range_val)
                except:
                    continue
            
            if valid_ranges:
                specs['electric']['range_km'] = max(valid_ranges)
            
            # Ceny
            price_texts = []
            
            # 1. Szukaj w specjalnych elementach HTML
            price_selectors = [
                '.price', '.Price', '.offer-price', '.product-price',
                '[data-price]', '[data-testid="price"]', '.cena',
                '.vehicle-price', '.car-price', '.model-price',
                '.msrp',  # Manufacturer's Suggested Retail Price
                '.value',  # BMW często używa .value dla cen
            ]
            
            for selector in price_selectors:
                try:
                    price_elements = soup.select(selector)
                    for elem in price_elements[:5]:
                        price_text = elem.get_text(strip=True)
                        if price_text and len(price_text) > 3:
                            price_texts.append(price_text)
                except:
                    continue
            
            # 2. Szukaj w tekstowych wzorcach (bardziej restrykcyjne)
            price_patterns = [
                r'od\s+(\d{1,3}(?:\s?\d{3}){1,2}[,\d]*)\s*z[łl]\b',
                r'cena\s+od\s+(\d{1,3}(?:\s?\d{3}){1,2}[,\d]*)\s*z[łl]\b',
                r'\b(\d{1,3}(?:\s?\d{3}){1,2}[,\d]*)\s*z[łl]\b(?!\s*(?:od|do|za))',
                r'\b(\d{1,3}[.,]\d{3}(?:[.,]\d{3})?)\s*z[łl]\b',
                r'\b(\d{1,3}[.,]\d{3}(?:[.,]\d{3})?)\s*€\b',
            ]
            
            for pattern in price_patterns:
                matches = re.findall(pattern, text, re.IGNORECASE)
                for match in matches[:3]:
                    if isinstance(match, tuple):
                        match = match[0]
                    price_texts.append(f"{match} zł")
            
            # 3. Przetwórz znalezione ceny
            seen_prices = set()
            for price_text in price_texts:
                try:
                    # Oczyść tekst ceny
                    clean_text = re.sub(r'[^\d,\s.]', '', price_text)
                    clean_text = clean_text.replace(',', '.').strip()
                    
                    # Usuń spacje z liczb
                    clean_text = re.sub(r'\s+', '', clean_text)
                    
                    # Konwertuj na liczbę
                    if '.' in clean_text:
                        parts = clean_text.split('.')
                        if len(parts) == 2 and len(parts[1]) <= 2:
                            price_value = float(clean_text)
                        else:
                            clean_text = clean_text.replace('.', '')
                            price_value = float(clean_text)
                    else:
                        price_value = float(clean_text)
                    
                    # Sprawdź czy to sensowna cena samochodu BMW
                    if 30000 <= price_value <= 2000000:
                        rounded_price = round(price_value, -2)
                        if rounded_price not in seen_prices:
                            seen_prices.add(rounded_price)
                            specs['prices'].append({
                                'amount': price_value,
                                'currency': 'PLN',
                                'formatted': f"{price_value:,.0f} zł".replace(',', ' '),
                                'source_text': price_text[:50]
                            })
                    
                except Exception as e:
                    continue
            
            # Ogranicz do 5 cen
            if specs['prices']:
                specs['prices'] = specs['prices'][:5]
            
            # Pojemność silnika
            capacity_match = re.search(r'\b(\d{1,3}[,\d]*)\s*cm[³3]\b', text)
            if capacity_match:
                specs['engine']['capacity_cc'] = capacity_match.group(1)
            
            # Pojemność baterii
            battery_match = re.search(r'\b(\d{2,3}[,\d]*)\s*kWh\b', text, re.IGNORECASE)
            if battery_match:
                specs['electric']['battery_kwh'] = battery_match.group(1)
            
            # Ładowanie
            charging_match = re.search(r'\b(\d{1,3}[,\d]*)\s*kW\s*(?:AC|DC)?\s*ładowan', text, re.IGNORECASE)
            if charging_match:
                specs['electric']['charging_kw'] = charging_match.group(1)
            
            # Sprawdź czy to strona specyfikacji
            specs_page_keywords = ['dane techniczne', 'technical data', 'specyfikacja', 'technische daten']
            for keyword in specs_page_keywords:
                if keyword in text_lower:
                    specs['is_specs_page'] = True
                    break
            
            # Oczyść puste wartości
            if 'engine' in specs:
                specs['engine'] = {k: v for k, v in specs['engine'].items() if v not in [0, None, '']}
            
        except Exception as e:
            self.logger.warning(f"Błąd przy ekstrakcji specyfikacji: {e}")
        
        return specs
    
    def scrape_page(self, url):
        """Scrapuje pojedynczą stronę"""
        if url in self.visited_urls:
            return None
            
        self.logger.info(f"🌐 {url[:80]}...")
        self.visited_urls.add(url)
        
        try:
            # Otwórz stronę z dłuższym timeoutem
            self.driver.set_page_load_timeout(45)
            self.driver.get(url)
            
            # Daj czas na załadowanie strony BMW
            time.sleep(3)  # BMW potrzebuje więcej czasu
            
            # Sprawdź czy to nie strona błędu
            page_title = self.driver.title.lower()
            if any(error_word in page_title for error_word in 
                  ['error', 'błąd', 'nie znaleziono', 'not found', '404', '500']):
                self.logger.warning(f"  ⚠️ Strona błędu: {page_title}")
                return None
            
            # Pobierz HTML
            html = self.driver.page_source
            soup = BeautifulSoup(html, 'html.parser')
            
            # Wyciągnij treść
            content_data = self.extract_content(soup, url)
            
            # Jeśli za mało treści, pomiń
            if content_data['word_count'] < 30:
                self.logger.info(f"  ⏭️ Za mało treści ({content_data['word_count']} słów)")
                return None
            
            # Wyciągnij specyfikacje
            full_text = content_data['content'] + ' ' + content_data['title'] + ' ' + content_data['description']
            specs = self.extract_specifications(soup, full_text, url)
            
            # Wykryj modele
            detected_models = self.detect_models(full_text, url)
            
            # Kategoryzuj
            categories = self.categorize_content(full_text, url, detected_models)
            
            # Przygotuj dane
            page_data = {
                'url': url,
                'title': content_data['title'],
                'description': content_data['description'],
                'content': content_data['content'],
                'word_count': content_data['word_count'],
                'categories': categories,
                'detected_models': detected_models,
                'specifications': specs,
                'is_model_page': 'models' in categories or bool(detected_models),
                'model_from_url': self.extract_model_from_url(url),
                'scraped_at': datetime.now().isoformat(),
                'priority': 1 if ('models' in categories or bool(detected_models)) else 2
            }
            
            # Dodaj do odpowiedniej listy
            if page_data['is_model_page']:
                self.model_pages.append(page_data)
                
                # Log z lepszymi informacjami
                model_info = []
                if page_data['model_from_url']:
                    model_info.append(f"URL: {page_data['model_from_url']}")
                if detected_models:
                    model_info.append(f"Tekst: {', '.join(detected_models[:2])}")
                
                spec_info = []
                if specs.get('engine', {}).get('power_hp'):
                    spec_info.append(f"{specs['engine']['power_hp']}KM")
                if specs.get('prices'):
                    price = specs['prices'][0]['formatted']
                    spec_info.append(f"{price}")
                
                log_msg = f"  🚗"
                if model_info:
                    log_msg += f" {' | '.join(model_info)}"
                if spec_info:
                    log_msg += f" [{' | '.join(spec_info)}]"
                log_msg += f" ({content_data['word_count']} słów, {len(specs.get('prices', []))} cen)"
                
                self.logger.info(log_msg)
            else:
                self.other_pages.append(page_data)
                self.logger.info(f"  📄 Inne [{categories[0] if categories else 'other'}] - {content_data['word_count']} słów")
            
            self.scraped_data.append(page_data)
            
            # Znajdź linki na stronie
            self.extract_links(url, soup)
            
            return page_data
            
        except Exception as e:
            self.logger.error(f"  ❌ Błąd: {str(e)[:80]}")
            return None
    
    def extract_links(self, base_url, soup):
        """Wyciąga linki ze strony"""
        try:
            links_found = 0
            
            for link in soup.find_all('a', href=True)[:80]:  # Limit 80 linków
                try:
                    href = link['href']
                    
                    # Normalizuj URL
                    normalized = self.normalize_url(href, base_url)
                    
                    if normalized and self.is_valid_url(normalized):
                        # Sprawdź czy już w kolejce
                        if normalized not in self.to_visit and normalized not in self.visited_urls:
                            
                            # Określ priorytet
                            priority = self.get_url_priority(normalized)
                            
                            # Dodaj do kolejki z priorytetem
                            self.to_visit.append((priority, normalized))
                            links_found += 1
                            
                except Exception as e:
                    continue
            
            # Posortuj kolejkę według priorytetu
            self.to_visit = deque(sorted(self.to_visit, key=lambda x: x[0]))
            
            if links_found > 0:
                self.logger.debug(f"Znaleziono {links_found} nowych linków")
                
        except Exception as e:
            self.logger.error(f"Błąd przy ekstrakcji linków: {e}")
    
    def get_url_priority(self, url):
        """Określa priorytet URL-a"""
        url_lower = url.lower()
        
        # Priorytet 1: Strony z modelami i specyfikacjami
        if any(pattern in url_lower for pattern in 
              ['/models/', '/x', '/i', '/m', '/seria', '/series',
               'dane-techniczne', 'technical-data', 'specyfikacja',
               '/bmw-']):  # Nowy wzorzec dla BMW
            return 1
        
        # Priorytet 2: Ważne sekcje
        if any(pattern in url_lower for pattern in 
              ['/elektro', '/electric', '/konfigurator', '/service', '/serwis']):
            return 2
        
        # Priorytet 3: Inne treści
        return 3
    
    def crawl(self):
        """Główna funkcja crawlowania"""
        self.logger.info("=" * 70)
        self.logger.info("🚗 BMW CRAWLER - START (Normalna przeglądarka)")
        self.logger.info("=" * 70)
        
        start_time = time.time()
        
        try:
            # 1. Zacznij od priorytetowych URL-i
            self.logger.info("\n🎯 Rozpoczynam od priorytetowych stron...")
            for url in self.priority_urls:
                if self.is_valid_url(url):
                    self.to_visit.append((1, url))
            
            # 2. Główna pętla crawlowania
            max_pages = 50  # Mniejsza liczba dla testów
            pages_scraped = 0
            
            while self.to_visit and pages_scraped < max_pages:
                try:
                    # Pobierz następny URL (z najwyższym priorytetem)
                    if not self.to_visit:
                        break
                    
                    priority, url = self.to_visit.popleft()
                    
                    # Scrapuj stronę
                    result = self.scrape_page(url)
                    if result:
                        pages_scraped += 1
                        
                        # Co 5 stron pokaż status
                        if pages_scraped % 5 == 0:
                            elapsed = time.time() - start_time
                            self.logger.info(f"\n📊 Status: {pages_scraped}/{max_pages} stron, {elapsed:.0f}s")
                            self.logger.info(f"   🚗 Modele: {len(self.model_pages)}")
                            self.logger.info(f"   📰 Inne: {len(self.other_pages)}")
                    
                    # Rate limiting - dłuższe czasy dla BMW
                    time.sleep(2 if priority == 1 else 1.5)
                    
                except Exception as e:
                    self.logger.error(f"Błąd w głównej pętli: {e}")
                    continue
            
            self.logger.info(f"\n✅ Zakończono crawlowanie. Zebrano {pages_scraped} stron.")
            
        except KeyboardInterrupt:
            self.logger.info("\n🛑 Crawler zatrzymany przez użytkownika")
        except Exception as e:
            self.logger.error(f"\n❌ Nieoczekiwany błąd: {e}")
            import traceback
            traceback.print_exc()
        finally:
            # Zawsze zapisz dane
            self.save_all_data()
            self.print_summary(start_time)
            
            # Zamknij przeglądarkę
            self.driver.quit()
    
    def save_progress(self):
        """Zapisuje postęp tymczasowo"""
        try:
            temp_file = self.output_dir / "temp_progress.json"
            with open(temp_file, 'w', encoding='utf-8') as f:
                json.dump({
                    'visited_urls': list(self.visited_urls),
                    'scraped_count': len(self.scraped_data),
                    'timestamp': datetime.now().isoformat()
                }, f, ensure_ascii=False, indent=2)
                
            self.logger.debug("Zapisano postęp tymczasowo")
        except Exception as e:
            self.logger.error(f"Błąd przy zapisie postępu: {e}")
    
    def save_all_data(self):
        """Zapisuje wszystkie zebrane dane"""
        if not self.scraped_data:
            self.logger.warning("Brak danych do zapisania!")
            return
        
        try:
            # 1. Wszystkie dane
            all_file = self.output_dir / "all_data.jsonl"
            with open(all_file, 'w', encoding='utf-8') as f:
                for item in self.scraped_data:
                    json.dump(item, f, ensure_ascii=False)
                    f.write('\n')
            
            # 2. Tylko modele
            models_file = self.output_dir / "models_data.json"
            with open(models_file, 'w', encoding='utf-8') as f:
                json.dump(self.model_pages, f, ensure_ascii=False, indent=2)
            
            # 3. Inne treści
            other_file = self.output_dir / "other_data.json"
            with open(other_file, 'w', encoding='utf-8') as f:
                json.dump(self.other_pages, f, ensure_ascii=False, indent=2)
            
            # 4. Statystyki
            stats = self.calculate_statistics()
            stats_file = self.output_dir / "stats.json"
            with open(stats_file, 'w', encoding='utf-8') as f:
                json.dump(stats, f, ensure_ascii=False, indent=2)
            
            # 5. URL-e
            urls_file = self.output_dir / "visited_urls.txt"
            with open(urls_file, 'w', encoding='utf-8') as f:
                for url in sorted(self.visited_urls):
                    f.write(url + '\n')
            
            # 6. Specjalny plik z cenami
            prices_file = self.output_dir / "prices_analysis.json"
            all_prices_data = []
            for page in self.scraped_data:
                prices = page.get('specifications', {}).get('prices', [])
                if prices:
                    for price in prices:
                        all_prices_data.append({
                            'url': page['url'],
                            'title': page['title'][:100],
                            'model': page.get('model_from_url', ''),
                            'detected_models': page.get('detected_models', []),
                            'price': price
                        })
            
            if all_prices_data:
                with open(prices_file, 'w', encoding='utf-8') as f:
                    json.dump(all_prices_data, f, ensure_ascii=False, indent=2)
            
            self.logger.info(f"\n💾 Dane zapisane w: {self.output_dir}")
            self.logger.info(f"   📄 Wszystkie dane: {all_file}")
            self.logger.info(f"   🚗 Modele: {models_file}")
            self.logger.info(f"   📰 Inne: {other_file}")
            self.logger.info(f"   📊 Statystyki: {stats_file}")
            if all_prices_data:
                self.logger.info(f"   💰 Analiza cen: {prices_file}")
            
        except Exception as e:
            self.logger.error(f"Błąd przy zapisie danych: {e}")
    
    def calculate_statistics(self):
        """Oblicza statystyki"""
        stats = {
            'total_pages': len(self.scraped_data),
            'model_pages': len(self.model_pages),
            'other_pages': len(self.other_pages),
            'visited_urls': len(self.visited_urls),
            'total_words': sum(p['word_count'] for p in self.scraped_data),
            'avg_words_per_page': sum(p['word_count'] for p in self.scraped_data) / len(self.scraped_data) if self.scraped_data else 0,
            'categories': {},
            'models_found': [],
            'models_from_url': [],
            'specs_stats': {},
            'prices_stats': {},
            'output_dir': str(self.output_dir),
            'scraped_at': datetime.now().isoformat()
        }
        
        # Licz kategorie
        for page in self.scraped_data:
            for category in page.get('categories', []):
                stats['categories'][category] = stats['categories'].get(category, 0) + 1
        
        # Zebrane modele z tekstu
        all_models = []
        for page in self.model_pages:
            all_models.extend(page.get('detected_models', []))
        
        if all_models:
            from collections import Counter
            model_counts = Counter(all_models)
            stats['models_found'] = dict(model_counts.most_common(20))
        
        # Modele z URL
        url_models = []
        for page in self.model_pages:
            if page.get('model_from_url'):
                url_models.append(page['model_from_url'])
        
        if url_models:
            from collections import Counter
            url_model_counts = Counter(url_models)
            stats['models_from_url'] = dict(url_model_counts.most_common(20))
        
        # Statystyki specyfikacji
        specs_fields = ['power_hp', 'power_kw', 'torque_nm', 'acceleration_0_100', 'top_speed']
        for field in specs_fields:
            values = []
            for page in self.scraped_data:
                specs = page.get('specifications', {})
                if field == 'power_hp' and 'engine' in specs:
                    values.append(specs['engine'].get(field))
                elif field == 'power_kw' and 'engine' in specs:
                    values.append(specs['engine'].get(field))
                elif field == 'torque_nm' and 'engine' in specs:
                    values.append(specs['engine'].get(field))
                elif field in ['acceleration_0_100', 'top_speed'] and 'performance' in specs:
                    values.append(specs['performance'].get(field))
            
            # Oczyść None wartości
            values = [v for v in values if v is not None]
            if values:
                stats['specs_stats'][field] = {
                    'count': len(values),
                    'avg': sum(values) / len(values),
                    'min': min(values),
                    'max': max(values)
                }
        
        # STATYSTYKI CEN
        all_prices = []
        price_sources = {}
        for page in self.scraped_data:
            prices = page.get('specifications', {}).get('prices', [])
            for price in prices:
                if 'amount' in price:
                    all_prices.append(price['amount'])
                    url_key = page['url']
                    price_sources[url_key] = price_sources.get(url_key, 0) + 1
        
        if all_prices:
            unique_prices = set(round(price / 1000) * 1000 for price in all_prices)
            
            stats['prices_stats'] = {
                'total_found': len(all_prices),
                'unique_prices': len(unique_prices),
                'pages_with_prices': sum(1 for page in self.scraped_data if page.get('specifications', {}).get('prices')),
                'min': min(all_prices),
                'max': max(all_prices),
                'avg': sum(all_prices) / len(all_prices),
                'avg_formatted': f"{sum(all_prices) / len(all_prices):,.0f} zł".replace(',', ' '),
                'sources_count': len(price_sources)
            }
        
        return stats
    
    def print_summary(self, start_time):
        """Wyświetla podsumowanie"""
        elapsed = time.time() - start_time
        elapsed_min = elapsed / 60
        
        stats = self.calculate_statistics()
        
        print("\n" + "=" * 70)
        print("🎉 BMW CRAWLER - PODSUMOWANIE")
        print("=" * 70)
        
        print(f"\n📊 STATYSTYKI:")
        print(f"  Stron zebranych: {stats['total_pages']}")
        print(f"  Stron z modelami: {stats['model_pages']}")
        print(f"  Innych stron: {stats['other_pages']}")
        print(f"  Słowa ogółem: {stats['total_words']:,}")
        print(f"  Średnio słów/stronę: {stats['avg_words_per_page']:.0f}")
        print(f"  ⏱️ Czas pracy: {elapsed_min:.1f} minut")
        
        if elapsed > 0:
            pages_per_min = stats['total_pages'] / elapsed_min
            print(f"  📈 Wydajność: {pages_per_min:.1f} stron/minutę")
        
        print(f"\n📂 KATEGORIE:")
        for category, count in sorted(stats['categories'].items(), key=lambda x: x[1], reverse=True):
            print(f"  {category}: {count} stron")
        
        if stats.get('models_from_url'):
            print(f"\n🚗 MODELE (z URL - NAJWIARYGODNIEJSZE):")
            for model, count in list(stats['models_from_url'].items())[:15]:
                print(f"  {model}: {count} stron")
        elif stats.get('models_found'):
            print(f"\n🚗 WYKRYTE MODELE (z tekstu):")
            for model, count in list(stats['models_found'].items())[:15]:
                print(f"  {model}: {count} stron")
        
        if stats.get('specs_stats'):
            print(f"\n🔧 SPECYFIKACJE:")
            if 'power_hp' in stats['specs_stats']:
                power = stats['specs_stats']['power_hp']
                print(f"  Średnia moc: {power['avg']:.0f} KM ({power['count']} stron)")
            if 'acceleration_0_100' in stats['specs_stats']:
                accel = stats['specs_stats']['acceleration_0_100']
                print(f"  0-100 km/h: {accel['avg']:.1f} s ({accel['count']} stron)")
        
        if stats.get('prices_stats'):
            prices = stats['prices_stats']
            print(f"\n💰 ANALIZA CEN:")
            print(f"  Znalezionych cen: {prices['total_found']}")
            print(f"  Unikalnych cen: {prices['unique_prices']}")
            print(f"  Stron z cenami: {prices['pages_with_prices']}")
            print(f"  Średnia cena: {prices['avg_formatted']}")
            print(f"  Zakres: {prices['min']:,.0f} - {prices['max']:,.0f} zł")
        
        print(f"\n💾 DANE ZAPISANE W: {self.output_dir}")
        print("=" * 70)

# Uruchom bez headless
if __name__ == "__main__":
    crawler = CompleteBMWCrawler(headless=False)  # Wyłącz headless dla BMW
    crawler.crawl()