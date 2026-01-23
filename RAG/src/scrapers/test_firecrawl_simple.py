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

class BMWCrawler:
    """Inteligentny crawler strony BMW z aktualnymi URL-ami"""
    
    def __init__(self):
        # Setup logging
        logging.basicConfig(
            level=logging.INFO, 
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('bmw_crawler.log', encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        # 🎯 FIXED ŚCIEŻKA - zawsze do głównego folderu RAG/output
        script_dir = Path(__file__).resolve().parent
        project_root = script_dir.parent.parent  # Idź do Chatbot_BMW
        self.output_dir = project_root / "RAG" / "output" / f"bmw_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        self.logger.info(f"📁 Output directory: {self.output_dir}")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup Chrome
        options = webdriver.ChromeOptions()
        
        # Opcje anty-detection
        options.add_argument('--disable-blink-features=AutomationControlled')
        options.add_experimental_option("excludeSwitches", ["enable-automation", "enable-logging"])
        options.add_experimental_option('useAutomationExtension', False)
        
        # User agent - wyglądamy jak normalna przeglądarka
        options.add_argument('user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36')
        
        # Performance options
        options.add_argument('--disable-dev-shm-usage')
        options.add_argument('--no-sandbox')
        options.add_argument('--window-size=1920,1080')
        options.add_argument('--start-maximized')
        options.add_argument('--disable-gpu')
        
        # Headless dla szybkości - WŁĄCZONE ale z opcjami
        options.add_argument('--headless=new')
        
        # Dodatkowe opcje
        options.add_argument('--disable-extensions')
        options.add_argument('--disable-notifications')
        
        self.driver = webdriver.Chrome(options=options)
        self.driver.set_page_load_timeout(45)
        self.driver.set_script_timeout(30)
        
        # Execute CDP commands to bypass detection
        self.driver.execute_cdp_cmd('Network.setUserAgentOverride', {
            "userAgent": 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        })
        self.driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
        
        # Stats and data
        self.visited_urls = set()
        self.error_urls = set()
        self.to_visit = deque()
        self.scraped_data = []
        self.model_pages = []
        self.other_pages = []
        
        # AKTUALNE PRIORYTETOWE URL-E - sprawdzone i działające
        self.priority_urls = [
            "https://www.bmw.pl/pl/index.html",
            "https://www.bmw.pl/pl/all-models.html",
            "https://www.bmw.pl/pl/topics/fascination-bmw/models.html",
            "https://www.bmw.pl/pl/topics/fascination-bmw/bmw-i.html",
            "https://www.bmw.pl/pl/topics/fascination-bmw/bmw-m.html",
            "https://www.bmw.pl/pl/topics/fascination-bmw/x-models.html",
            "https://www.bmw.pl/pl/topics/offers-and-services/service.html",
            "https://www.bmw.pl/pl/topics/offers-and-services/parts-and-accessories.html",
            "https://www.bmw.pl/pl/topics/offers-and-services/financial-services.html",
            "https://www.bmw.pl/pl/configurator.html",
            "https://www.bmw.pl/pl/disclaimer.html",
            "https://www.bmw.pl/pl/topics/fascination-bmw/7-series.html",
            "https://www.bmw.pl/pl/topics/fascination-bmw/5-series.html",
            "https://www.bmw.pl/pl/topics/fascination-bmw/3-series.html",
            "https://www.bmw.pl/pl/topics/fascination-bmw/2-series.html",
            "https://www.bmw.pl/pl/topics/fascination-bmw/x1.html",
            "https://www.bmw.pl/pl/topics/fascination-bmw/x3.html",
            "https://www.bmw.pl/pl/topics/fascination-bmw/x5.html",
            "https://www.bmw.pl/pl/topics/fascination-bmw/x7.html",
            "https://www.bmw.pl/pl/topics/fascination-bmw/i4.html",
            "https://www.bmw.pl/pl/topics/fascination-bmw/ix.html",
            "https://www.bmw.pl/pl/topics/fascination-bmw/ix1.html",
        ]
        
        # Alternatywne domeny do sprawdzenia
        self.bmw_domains = [
            "https://www.bmw.pl",
            "https://bmw.pl",
        ]
        
        # Wzorce dla modeli - poprawione
        self.model_patterns = {
            # Seria
            '1 Series': r'\b1\s*Seri(?:es|a|i)\b|\b118i\b|\b120i\b|\bM135i\b',
            '2 Series': r'\b2\s*Seri(?:es|a|i)\b|\b218i\b|\b220i\b|\bM240i\b',
            '3 Series': r'\b3\s*Seri(?:es|a|i)\b|\b318i\b|\b320i\b|\b330i\b|\bM340i\b',
            '4 Series': r'\b4\s*Seri(?:es|a|i)\b|\b420i\b|\b430i\b|\bM440i\b',
            '5 Series': r'\b5\s*Seri(?:es|a|i)\b|\b520i\b|\b530i\b|\bM550i\b',
            '7 Series': r'\b7\s*Seri(?:es|a|i)\b|\b740i\b|\b750i\b|\bM760i\b',
            '8 Series': r'\b8\s*Seri(?:es|a|i)\b|\b840i\b|\bM850i\b',
            
            # X Series
            'X1': r'\bX1\b',
            'X2': r'\bX2\b',
            'X3': r'\bX3\b',
            'X4': r'\bX4\b',
            'X5': r'\bX5\b',
            'X6': r'\bX6\b',
            'X7': r'\bX7\b',
            'XM': r'\bXM\b',
            
            # i Series (elektryczne)
            'i3': r'\bi3\b',
            'i4': r'\bi4\b',
            'i5': r'\bi5\b',
            'i7': r'\bi7\b',
            'iX': r'\biX\b',
            'iX1': r'\biX1\b',
            'iX2': r'\biX2\b',
            'iX3': r'\biX3\b',
            
            # M Series
            'M2': r'\bM2\b',
            'M3': r'\bM3\b',
            'M4': r'\bM4\b',
            'M5': r'\bM5\b',
            'M8': r'\bM8\b',
            'X3 M': r'\bX3\s*M\b',
            'X4 M': r'\bX4\s*M\b',
            'X5 M': r'\bX5\s*M\b',
            'X6 M': r'\bX6\s*M\b',
            
            # Inne
            'Z4': r'\bZ4\b',
            '2 Series Gran Coupé': r'2\s*Series\s*Gran\s*Coup[ée]',
            '2 Series Active Tourer': r'2\s*Series\s*Active\s*Tourer',
            '2 Series Coupé': r'2\s*Series\s*Coup[ée]',
        }
        
        # Keywords dla różnych kategorii
        self.category_keywords = {
            'models': ['model', 'silnik', 'moc', 'przyspieszenie', 'wymiary', 'cena', 'specyfikacja', 'dane techniczne'],
            'electric': ['elektryczny', 'elektryczna', 'elektryczne', 'ładowanie', 'bateria', 'zasięg', 'ev', 'phev', 'kwh'],
            'service': ['serwis', 'przegląd', 'naprawa', 'części', 'gwarancja', 'warsztat', 'oryginalne części'],
            'configurator': ['konfigurator', 'wyposażenie', 'opcje', 'pakiety', 'kolory', 'tapicerka'],
            'offers': ['oferta', 'promocja', 'leasing', 'finansowanie', 'raty', 'rabat'],
            'news': ['nowość', 'premiera', 'aktualność', 'informacja', 'wydarzenie'],
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
        url = re.sub(r'\?.*', '', url)  # Usuń wszystkie parametry
        
        # Upewnij się że to BMW
        if 'bmw.pl' not in url:
            # Sprawdź czy można przekonwertować na BMW
            if 'bmw.' in url and 'pl' in url:
                parts = url.split('/')
                domain = parts[2] if len(parts) > 2 else ''
                if 'bmw' in domain and 'pl' in domain:
                    # Zachowaj oryginalny URL
                    pass
                else:
                    return None
            else:
                return None
        
        return url.strip('/')
    
    def is_valid_url(self, url):
        """Sprawdza czy URL jest poprawny do crawlowania"""
        if not url:
            return False
            
        # Musi być BMW Polska
        if not ('bmw.pl' in url and '/pl/' in url):
            return False
            
        # Nie może być już odwiedzony ani błędny
        if url in self.visited_urls or url in self.error_urls:
            return False
            
        # Unikaj niepotrzebnych rozszerzeń
        skip_extensions = ['.pdf', '.jpg', '.jpeg', '.png', '.gif', '.svg', 
                          '.mp4', '.avi', '.mov', '.mp3', '.wav', '.zip', 
                          '.rar', '.7z', '.exe', '.dmg', '.msi', '.css', '.js']
        
        if any(url.lower().endswith(ext) for ext in skip_extensions):
            return False
            
        # Unikaj social media i tracking
        skip_patterns = [
            r'facebook\.com', r'twitter\.com', r'instagram\.com', 
            r'linkedin\.com', r'youtube\.com', r'google\.com',
            r'/api/', r'/ajax/', r'/rest/', r'/graphql', r'/wp-json/',
            r'/login', r'/register', r'/signin', r'/signup', r'/logout',
            r'/cart', r'/checkout', r'/basket', r'/order', r'/konto',
            r'\.xml$', r'\.json$', r'\.rss$', r'\.atom$',
            r'/tracking/', r'/analytics/', r'/stats/',
            r'/search\b', r'/szukaj\b',
        ]
        
        for pattern in skip_patterns:
            if re.search(pattern, url, re.IGNORECASE):
                return False
                
        # Sprawdź głębokość URL-a
        parsed = urlparse(url)
        path_depth = parsed.path.strip('/').count('/')
        if path_depth > 6:
            return False
            
        return True
    
    def check_page_availability(self, url):
        """Sprawdza czy strona istnieje przed pełnym scrapowaniem"""
        try:
            # Szybkie sprawdzenie HEAD request przez JavaScript
            check_script = """
            return fetch(arguments[0], {method: 'HEAD', mode: 'no-cors'})
                .then(response => true)
                .catch(error => false);
            """
            
            # Uruchom sprawdzenie
            result = self.driver.execute_async_script(check_script, url)
            return result is True
        except:
            # Jeśli nie można sprawdzić, załóż że strona istnieje
            return True
    
    def accept_cookies_if_needed(self):
        """Próbuje zaakceptować cookies"""
        try:
            time.sleep(1)
            
            # Różne selektory dla cookie bannerów BMW
            cookie_selectors = [
                'button#onetrust-accept-btn-handler',
                'button[data-test-id*="cookie"]',
                'button[aria-label*="cookie"]',
                'button[aria-label*="Cookie"]',
                'button:contains("Akceptuj")',
                'button:contains("Accept")',
                'button:contains("Zaakceptuj")',
                'button:contains("Zgadzam się")',
                '.cookie-consent button',
                '.cookie-banner button',
            ]
            
            for selector in cookie_selectors:
                try:
                    # Spróbuj znaleźć przez CSS
                    elements = self.driver.find_elements(By.CSS_SELECTOR, selector)
                    for element in elements:
                        if element.is_displayed():
                            element.click()
                            self.logger.debug("✅ Zaakceptowano cookies")
                            time.sleep(0.5)
                            return True
                except:
                    continue
                    
        except Exception as e:
            self.logger.debug(f"Błąd przy akceptacji cookies: {e}")
        
        return False
    
    def extract_content(self, soup, url):
        """Wyciąga treść ze strony"""
        try:
            # Tytuł
            title = soup.title.string if soup.title else ""
            
            # Meta description
            meta_desc = soup.find('meta', {'name': 'description'})
            description = meta_desc['content'] if meta_desc else ""
            
            # H1
            h1_elem = soup.find('h1')
            h1 = h1_elem.get_text(strip=True) if h1_elem else ""
            
            # Główna treść - selektory specyficzne dla BMW
            content_selectors = [
                'main', 
                'article', 
                '.content', 
                '.main-content',
                '.page-content',
                '#content', 
                '.article-content',
                '.text-content',
                '.body-content',
                '.editor-content',
                '[role="main"]',
                '.bmw-content',
                '.module-text',
                '.teaser-text',
                '.text-module',
                '.rich-text'
            ]
            
            main_content = None
            for selector in content_selectors:
                try:
                    elements = soup.select(selector)
                    for elem in elements:
                        text = elem.get_text(strip=True)
                        if len(text) > 100:  # Wystarczająco treści
                            main_content = elem
                            break
                    if main_content:
                        break
                except:
                    continue
            
            # Zbierz tekst
            text_parts = []
            
            # Dodaj tytuł i H1
            if title and len(title) > 10:
                text_parts.append(f"Tytuł: {title}")
            if h1 and len(h1) > 10:
                text_parts.append(f"Nagłówek: {h1}")
            if description and len(description) > 20:
                text_parts.append(f"Opis: {description}")
            
            if main_content:
                # Pobierz tekst z głównego kontenera
                elements = main_content.find_all(['p', 'h2', 'h3', 'h4', 'li', 'dt', 'dd'])
                for elem in elements[:100]:  # Limit
                    text = elem.get_text(strip=True)
                    if 30 < len(text) < 1000:
                        text_parts.append(text)
            else:
                # Fallback: wszystkie paragrafy i nagłówki
                elements = soup.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'li'])
                for elem in elements[:80]:
                    text = elem.get_text(strip=True)
                    if 40 < len(text) < 800:
                        text_parts.append(text)
            
            content = ' '.join(text_parts)
            
            # Oczyść tekst
            content = re.sub(r'\s+', ' ', content)
            content = re.sub(r'\n+', ' ', content)
            
            word_count = len(content.split())
            
            return {
                'title': title[:500],
                'h1': h1[:500],
                'description': description[:500],
                'content': content[:25000],
                'word_count': word_count
            }
            
        except Exception as e:
            self.logger.error(f"Błąd przy ekstrakcji treści: {e}")
            return {
                'title': '',
                'h1': '',
                'description': '',
                'content': '',
                'word_count': 0
            }
    
    def detect_models(self, text, url):
        """Wykrywa modele BMW w tekście"""
        detected_models = []
        
        # Sprawdź w tekście
        for model_name, pattern in self.model_patterns.items():
            try:
                if re.search(pattern, text, re.IGNORECASE):
                    # Sprawdź czy to nie fałszywe trafienie
                    if self.validate_model_detection(model_name, text):
                        detected_models.append(model_name)
            except:
                continue
        
        # Sprawdź też w URL
        url_lower = url.lower()
        for model_name in self.model_patterns.keys():
            model_simple = model_name.lower().replace(' ', '-').replace('series', '').strip('-')
            if model_simple and model_simple in url_lower:
                if model_name not in detected_models:
                    detected_models.append(model_name)
        
        # Usuń duplikaty i ogranicz
        detected_models = list(set(detected_models))
        if len(detected_models) > 8:
            # Ogranicz do 8 najważniejszych
            detected_models = detected_models[:8]
        
        return detected_models
    
    def validate_model_detection(self, model_name, text):
        """Sprawdza czy wykrycie modelu jest prawidłowe"""
        model_lower = model_name.lower()
        text_lower = text.lower()
        
        # Wyjątki dla fałszywych trafień
        false_positives = {
            'x1': ['example1', 'ex1', 'x10', 'x100'],
            'x2': ['example2', 'ex2', 'x20', 'x200'],
            'x3': ['example3', 'ex3', 'x30', 'x300'],
            'x4': ['example4', 'ex4', 'x40', 'x400'],
            'x5': ['example5', 'ex5', 'x50', 'x500'],
            'i3': ['intel i3', 'core i3', 'processor i3'],
            'i5': ['intel i5', 'core i5', 'processor i5'],
            'i7': ['intel i7', 'core i7', 'processor i7'],
        }
        
        if model_lower in false_positives:
            for fp in false_positives[model_lower]:
                if fp in text_lower:
                    return False
        
        return True
    
    def categorize_content(self, text, url, detected_models):
        """Kategoryzuje treść"""
        categories = set()
        text_lower = text.lower()
        url_lower = url.lower()
        
        # Priorytet: modele
        if detected_models:
            categories.add('models')
        
        # Sprawdź słowa kluczowe
        for category, keywords in self.category_keywords.items():
            if category == 'models' and 'models' in categories:
                continue
            
            keyword_count = 0
            for keyword in keywords:
                if keyword in text_lower:
                    keyword_count += 1
                    if keyword_count >= 2:
                        categories.add(category)
                        break
        
        # Sprawdź URL patterns
        url_patterns = {
            'models': ['/models/', '/seria', '/series/', '/x1', '/x2', '/x3', '/x4', '/x5', '/x6', '/x7', '/i3', '/i4', '/i5', '/i7', '/ix'],
            'electric': ['/elektro', '/electric', '/i-models', '/i-series'],
            'service': ['/serwis', '/service', '/warsztat', '/parts'],
            'configurator': ['/konfigurator', '/configure'],
            'offers': ['/oferty', '/offers', '/promocje'],
        }
        
        for category, patterns in url_patterns.items():
            if any(pattern in url_lower for pattern in patterns):
                categories.add(category)
        
        # Jeśli żadna kategoria, to 'other'
        if not categories:
            categories.add('other')
        
        return list(categories)
    
    def scrape_page(self, url):
        """Scrapuje pojedynczą stronę"""
        if url in self.visited_urls:
            return None
            
        self.logger.info(f"Scrapuję: {url}")
        self.visited_urls.add(url)
        
        try:
            # Otwórz stronę
            self.driver.get(url)
            time.sleep(3)  # Czekaj na załadowanie
            
            # Spróbuj zaakceptować cookies
            self.accept_cookies_if_needed()
            
            # Sprawdź czy to nie strona błędu
            page_title = self.driver.title.lower()
            current_url = self.driver.current_url.lower()
            
            error_indicators = [
                'error', 'błąd', 'nie znaleziono', 'not found', '404',
                '500', '503', 'dostęp zabroniony', 'forbidden',
                'strona niedostępna', 'page not available'
            ]
            
            # Sprawdź tytuł i URL
            if any(indicator in page_title for indicator in error_indicators):
                self.logger.warning(f"Strona błędu: {url}")
                self.error_urls.add(url)
                return None
            
            if any(indicator in current_url for indicator in ['error', '404']):
                self.logger.warning(f"URL błędu: {url}")
                self.error_urls.add(url)
                return None
            
            # Pobierz HTML
            html = self.driver.page_source
            soup = BeautifulSoup(html, 'html.parser')
            
            # Sprawdź czy jest treść
            body = soup.find('body')
            if not body:
                self.logger.info(f"Brak body na stronie: {url}")
                return None
            
            body_text = body.get_text(strip=True)
            if len(body_text) < 200:
                self.logger.info(f"Za mało treści ({len(body_text)} znaków): {url}")
                return None
            
            # Wyciągnij treść
            content_data = self.extract_content(soup, url)
            
            # Jeśli za mało treści, pomiń
            if content_data['word_count'] < 100:
                self.logger.info(f"Za mało słów ({content_data['word_count']}): {url}")
                return None
            
            # Wyciągnij specyfikacje
            full_text = content_data['content']
            specs = {
                'detected_models': [],
                'engine': {},
                'performance': {},
                'prices': []
            }
            
            # Wykryj modele
            detected_models = self.detect_models(full_text, url)
            specs['detected_models'] = detected_models
            
            # Kategoryzuj
            categories = self.categorize_content(full_text, url, detected_models)
            
            # Przygotuj dane
            page_data = {
                'url': url,
                'title': content_data['title'],
                'h1': content_data['h1'],
                'description': content_data['description'],
                'content': content_data['content'],
                'word_count': content_data['word_count'],
                'categories': categories,
                'detected_models': detected_models,
                'specifications': specs,
                'is_model_page': bool(detected_models),
                'scraped_at': datetime.now().isoformat()
            }
            
            # Dodaj do odpowiedniej listy
            if page_data['is_model_page']:
                self.model_pages.append(page_data)
                models_str = ', '.join(detected_models[:3])
                if len(detected_models) > 3:
                    models_str += f' (+{len(detected_models)-3})'
                self.logger.info(f"✅ MODEL [{len(detected_models)}]: {models_str} - {content_data['word_count']} słów")
            else:
                self.other_pages.append(page_data)
                category_str = categories[0] if categories else 'other'
                self.logger.info(f"📄 {category_str.upper()} - {content_data['word_count']} słów")
            
            self.scraped_data.append(page_data)
            
            # Znajdź linki na stronie
            self.extract_links(url, soup)
            
            return page_data
            
        except TimeoutException:
            self.logger.warning(f"Timeout: {url}")
            self.error_urls.add(url)
            return None
        except Exception as e:
            self.logger.error(f"Błąd: {url} - {str(e)[:100]}")
            self.error_urls.add(url)
            return None
    
    def extract_links(self, base_url, soup):
        """Wyciąga linki ze strony"""
        try:
            links_found = 0
            new_links = set()
            
            for link in soup.find_all('a', href=True):
                try:
                    href = link['href']
                    normalized = self.normalize_url(href, base_url)
                    
                    if normalized and self.is_valid_url(normalized):
                        new_links.add(normalized)
                except:
                    continue
            
            # Dodaj nowe linki do kolejki
            for link in new_links:
                if link not in self.visited_urls and link not in self.error_urls:
                    # Sprawdź czy już w kolejce
                    in_queue = any(link == url for _, url in self.to_visit)
                    if not in_queue:
                        priority = self.get_url_priority(link)
                        self.to_visit.append((priority, link))
                        links_found += 1
            
            # Posortuj kolejkę
            if self.to_visit:
                self.to_visit = deque(sorted(self.to_visit, key=lambda x: x[0]))
            
            if links_found > 0:
                self.logger.debug(f"Znaleziono {links_found} linków")
                
        except Exception as e:
            self.logger.error(f"Błąd przy linkach: {e}")
    
    def get_url_priority(self, url):
        """Określa priorytet URL-a"""
        url_lower = url.lower()
        
        # Priorytet 1: Modele i specyfikacje
        if any(pattern in url_lower for pattern in 
              ['/models/', '/seria', '/series/', '/x1', '/x2', '/x3', '/x4', '/x5', '/x6', '/x7',
               '/i3', '/i4', '/i5', '/i7', '/ix', '/ix1', '/ix2', '/ix3',
               '/m2', '/m3', '/m4', '/m5', '/m8', '/z4']):
            return 1
        
        # Priorytet 2: Ważne sekcje
        if any(pattern in url_lower for pattern in 
              ['/configurator', '/configure', '/service', '/serwis',
               '/parts', '/czesci', '/offers', '/oferty']):
            return 2
        
        # Priorytet 3: Inne
        return 3
    
    def crawl(self):
        """Główna funkcja crawlowania"""
        self.logger.info("=" * 70)
        self.logger.info("🚀 INTELLIGENTNY BMW CRAWLER - START")
        self.logger.info(f"📁 Dane będą zapisane w: {self.output_dir}")
        self.logger.info("=" * 70)
        
        start_time = time.time()
        
        try:
            # 1. Zacznij od priorytetowych URL-i
            self.logger.info("\n🎯 Rozpoczynam od priorytetowych stron...")
            for url in self.priority_urls:
                if self.is_valid_url(url):
                    self.to_visit.append((1, url))
            
            # 2. Główna pętla crawlowania
            max_pages = 100
            pages_scraped = 0
            consecutive_errors = 0
            max_consecutive_errors = 10  # Zwiększone
            
            while self.to_visit and pages_scraped < max_pages and consecutive_errors < max_consecutive_errors:
                try:
                    # Pobierz następny URL
                    if not self.to_visit:
                        break
                    
                    priority, url = self.to_visit.popleft()
                    
                    # Scrapuj stronę
                    result = self.scrape_page(url)
                    if result:
                        pages_scraped += 1
                        consecutive_errors = 0
                        
                        # Co 3 strony pokaż status
                        if pages_scraped % 3 == 0:
                            self.log_status(pages_scraped, start_time)
                    else:
                        consecutive_errors += 1
                        if consecutive_errors >= 3:
                            self.logger.warning(f"Kolejne błędy: {consecutive_errors}/{max_consecutive_errors}")
                    
                    # Rate limiting
                    time.sleep(1.5 if priority == 1 else 1.0)
                    
                except KeyboardInterrupt:
                    self.logger.info("\n🛑 Zatrzymano przez użytkownika")
                    break
                except Exception as e:
                    self.logger.error(f"Błąd w pętli: {e}")
                    consecutive_errors += 1
                    time.sleep(2)  # Dłuższa przerwa przy błędzie
            
            # Podsumowanie
            if consecutive_errors >= max_consecutive_errors:
                self.logger.warning(f"⚠️ Zatrzymano po {max_consecutive_errors} błędach")
            
            self.logger.info(f"\n✅ Zakończono. Stron: {pages_scraped}")
            self.logger.info(f"   🔗 Odwiedzone: {len(self.visited_urls)}")
            self.logger.info(f"   ❌ Błędy: {len(self.error_urls)}")
            
        except Exception as e:
            self.logger.error(f"\n❌ Nieoczekiwany błąd: {e}")
        finally:
            # Zawsze zapisz dane
            self.save_all_data()
            self.print_summary(start_time)
            
            # Zamknij przeglądarkę
            try:
                self.driver.quit()
            except:
                pass
    
    def save_all_data(self):
        """Zapisuje wszystkie zebrane dane"""
        if not self.scraped_data:
            self.logger.warning("Brak danych do zapisania!")
            return
        
        try:
            # 1. Wszystkie dane (JSONL)
            all_file = self.output_dir / "all_data.jsonl"
            with open(all_file, 'w', encoding='utf-8') as f:
                for item in self.scraped_data:
                    json.dump(item, f, ensure_ascii=False)
                    f.write('\n')
            
            # 2. Modele
            if self.model_pages:
                models_file = self.output_dir / "models_data.json"
                with open(models_file, 'w', encoding='utf-8') as f:
                    json.dump(self.model_pages, f, ensure_ascii=False, indent=2)
            
            # 3. Inne
            if self.other_pages:
                other_file = self.output_dir / "other_data.json"
                with open(other_file, 'w', encoding='utf-8') as f:
                    json.dump(self.other_pages, f, ensure_ascii=False, indent=2)
            
            # 4. Statystyki
            stats = self.calculate_statistics()
            stats_file = self.output_dir / "stats.json"
            with open(stats_file, 'w', encoding='utf-8') as f:
                json.dump(stats, f, ensure_ascii=False, indent=2)
            
            # 5. Podsumowanie
            summary_file = self.output_dir / "summary.txt"
            with open(summary_file, 'w', encoding='utf-8') as f:
                f.write(self.generate_summary_text(stats))
            
            self.logger.info(f"\n💾 Dane zapisane w: {self.output_dir}")
            
        except Exception as e:
            self.logger.error(f"Błąd przy zapisie: {e}")
    
    def calculate_statistics(self):
        """Oblicza statystyki"""
        stats = {
            'total_pages': len(self.scraped_data),
            'model_pages': len(self.model_pages),
            'other_pages': len(self.other_pages),
            'visited_urls': len(self.visited_urls),
            'error_urls': len(self.error_urls),
            'total_words': sum(p.get('word_count', 0) for p in self.scraped_data),
            'output_dir': str(self.output_dir),
            'scraped_at': datetime.now().isoformat()
        }
        
        # Średnie
        if self.scraped_data:
            stats['avg_words_per_page'] = stats['total_words'] / stats['total_pages']
        
        # Kategorie
        stats['categories'] = {}
        for page in self.scraped_data:
            for category in page.get('categories', []):
                stats['categories'][category] = stats['categories'].get(category, 0) + 1
        
        # Modele
        all_models = []
        for page in self.model_pages:
            all_models.extend(page.get('detected_models', []))
        
        if all_models:
            from collections import Counter
            model_counts = Counter(all_models)
            stats['models_found'] = dict(model_counts.most_common(15))
            stats['unique_models'] = len(set(all_models))
        
        return stats
    
    def generate_summary_text(self, stats):
        """Generuje tekstowe podsumowanie"""
        summary = []
        summary.append("=" * 70)
        summary.append("🎉 BMW CRAWLER - PODSUMOWANIE")
        summary.append("=" * 70)
        summary.append(f"\n📊 STATYSTYKI:")
        summary.append(f"  Stron zebranych: {stats['total_pages']}")
        summary.append(f"  Stron z modelami: {stats['model_pages']}")
        summary.append(f"  Innych stron: {stats['other_pages']}")
        summary.append(f"  Słowa ogółem: {stats['total_words']:,}")
        if 'avg_words_per_page' in stats:
            summary.append(f"  Średnio słów/stronę: {stats['avg_words_per_page']:.0f}")
        
        if stats.get('models_found'):
            summary.append(f"\n🚗 MODELE:")
            summary.append(f"  Unikalne modele: {stats.get('unique_models', 0)}")
            for model, count in list(stats['models_found'].items())[:10]:
                summary.append(f"  {model}: {count} stron")
        
        summary.append(f"\n💾 DANE: {stats['output_dir']}")
        summary.append("=" * 70)
        
        return '\n'.join(summary)
    
    def log_status(self, pages_scraped, start_time):
        """Loguje status"""
        elapsed = time.time() - start_time
        elapsed_min = elapsed / 60
        
        self.logger.info("\n" + "=" * 50)
        self.logger.info(f"📊 STATUS: Strona {pages_scraped}")
        self.logger.info(f"⏱️ Czas: {elapsed_min:.1f} minut")
        self.logger.info(f"🚗 Modele: {len(self.model_pages)}")
        self.logger.info(f"📰 Inne: {len(self.other_pages)}")
        self.logger.info(f"🔗 Kolejka: {len(self.to_visit)}")
        self.logger.info("=" * 50)
    
    def print_summary(self, start_time):
        """Wyświetla podsumowanie"""
        stats = self.calculate_statistics()
        
        print("\n" + "=" * 70)
        print("🎉 INTELLIGENTNY BMW CRAWLER - PODSUMOWANIE")
        print("=" * 70)
        
        print(f"\n📊 STATYSTYKI:")
        print(f"  Stron zebranych: {stats['total_pages']}")
        print(f"  Stron z modelami: {stats['model_pages']}")
        print(f"  Innych stron: {stats['other_pages']}")
        print(f"  Słowa ogółem: {stats['total_words']:,}")
        
        elapsed = time.time() - start_time
        elapsed_min = elapsed / 60
        print(f"  ⏱️ Czas pracy: {elapsed_min:.1f} minut")
        
        if elapsed > 0 and stats['total_pages'] > 0:
            pages_per_min = stats['total_pages'] / elapsed_min
            print(f"  📈 Wydajność: {pages_per_min:.1f} stron/minutę")
        
        if stats.get('models_found'):
            print(f"\n🚗 NAJCZĘSTSZE MODELE:")
            for model, count in list(stats['models_found'].items())[:10]:
                print(f"  {model}: {count} stron")
        
        print(f"\n💾 DANE ZAPISANE W: {self.output_dir}")
        print("=" * 70)

# Uruchom
if __name__ == "__main__":
    crawler = BMWCrawler()
    crawler.crawl()