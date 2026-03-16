import time
import csv
import re
import os
from datetime import datetime
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import Select
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse

# ============================================
# KONFIGURACJA
# ============================================
OUTPUT_DIR = r"C:\Users\hellb\Documents\Chatbot_BMW\RAG\src\BMW models"
MAIN_URL = "https://www.bmw.pl/pl/all-models.html"
# ============================================

def setup_driver():
    """Konfiguruje driver Chrome"""
    chrome_options = Options()
    # chrome_options.add_argument("--headless")  # Odkomentuj, gdy wszystko będzie działać
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--window-size=1920,1080")
    chrome_options.add_argument("--disable-blink-features=AutomationControlled")
    chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
    chrome_options.add_experimental_option('useAutomationExtension', False)
    
    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=chrome_options)
    driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
    return driver

def get_all_model_links(driver):
    """Pobiera linki do wszystkich modeli ze strony głównej all-models.html"""
    print("\n🔍 Szukam wszystkich modeli na stronie głównej...")
    driver.get(MAIN_URL)
    time.sleep(5)  # Czas na załadowanie JavaScriptu
    
    model_links = []
    
    # Próba 1: Znajdź wszystkie karty modeli
    selectors = [
        "a[href*='/all-models/']",  # Linki zawierające /all-models/
        "a[href*='/pl/all-models/']",
        "div[class*='model-card'] a",  # Karty modeli
        "div[class*='card'] a",
        "a[class*='model']"
    ]
    
    for selector in selectors:
        links = driver.find_elements(By.CSS_SELECTOR, selector)
        for link in links:
            href = link.get_attribute('href')
            if href and '/all-models/' in href and 'dane-techniczne' not in href:
                # Filtrujemy tylko linki do modeli, a nie do wersji
                if href not in [l['url'] for l in model_links]:
                    model_name = link.text.strip() or href.split('/')[-2] if href.split('/')[-2] else "Nieznany"
                    model_links.append({
                        'url': href,
                        'name': model_name
                    })
                    print(f"  ✅ Znaleziono: {model_name}")
    
    # Próba 2: Jeśli nie znaleziono, użyj BeautifulSoup do analizy HTML
    if not model_links:
        soup = BeautifulSoup(driver.page_source, 'html.parser')
        for a in soup.find_all('a', href=True):
            href = a['href']
            if '/all-models/' in href and 'dane-techniczne' not in href:
                full_url = urljoin(MAIN_URL, href)
                model_name = a.get_text(strip=True) or href.split('/')[-2] if href.split('/')[-2] else "Nieznany"
                if full_url not in [l['url'] for l in model_links]:
                    model_links.append({
                        'url': full_url,
                        'name': model_name
                    })
                    print(f"  ✅ Znaleziono: {model_name}")
    
    # Usuń duplikaty i modele koncepcyjne
    filtered_links = []
    seen_urls = set()
    for link in model_links:
        if link['url'] not in seen_urls and 'concept' not in link['url'].lower() and 'vision' not in link['url'].lower():
            seen_urls.add(link['url'])
            filtered_links.append(link)
    
    print(f"\n📊 Znaleziono łącznie {len(filtered_links)} modeli")
    return filtered_links

def find_technical_data_page(driver, model_url):
    """Znajduje podstronę z danymi technicznymi dla modelu"""
    print(f"  🔍 Szukam strony z danymi technicznymi...")
    driver.get(model_url)
    time.sleep(3)
    
    # Sprawdź czy jesteśmy już na stronie z danymi technicznymi
    if 'dane-techniczne' in driver.current_url:
        print(f"  ✅ Już na stronie danych technicznych")
        return driver.current_url
    
    # Szukaj linku do danych technicznych
    tech_selectors = [
        "a[href*='dane-techniczne']",
        "a[href*='technical-data']",
        "a[href*='specification']",
        "a:contains('Dane techniczne')",
        "a:contains('Specyfikacja')"
    ]
    
    for selector in tech_selectors:
        try:
            link = driver.find_element(By.CSS_SELECTOR, selector)
            tech_url = link.get_attribute('href')
            print(f"  ✅ Znaleziono link do danych technicznych")
            return tech_url
        except:
            continue
    
    # Spróbuj znaleźć w menu nawigacyjnym
    try:
        nav_links = driver.find_elements(By.CSS_SELECTOR, "nav a, [class*='navigation'] a")
        for link in nav_links:
            href = link.get_attribute('href')
            if href and ('dane-techniczne' in href or 'specs' in href):
                print(f"  ✅ Znaleziono w menu")
                return href
    except:
        pass
    
    # Jeśli nie znaleziono, spróbuj skonstruować URL
    parsed = urlparse(model_url)
    base_url = f"{parsed.scheme}://{parsed.netloc}"
    path_parts = parsed.path.split('/')
    
    # Usuń ewentualne fragmenty URL
    if path_parts[-1]:
        tech_url = model_url.rstrip('/') + '/dane-techniczne.html'
        print(f"  🔧 Próbuję skonstruowany URL: {tech_url}")
        return tech_url
    
    print(f"  ⚠️ Nie znaleziono strony z danymi technicznymi")
    return None

def extract_table_data(html_content, model_name, version_name, version_code):
    """Wyciąga dane z tabel na stronie"""
    soup = BeautifulSoup(html_content, 'html.parser')
    
    version_data = {
        'Model': f"{model_name} - {version_name}",
        'Kod_wersji': version_code,
        'Data_scrapowania': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    # Znajdź wszystkie tabele
    tables = soup.find_all('table')
    
    for table in tables:
        rows = table.find_all('tr')
        for row in rows:
            cells = row.find_all(['th', 'td'])
            if len(cells) >= 2:
                key = cells[0].get_text(strip=True)
                value = cells[1].get_text(strip=True)
                
                key = re.sub(r'\s+', ' ', key).strip()
                value = re.sub(r'\s+', ' ', value).strip()
                
                if key and value and len(key) > 1 and len(value) > 0:
                    key = key.replace('\n', ' ').replace('\r', '').replace('\xa0', ' ')
                    value = value.replace('\n', ' ').replace('\r', '').replace('\xa0', ' ')
                    
                    # Ujednolicenie nazw kluczy
                    if 'Moc w kW' in key or 'moc znamionowa' in key.lower():
                        key = 'Moc (KM/kW)'
                    elif 'Moment obrotowy' in key:
                        key = 'Moment obrotowy (Nm)'
                    elif 'Pojemność' in key and 'cm³' in value:
                        key = 'Pojemność silnika (cm³)'
                    elif 'Przyspieszenie' in key or '0-100' in key:
                        key = 'Przyspieszenie 0-100 km/h (s)'
                    elif 'Prędkość maksymalna' in key:
                        key = 'Prędkość maksymalna (km/h)'
                    elif 'Zużycie paliwa' in key or 'Zużycie energii' in key:
                        key = 'Zużycie paliwa (l/100km)'
                    elif 'Emisja CO₂' in key:
                        key = 'Emisja CO₂ (g/km)'
                    elif 'Długość' in key and 'mm' in value:
                        key = 'Długość (mm)'
                    elif 'Szerokość' in key and 'mm' in value:
                        key = 'Szerokość (mm)'
                    elif 'Wysokość' in key and 'mm' in value:
                        key = 'Wysokość (mm)'
                    elif 'Rozstaw osi' in key:
                        key = 'Rozstaw osi (mm)'
                    elif 'Pojemność bagażnika' in key:
                        key = 'Pojemność bagażnika (l)'
                    elif 'Masa własna' in key or 'Dopuszczalna masa' in key:
                        key = 'Masa (kg)'
                    
                    version_data[key] = value
    
    return version_data

def scrape_model_versions(tech_url, model_name):
    """Scrapuje wszystkie wersje danego modelu z dropdowna na stronie danych technicznych"""
    driver = setup_driver()
    all_data = []
    
    try:
        print(f"\n🌐 Otwieram stronę danych technicznych: {model_name}")
        driver.get(tech_url)
        WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.TAG_NAME, "body")))
        time.sleep(3)
        
        # Znajdź dropdown z wersjami
        try:
            selectors = ["select", "div[class*='variant-picker'] select", "div[class*='model-variant'] select"]
            dropdown = None
            for selector in selectors:
                try:
                    dropdown = driver.find_element(By.CSS_SELECTOR, selector)
                    break
                except:
                    continue
            
            if not dropdown:
                print("⚠️ Nie znaleziono dropdowna, zapisuję tylko główną stronę")
                html = driver.page_source
                data = extract_table_data(html, model_name, "wersja podstawowa", "podstawowa")
                data['URL'] = tech_url
                all_data.append(data)
                return all_data
            
            select = Select(dropdown)
            options = select.options
            print(f"📋 Znaleziono {len(options)} wersji")
            
            for i, option in enumerate(options):
                version_text = option.text.strip()
                if not version_text or version_text in ['- Wybierz -', '', 'Wybierz', 'Wybierz wersję']:
                    continue
                
                print(f"  ➡️  {i+1}. {version_text}")
                
                try:
                    dropdown = driver.find_element(By.CSS_SELECTOR, "select")
                    select = Select(dropdown)
                    select.select_by_index(i)
                    time.sleep(3)
                    
                    current_url = driver.current_url
                    version_code = current_url.split('/')[-1].replace('.bmw', '')
                    
                    html = driver.page_source
                    data = extract_table_data(html, model_name, version_text, version_code)
                    data['URL'] = current_url
                    all_data.append(data)
                    print(f"     ✅ Zebrano {len(data)} pól")
                    
                except Exception as e:
                    print(f"     ❌ Błąd: {e}")
                    continue
                    
        except Exception as e:
            print(f"⚠️ Błąd dropdowna: {e}")
            html = driver.page_source
            data = extract_table_data(html, model_name, "wersja podstawowa", "podstawowa")
            data['URL'] = tech_url
            all_data.append(data)
    
    finally:
        driver.quit()
    
    return all_data

def save_to_csv(all_model_data):
    """Zapisuje dane wszystkich modeli do osobnych plików CSV"""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    saved_files = []
    
    for model_name, data in all_model_data.items():
        if not data:
            continue
            
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_name = model_name.replace(' ', '_').replace('/', '_')[:50]
        filename = f"{safe_name}_{timestamp}.csv"
        filepath = os.path.join(OUTPUT_DIR, filename)
        
        all_keys = set()
        for item in data:
            all_keys.update(item.keys())
        
        priority_keys = ['Model', 'Kod_wersji', 'Data_scrapowania', 'URL']
        other_keys = sorted([k for k in all_keys if k not in priority_keys])
        fieldnames = priority_keys + other_keys
        
        with open(filepath, 'w', newline='', encoding='utf-8-sig') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(data)
        
        print(f"✅ Zapisano: {filename} ({len(data)} wersji)")
        saved_files.append(filepath)
    
    return saved_files

def main():
    print("=" * 70)
    print("🚀 SUPER SCRAPER - WSZYSTKIE MODELE BMW")
    print("=" * 70)
    print(f"📁 Pliki CSV będą w: {OUTPUT_DIR}")
    print("=" * 70)
    
    start_time = time.time()
    main_driver = setup_driver()
    
    try:
        # Krok 1: Pobierz wszystkie linki do modeli
        model_links = get_all_model_links(main_driver)
        
        if not model_links:
            print("❌ Nie znaleziono żadnych modeli!")
            return
        
        # Krok 2: Dla każdego modelu, znajdź stronę z danymi technicznymi
        all_model_data = {}
        
        for idx, model in enumerate(model_links, 1):
            print(f"\n{'='*60}")
            print(f"🔍 [{idx}/{len(model_links)}] Przetwarzam: {model['name']}")
            print(f"   URL: {model['url']}")
            
            try:
                # Znajdź stronę z danymi technicznymi
                tech_url = find_technical_data_page(main_driver, model['url'])
                
                if tech_url:
                    # Zbierz dane techniczne dla wszystkich wersji
                    model_data = scrape_model_versions(tech_url, model['name'])
                    if model_data:
                        all_model_data[model['name']] = model_data
                        print(f"✅ Zebrano {len(model_data)} wersji dla {model['name']}")
                    else:
                        print(f"⚠️ Brak danych dla {model['name']}")
                else:
                    print(f"⚠️ Nie znaleziono strony z danymi technicznymi dla {model['name']}")
                
            except Exception as e:
                print(f"❌ Błąd dla modelu {model['name']}: {e}")
                continue
            
            # Przerwa między modelami
            if idx < len(model_links):
                print("\n⏳ Oczekiwanie 5 sekund przed kolejnym modelem...")
                time.sleep(5)
        
        # Krok 3: Zapisz wszystkie dane
        saved_files = save_to_csv(all_model_data)
        
        # Podsumowanie
        elapsed = time.time() - start_time
        print("\n" + "="*70)
        print("✅ SUPER SCRAPING ZAKOŃCZONY!")
        print("="*70)
        print(f"📁 Folder: {OUTPUT_DIR}")
        print(f"\n📊 Podsumowanie:")
        print(f"   • Znalezionych modeli: {len(model_links)}")
        print(f"   • Przetworzonych modeli: {len(all_model_data)}")
        print(f"   • Zapisz plików CSV: {len(saved_files)}")
        print(f"   • Czas: {elapsed/60:.1f} minut")
        
    finally:
        main_driver.quit()

if __name__ == "__main__":
    main()