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
from bs4 import BeautifulSoup

# ============================================
# KONFIGURACJA - ZMIEN TYLKO TO!
# ============================================
OUTPUT_DIR = r"C:\Users\hellb\Documents\Chatbot_BMW\RAG\src\BMW models"
# ============================================

def setup_driver():
    """Konfiguruje driver Chrome"""
    chrome_options = Options()
    # chrome_options.add_argument("--headless")  # Odkomentuj jak nie chcesz widzieć przeglądarki
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--window-size=1920,1080")
    
    # Unikaj wykrycia jako bot
    chrome_options.add_argument("--disable-blink-features=AutomationControlled")
    chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
    chrome_options.add_experimental_option('useAutomationExtension', False)
    
    driver = webdriver.Chrome(options=chrome_options)
    driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
    
    return driver

def extract_table_data(html_content, version_name, version_code):
    """Wyciąga dane z tabel na stronie"""
    soup = BeautifulSoup(html_content, 'html.parser')
    
    # Dane dla tej wersji
    version_data = {
        'Model': version_name,
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
                
                # Oczyść dane
                key = re.sub(r'\s+', ' ', key).strip()
                value = re.sub(r'\s+', ' ', value).strip()
                
                # Zapisz tylko sensowne dane
                if key and value and len(key) > 1 and len(value) > 0:
                    # Usuń dziwne znaki
                    key = key.replace('\n', ' ').replace('\r', '').replace('\xa0', ' ')
                    value = value.replace('\n', ' ').replace('\r', '').replace('\xa0', ' ')
                    
                    # Ujednolić nazwy kluczy
                    key = key.replace('w kW (KM)', 'Moc')
                    key = key.replace('w Nm', 'Moment obrotowy')
                    
                    version_data[key] = value
    
    # Wyciągnij pierwszą linię z zużyciem paliwa (często poza tabelą)
    first_paragraph = soup.find('p')
    if first_paragraph:
        text = first_paragraph.get_text(strip=True)
        fuel_match = re.search(r'Zużycie energii.*?(\d+[.,]\d+\s*[-–]\s*\d+[.,]\d+)\s*l/100 km', text)
        if fuel_match:
            version_data['Zużycie paliwa (zakres)'] = fuel_match.group(1)
    
    return version_data

def scrape_model_versions(model_url, model_name):
    """Scrapuje wszystkie wersje danego modelu z dropdowna"""
    driver = setup_driver()
    all_data = []
    
    try:
        print(f"\n🌐 Otwieram stronę: {model_name}")
        driver.get(model_url)
        
        # Poczekaj na załadowanie strony
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.TAG_NAME, "body"))
        )
        
        # Znajdź dropdown z wersjami
        try:
            # Czasami dropdown jest w iframe lub ma specyficzną strukturę
            dropdown = WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "select"))
            )
            
            select = Select(dropdown)
            options = select.options
            
            print(f"📋 Znaleziono {len(options)} wersji:")
            
            # Dla każdej opcji, wybierz i scrapuj
            for i, option in enumerate(options):
                version_text = option.text.strip()
                
                # Pomiń puste lub "-- Wybierz --"
                if not version_text or version_text in ['- Wybierz -', '', 'Wybierz']:
                    continue
                
                print(f"  ➡️  {i+1}. {version_text}")
                
                try:
                    # Odśwież referencję do dropdowna
                    dropdown = driver.find_element(By.CSS_SELECTOR, "select")
                    select = Select(dropdown)
                    
                    # Wybierz opcję
                    select.select_by_index(i)
                    
                    # Poczekaj na załadowanie danych
                    time.sleep(2)
                    
                    # Pobierz kod wersji z URL
                    current_url = driver.current_url
                    version_code = current_url.split('/')[-1].replace('.bmw', '')
                    
                    # Wyciągnij dane
                    html = driver.page_source
                    data = extract_table_data(html, f"{model_name} - {version_text}", version_code)
                    data['URL'] = current_url
                    data['Model_bazowy'] = model_name
                    
                    all_data.append(data)
                    print(f"     ✅ Zebrano {len(data)} pól")
                    
                except Exception as e:
                    print(f"     ❌ Błąd: {str(e)}")
                    continue
                    
        except Exception as e:
            print(f"⚠️ Nie znaleziono dropdowna, zapisuję tylko główną stronę")
            html = driver.page_source
            data = extract_table_data(html, f"{model_name} - wersja podstawowa", "podstawowa")
            data['URL'] = model_url
            data['Model_bazowy'] = model_name
            all_data.append(data)
    
    finally:
        driver.quit()
    
    return all_data

def save_to_csv(data, model_name):
    """Zapisuje dane do pliku CSV w wskazanym folderze"""
    if not data:
        print(f"❌ Brak danych do zapisania dla {model_name}")
        return None
    
    # Upewnij się, że folder istnieje
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Generuj nazwę pliku
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_model_name = model_name.replace(' ', '_').replace('/', '_')
    filename = f"{safe_model_name}_{timestamp}.csv"
    filepath = os.path.join(OUTPUT_DIR, filename)
    
    # Zbierz wszystkie unikalne klucze
    all_keys = set()
    for item in data:
        all_keys.update(item.keys())
    
    # Uporządkuj klucze - najważniejsze na początku
    priority_keys = ['Model_bazowy', 'Model', 'Kod_wersji', 'Data_scrapowania', 'URL']
    other_keys = sorted([k for k in all_keys if k not in priority_keys])
    fieldnames = priority_keys + other_keys
    
    with open(filepath, 'w', newline='', encoding='utf-8-sig') as csvfile:  # utf-8-sig dla polskich znaków
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data)
    
    print(f"\n✅ Zapisano: {filepath}")
    print(f"   📊 Wersji: {len(data)}, kolumn: {len(fieldnames)}")
    
    return filepath

def main():
    print("=" * 70)
    print("🚀 SCRAPER BMW - WSZYSTKIE WERSJE MODELI")
    print("=" * 70)
    print(f"📁 Pliki będą zapisywane w: {OUTPUT_DIR}")
    print("=" * 70)
    
    # Lista modeli do zescrapowania (możesz dowolnie rozszerzać)
    models_to_scrape = [
        {
            'url': 'https://www.bmw.pl/pl/all-models/x-series/x7/bmw-x7-dane-techniczne.html',
            'name': 'BMW X7'
        },
        {
            'url': 'https://www.bmw.pl/pl/all-models/x-series/x5/bmw-x5-dane-techniczne.html',
            'name': 'BMW X5'
        },
        {
            'url': 'https://www.bmw.pl/pl/all-models/x-series/x3/bmw-x3-dane-techniczne.html',
            'name': 'BMW X3'
        },
        {
            'url': 'https://www.bmw.pl/pl/all-models/x-series/x1/bmw-x1-dane-techniczne.html',
            'name': 'BMW X1'
        },
        {
            'url': 'https://www.bmw.pl/pl/all-models/3-series/sedan/bmw-3-series-sedan-dane-techniczne.html',
            'name': 'BMW Seria 3 Sedan'
        },
        {
            'url': 'https://www.bmw.pl/pl/all-models/5-series/sedan/bmw-5-series-sedan-dane-techniczne.html',
            'name': 'BMW Seria 5 Sedan'
        }
    ]
    
    start_time = time.time()
    all_saved_files = []
    
    # Scrapuj każdy model z listy
    for model in models_to_scrape:
        print(f"\n{'='*60}")
        print(f"🔍 Przetwarzam: {model['name']}")
        print(f"{'='*60}")
        
        try:
            # Scrapuj wszystkie wersje modelu
            model_data = scrape_model_versions(model['url'], model['name'])
            
            # Zapisz do pliku
            if model_data:
                saved_file = save_to_csv(model_data, model['name'])
                if saved_file:
                    all_saved_files.append(saved_file)
            else:
                print(f"⚠️ Brak danych dla {model['name']}")
                
        except Exception as e:
            print(f"❌ Błąd dla modelu {model['name']}: {str(e)}")
            continue
        
        # Mała przerwa między modelami (żeby nie przeciążać serwera)
        time.sleep(3)
    
    # Podsumowanie
    elapsed = time.time() - start_time
    print("\n" + "="*70)
    print("✅ SCRAPING ZAKOŃCZONY!")
    print("="*70)
    print(f"📁 Zapisz pliki w folderze:\n   {OUTPUT_DIR}")
    print(f"\n📊 Podsumowanie:")
    print(f"   • Przetworzonych modeli: {len(models_to_scrape)}")
    print(f"   • Zapisz plików CSV: {len(all_saved_files)}")
    print(f"   • Czas wykonania: {elapsed/60:.1f} minut")
    print("\n📋 Lista zapisanych plików:")
    for i, file in enumerate(all_saved_files, 1):
        print(f"   {i}. {os.path.basename(file)}")
    print("="*70)

if __name__ == "__main__":
    main()