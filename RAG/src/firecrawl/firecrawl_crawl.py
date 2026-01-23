# RAG/src/firecrawl_crawl.py
import os
import json
import time
from pathlib import Path
from dotenv import load_dotenv
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse, urljoin
from collections import deque
from tqdm import tqdm  # pip install tqdm

from normalizer import normalize_specs
from config_sites import SITE_CONFIG

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_PATH = BASE_DIR / "data/firecrawl_results.json"
FAILED_PATH = BASE_DIR / "data/failed_urls.json"
REQUEST_DELAY = float(os.getenv("REQUEST_DELAY", 1))
SAVE_EVERY = 10
MAX_RETRIES = 10
MAX_DEPTH = None  # brak limitu głębokości

SOURCES = os.getenv("SOURCES", "https://www.bmw.pl").split(",")

# --- checkpoint ---
def load_checkpoint(output_path):
    if Path(output_path).exists():
        try:
            with open(output_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            visited_urls = {item["url"] for item in data}
            return data, visited_urls
        except json.JSONDecodeError:
            print(f"⚠️ Plik {output_path} jest pusty lub uszkodzony, tworzymy nowy...")
            return [], set()
    return [], set()

def save_partial_results(results, output_path):
    OUTPUT_PATH.parent.mkdir(exist_ok=True, parents=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

def save_failed_urls(failed_urls, output_path):
    OUTPUT_PATH.parent.mkdir(exist_ok=True, parents=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(list(failed_urls), f, ensure_ascii=False, indent=2)

# --- fetch z retry ---
def fetch_with_retry(url):
    for attempt in range(1, MAX_RETRIES+1):
        try:
            print(f"🌐 Fetching {url} (attempt {attempt})")
            response = requests.get(url)
            response.raise_for_status()
            return response
        except Exception as e:
            wait_time = attempt * 2
            print(f"⚠️ Attempt {attempt}/{MAX_RETRIES} failed for {url}: {e} | retrying in {wait_time}s")
            time.sleep(wait_time)
    print(f"❌ Failed to fetch {url} after {MAX_RETRIES} attempts.")
    return None

# --- crawl pojedynczej strony ---
def crawl_page(url, domain):
    response = fetch_with_retry(url)
    if not response:
        return None, []

    soup = BeautifulSoup(response.text, "html.parser")

    # Zbierz wszystkie linki w tej samej domenie/subdomenach
    links = []
    for a in soup.find_all("a", href=True):
        href = a['href']
        joined = urljoin(url, href)
        parsed = urlparse(joined)
        if domain in parsed.netloc:
            links.append(parsed.geturl())

    print(f"   🔗 Found {len(links)} links on {url}")
    if links:
        print(f"   First links: {links[:5]}")  # pokaż pierwsze 5 linków

    # Jeśli to strona modelu w SITE_CONFIG, scrapuj specjalnie
    config = SITE_CONFIG.get(domain)
    if config and any(part in url for part in ['/models/', 'model-tile']):
        model_name = soup.select_one(config['model_name']).get_text(strip=True) if soup.select_one(config['model_name']) else "Unknown"
        description = soup.select_one(config['description']).get_text(strip=True) if soup.select_one(config['description']) else ""
        specs = {}
        for row in soup.select(config['spec_table']):
            cells = row.find_all("td")
            if len(cells) == 2:
                key = cells[0].get_text(strip=True)
                value = cells[1].get_text(strip=True)
                specs[key] = value
        specs_normalized = normalize_specs(specs)
        return {
            "url": url,
            "model": model_name,
            "description": description,
            "specs": specs,
            "specs_normalized": specs_normalized
        }, links

    # Inne strony – zbierz cały tekst, tytuły, nagłówki i obrazy
    title = soup.title.string.strip() if soup.title else ""
    text = ' '.join([p.get_text(strip=True) for p in soup.find_all('p')])
    h1 = [h.get_text(strip=True) for h in soup.find_all('h1')]
    h2 = [h.get_text(strip=True) for h in soup.find_all('h2')]
    images = [img.get('src') for img in soup.find_all('img') if img.get('src')]

    return {
        "url": url,
        "title": title,
        "text": text,
        "metadata": {
            "h1": h1,
            "h2": h2,
            "images": images
        }
    }, links

# --- główny crawler ---
def crawl_sites():
    all_results, visited_urls = load_checkpoint(OUTPUT_PATH)
    failed_urls = set()
    queue = deque()

    for src in SOURCES:
        domain = urlparse(src).netloc
        queue.append((src, domain, 0))

    processed = 0
    pbar = tqdm(total=1, unit='page', ncols=100, desc='Crawling', dynamic_ncols=True)

    while queue:
        url, domain, depth = queue.popleft()
        if url in visited_urls or (MAX_DEPTH is not None and depth > MAX_DEPTH):
            continue

        processed += 1
        data, links = crawl_page(url, domain)

        if data:
            all_results.append(data)
            visited_urls.add(url)
        else:
            failed_urls.add(url)

        # dodaj nowe linki do kolejki
        new_links = 0
        for link in links:
            if link not in visited_urls:
                queue.append((link, domain, depth+1))
                new_links += 1

        # aktualizacja paska postępu
        pbar.total = processed + len(queue)
        pbar.update(1)
        pbar.set_postfix({
            'Visited': processed,
            'Queue': len(queue),
            'New': new_links
        })

        if processed % SAVE_EVERY == 0:
            save_partial_results(all_results, OUTPUT_PATH)
            save_failed_urls(failed_urls, FAILED_PATH)

        time.sleep(REQUEST_DELAY)

    pbar.close()
    save_partial_results(all_results, OUTPUT_PATH)
    save_failed_urls(failed_urls, FAILED_PATH)
    print(f"✅ Crawl finished. Results saved to {OUTPUT_PATH}")
    print(f"❌ Failed URLs saved to {FAILED_PATH}")

if __name__ == "__main__":
    crawl_sites()
