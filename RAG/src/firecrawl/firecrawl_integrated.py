# RAG/src/firecrawl/firecrawl_integrated.py
import os
import json
import time
import asyncio
from pathlib import Path
from collections import deque
from urllib.parse import urlparse, urljoin
from dotenv import load_dotenv
import requests
from bs4 import BeautifulSoup
from playwright.async_api import async_playwright
from normalizer import normalize_specs
from config_sites import SITE_CONFIG  # konfiguracja dla stron statycznych

load_dotenv()

# --- Ścieżki z .env ---
BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_PATH = Path(os.getenv("FIRECRAWL_OUTPUT", BASE_DIR / "data/firecrawl_all.json"))
FAILED_PATH = Path(os.getenv("FAILED_URLS", BASE_DIR / "data/failed_urls.json"))
REQUEST_DELAY = float(os.getenv("REQUEST_DELAY", 1))
SOURCES = os.getenv("SOURCES", "").split(",")

SAVE_EVERY = 10
MAX_RETRIES = 10
MAX_DEPTH = None  # brak limitu głębokości
DYNAMIC_DOMAINS = ["mini.com.pl"]  # domeny wymagające JS (Playwright)

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
    output_path.parent.mkdir(exist_ok=True, parents=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

def save_failed_urls(failed_urls, output_path):
    output_path.parent.mkdir(exist_ok=True, parents=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(list(failed_urls), f, ensure_ascii=False, indent=2)

# --- statyczne strony ---
def fetch_with_retry(url, retries=MAX_RETRIES):
    for attempt in range(1, retries + 1):
        try:
            response = requests.get(url)
            response.raise_for_status()
            return response.text
        except Exception as e:
            wait_time = attempt * 2
            print(f"⚠️ Attempt {attempt}/{retries} failed for {url}: {e}, retry in {wait_time}s")
            time.sleep(wait_time)
    print(f"❌ Failed to fetch {url} after {retries} attempts.")
    return None

def crawl_static(url, domain):
    config = SITE_CONFIG.get(domain)
    html = fetch_with_retry(url)
    if not html:
        return None, []

    soup = BeautifulSoup(html, "html.parser")
    links = []
    for a in soup.find_all("a", href=True):
        joined = urljoin(url, a['href'])
        parsed = urlparse(joined)
        if domain in parsed.netloc:
            links.append(parsed.geturl())

    # Dane dla stron modeli
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
        data = {
            "url": url,
            "model": model_name,
            "description": description,
            "specs": specs,
            "specs_normalized": specs_normalized
        }
    else:
        title = soup.title.string.strip() if soup.title else ""
        text = ' '.join([p.get_text(strip=True) for p in soup.find_all('p')])
        h1 = [h.get_text(strip=True) for h in soup.find_all('h1')]
        h2 = [h.get_text(strip=True) for h in soup.find_all('h2')]
        images = [img.get('src') for img in soup.find_all('img') if img.get('src')]
        data = {
            "url": url,
            "title": title,
            "text": text,
            "metadata": {"h1": h1, "h2": h2, "images": images},
            "model": None,
            "specs": {},
            "specs_normalized": {}
        }

    return data, links

# --- dynamiczne strony ---
async def crawl_dynamic(page, url, domain):
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            print(f"🌐 Fetching dynamic page {url} (attempt {attempt}/{MAX_RETRIES})")
            await page.goto(url, timeout=120000)  # 120s timeout na pojedynczą próbę
            await page.wait_for_load_state('networkidle')
            content = await page.content()
            break
        except Exception as e:
            wait_time = attempt * 5  # progresywny backoff
            print(f"⚠️ Attempt {attempt}/{MAX_RETRIES} failed for {url}: {e}, retry in {wait_time}s")
            await asyncio.sleep(wait_time)
    else:
        print(f"❌ Failed to fetch {url} after {MAX_RETRIES} attempts.")
        return None, []

    soup = BeautifulSoup(content, "html.parser")
    links = []
    for a in soup.find_all("a", href=True):
        joined = urljoin(url, a['href'])
        parsed = urlparse(joined)
        if domain in parsed.netloc:
            links.append(parsed.geturl())

    title = soup.title.string.strip() if soup.title else ""
    text = ' '.join([p.get_text(strip=True) for p in soup.find_all('p')])
    h1 = [h.get_text(strip=True) for h in soup.find_all('h1')]
    h2 = [h.get_text(strip=True) for h in soup.find_all('h2')]
    images = [img.get('src') for img in soup.find_all('img') if img.get('src')]

    data = {
        "url": url,
        "title": title,
        "text": text,
        "metadata": {"h1": h1, "h2": h2, "images": images},
        "model": None,
        "specs": {},
        "specs_normalized": {}
    }

    return data, links

# --- główny crawler ---
async def crawl_all():
    all_results, visited_urls = load_checkpoint(OUTPUT_PATH)
    failed_urls = set()
    queue = deque([(url, 0) for url in SOURCES])
    processed = 0

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context()

        while queue:
            url, depth = queue.popleft()
            if url in visited_urls or (MAX_DEPTH is not None and depth > MAX_DEPTH):
                continue

            domain = urlparse(url).netloc
            page = None
            try:
                if any(d in domain for d in DYNAMIC_DOMAINS):
                    page = await context.new_page()
                    data, links = await crawl_dynamic(page, url, domain)
                else:
                    data, links = crawl_static(url, domain)

                if not data:
                    failed_urls.add(url)
                    save_failed_urls(failed_urls, FAILED_PATH)
                    continue

                all_results.append(data)
                visited_urls.add(url)
                processed += 1

                # dodaj linki do kolejki
                for link in links:
                    if link not in visited_urls:
                        queue.append((link, depth + 1))

                # checkpoint
                if processed % SAVE_EVERY == 0:
                    save_partial_results(all_results, OUTPUT_PATH)
                    save_failed_urls(failed_urls, FAILED_PATH)
                    queue_len = len(queue)
                    progress = processed / (processed + queue_len) * 100 if (processed + queue_len) else 0
                    print(f"💾 Saved progress | Visited: {processed}, Queue: {queue_len}, Progress: {progress:.2f}%")

                await asyncio.sleep(REQUEST_DELAY)

            except Exception as e:
                print(f"❌ Failed processing {url}: {e}")
                failed_urls.add(url)
                save_failed_urls(failed_urls, FAILED_PATH)
            finally:
                if page:
                    await page.close()

        save_partial_results(all_results, OUTPUT_PATH)
        save_failed_urls(failed_urls, FAILED_PATH)
        print(f"✅ Crawl finished. Results saved to {OUTPUT_PATH}")
        print(f"❌ Failed URLs saved to {FAILED_PATH}")
        await browser.close()

if __name__ == "__main__":
    asyncio.run(crawl_all())
