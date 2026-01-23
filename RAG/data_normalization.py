import os
from pathlib import Path
from bs4 import BeautifulSoup
from datetime import datetime
import json
import re

# =========================
# KONFIGURACJA
# =========================

BASE_INPUT_DIR = Path(r"C:\Users\hellb\Documents\Chatbot_BMW\RAG\output")
timestamp = datetime.now().strftime("%d_%m_%Y_%H%M")
OUTPUT_DIR = Path(r"C:\Users\hellb\Documents\Chatbot_BMW\RAG\rag_chunks") / f"processed_{timestamp}"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

ALL_MODELS = []  # tutaj zbieramy wszystkie modele BMW

# =========================
# FUNKCJE POMOCNICZE
# =========================

def html_to_text(html: str) -> str:
    """Czyści HTML i usuwa JS/CSS"""
    soup = BeautifulSoup(html, "html.parser")
    for script in soup(["script", "style", "noscript"]):
        script.decompose()
    text = soup.get_text(separator="\n", strip=True)
    return text

def classify_category(text: str) -> str:
    """Klasyfikacja na modele, news lub support"""
    text_lower = text.lower()
    keywords_models = ["bmw", "silnik", "km", "moc", "zasięg", "typ nadwozia", "wyposażenie"]
    keywords_news = ["nowość", "premiera", "oferta", "event"]
    
    if any(k in text_lower for k in keywords_models):
        return "models"
    elif any(k in text_lower for k in keywords_news):
        return "news"
    else:
        return "support"

def extract_model_data(text: str, file_path: Path, scrap_folder: str) -> dict:
    """Wyciąga dane techniczne modelu BMW z tekstu"""
    model_info = {}

    # Nazwa modelu w pierwszej linii lub nagłówku
    first_line = text.strip().split("\n")[0]
    if "BMW" in first_line:
        model_info["model_name"] = first_line.strip()

    # Moc silnika
    km_match = re.search(r"(\d{2,4})\s*KM", text)
    if km_match:
        model_info["moc_silnika"] = km_match.group(0)

    # Typ nadwozia
    body_match = re.search(r"(SUV|sedan|kombi|cabrio|coupe|roadster)", text, re.I)
    if body_match:
        model_info["typ_nadwozia"] = body_match.group(0)

    # Zasięg
    range_match = re.search(r"(\d{1,4})\s*km", text, re.I)
    if range_match:
        model_info["zasieg"] = range_match.group(0)

    # Metadane źródłowe
    model_info["source_file"] = str(file_path.relative_to(BASE_INPUT_DIR))
    model_info["scrap_folder"] = scrap_folder

    return model_info

def split_text(text: str, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """Dzieli tekst na fragmenty z overlapem"""
    chunks = []
    start = 0
    text_len = len(text)
    while start < text_len:
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks

# =========================
# PRZETWARZANIE FOLDERU SCRAPU
# =========================

def process_scrap_folder(folder_path: Path):
    all_files = list(folder_path.rglob("*.html"))
    print(f"Znaleziono {len(all_files)} plików HTML w {folder_path}")
    chunk_counter = 0

    for file in all_files:
        html = file.read_text(encoding="utf-8")
        text = html_to_text(html)
        category = classify_category(text)

        metadata = {
            "category": category,
            "source_file": str(file.relative_to(BASE_INPUT_DIR)),
            "scrap_folder": str(folder_path.name)
        }

        if category == "models":
            model_data = extract_model_data(text, file, str(folder_path.name))
            metadata.update(model_data)
            ALL_MODELS.append(model_data)
        else:
            metadata["model_name"] = None

        chunks = split_text(text)
        for i, chunk in enumerate(chunks):
            # Chunk
            chunk_file = OUTPUT_DIR / f"{file.stem}_chunk{i}.txt"
            chunk_file.write_text(chunk, encoding="utf-8")

            # Metadane do JSON
            meta_file = OUTPUT_DIR / f"{file.stem}_chunk{i}_meta.json"
            meta_file.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

            chunk_counter += 1

    print(f"\n✅ Zapisano {chunk_counter} chunków w {OUTPUT_DIR}")

# =========================
# ENTRYPOINT
# =========================

if __name__ == "__main__":
    for scrap_folder in BASE_INPUT_DIR.iterdir():
        if scrap_folder.is_dir():
            process_scrap_folder(scrap_folder)

    # Zapis wszystkich modeli do jednego JSONa
    models_file = OUTPUT_DIR / "all_models.json"
    models_file.write_text(json.dumps(ALL_MODELS, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n✅ Zapisano wszystkie modele do {models_file}")
