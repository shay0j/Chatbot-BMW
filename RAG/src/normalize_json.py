# RAG/src/normalize_json.py
import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
INPUT_FILE = BASE_DIR / "output/firecrawl_results.json"
OUTPUT_FILE = BASE_DIR / "output/firecrawl_normalized.json"

def normalize():
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    normalized = []
    for item in raw_data:
        normalized.append({
            "model": item.get("model", "Unknown"),
            "engine": item.get("engine", "Unknown"),
            "power": item.get("power", "Unknown"),
            "year": item.get("year", "Unknown")
        })

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(normalized, f, ensure_ascii=False, indent=2)

    print(f"Normalized JSON saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    normalize()
