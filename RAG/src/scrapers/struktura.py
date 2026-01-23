from pathlib import Path
import os

BASE_DIR = Path(__file__).resolve().parent.parent

print("🔍 SZUKAM FOLDERU Z DANAMI BMW:")

# Sprawdź różne możliwe lokalizacje
possible_paths = [
    BASE_DIR / "output",
    BASE_DIR.parent / "output",
    BASE_DIR / "RAG" / "output",
    BASE_DIR.parent / "RAG" / "output",
    Path(r"C:\Users\hellb\Documents\Chatbot_BMW\RAG\output"),
    Path(r"C:\Users\hellb\Documents\Chatbot_BMW\output"),
]

for path in possible_paths:
    print(f"  📁 {path} - ", end="")
    if path.exists():
        print("✅ ISTNIEJE")
        # Pokaż zawartość
        items = list(path.iterdir())
        print(f"     Zawartość ({len(items)} items):")
        for item in items[:10]:  # Pierwsze 10
            if item.is_dir():
                print(f"       📂 {item.name}/")
            else:
                print(f"       📄 {item.name}")
        if len(items) > 10:
            print(f"       ... i {len(items)-10} więcej")
    else:
        print("❌ NIE ISTNIEJE")

# Sprawdź też aktualny folder
print(f"\n📁 AKTUALNY FOLDER: {Path.cwd()}")
print(f"📁 SKRYPT JEST W: {BASE_DIR}")

# Sprawdź czy crawler zapisuje gdzie indziej
print("\n🔎 SZUKAM PLIKÓW HTML W PROJEKCIE:")
html_files = list(BASE_DIR.rglob("*.html"))
print(f"  Znaleziono {len(html_files)} plików .html")
if html_files:
    for html in html_files[:5]:
        print(f"    📄 {html.relative_to(BASE_DIR)}")