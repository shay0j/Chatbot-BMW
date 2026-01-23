from pathlib import Path
import json

BASE_DIR = Path(__file__).resolve().parent.parent
RAG_CHUNKS_DIR = BASE_DIR / "RAG" / "rag_chunks"

# Znajdź najnowszy folder
folders = [d for d in RAG_CHUNKS_DIR.iterdir() if d.is_dir()]
if not folders:
    print("ERROR: Brak folderów w rag_chunks")
    exit(1)

PROCESSED_FOLDER = max(folders, key=lambda d: d.stat().st_mtime)
print(f"Analizuję folder: {PROCESSED_FOLDER.name}")

# 1. Sprawdź all_models.json
ALL_MODELS_FILE = PROCESSED_FOLDER / "all_models.json"
if ALL_MODELS_FILE.exists():
    with open(ALL_MODELS_FILE, 'r', encoding='utf-8') as f:
        models = json.load(f)
    
    print(f"\nALL_MODELS.JSON - Liczba wpisów: {len(models)}")
    
    # Statystyki pól
    field_stats = {}
    for model in models:
        for key, value in model.items():
            if value and str(value).strip():  # Tylko niepuste wartości
                field_stats[key] = field_stats.get(key, 0) + 1
    
    print("Statystyki pól (niepuste):")
    for field, count in sorted(field_stats.items()):
        percentage = (count / len(models)) * 100
        print(f"  {field}: {count} ({percentage:.1f}%)")
    
    # Pokaż unikalne wartości dla kluczowych pól
    print("\nUnikalne wartości dla kluczowych pól:")
    
    # Zasięg
    zasiegi = set()
    for model in models:
        if zasieg := model.get('zasieg'):
            zasiegi.add(str(zasieg).strip())
    print(f"  zasieg: {sorted(list(zasiegi))[:20]}")  # Pierwsze 20
    
    # Typ nadwozia
    typy = set()
    for model in models:
        if typ := model.get('typ_nadwozia'):
            typy.add(str(typ).strip())
    print(f"  typ_nadwozia: {sorted(list(typy))[:20]}")
    
    # Moc silnika
    moce = set()
    for model in models:
        if moc := model.get('moc_silnika'):
            moce.add(str(moc).strip())
    print(f"  moc_silnika: {sorted(list(moce))[:20]}")
    
    # Pokaż przykłady z problemami
    print("\nPRZYKŁADY Z NIEPRAWIDŁOWYMI DANYMI:")
    problem_count = 0
    for i, model in enumerate(models[:50]):  # Sprawdź tylko pierwsze 50
        problems = []
        
        # Sprawdź zasięg
        zasieg = model.get('zasieg', '')
        zasieg_str = str(zasieg).strip()
        if zasieg_str in ['100 km', '000 km', '5000 km', '100\xa0km', '0 km', '']:
            problems.append(f"zasięg='{zasieg_str}'")
        
        # Sprawdź puste kluczowe pola
        if not model.get('model_name') or str(model.get('model_name')).strip() == '':
            problems.append("brak nazwy modelu")
        if not model.get('typ_nadwozia') or str(model.get('typ_nadwozia')).strip() == '':
            problems.append("brak typu nadwozia")
        if not model.get('moc_silnika') or str(model.get('moc_silnika')).strip() == '':
            problems.append("brak mocy")
        
        if problems:
            model_name = model.get('model_name', 'Brak nazwy')
            print(f"  {i}. {model_name[:50]}...")
            print(f"     Problemy: {', '.join(problems)}")
            print(f"     Source: {model.get('source_file', 'Brak')}")
            problem_count += 1
            
            if problem_count >= 10:  # Limit przykładów
                print("  ... i więcej")
                break
    
    # Statystyki problemów
    total_problems = sum(1 for model in models if not model.get('model_name') or str(model.get('model_name')).strip() == '')
    print(f"\nSTATYSTYKI PROBLEMÓW:")
    print(f"  Modele bez nazwy: {total_problems}/{len(models)} ({total_problems/len(models)*100:.1f}%)")
    
    # Modele z pełnymi danymi
    full_data = 0
    for model in models:
        if (model.get('model_name') and model.get('typ_nadwozia') and 
            model.get('moc_silnika') and model.get('zasieg') and 
            str(model.get('zasieg')).strip() not in ['100 km', '000 km', '5000 km', '100\xa0km', '0 km', '']):
            full_data += 1
    
    print(f"  Modele z pełnymi danymi: {full_data}/{len(models)} ({full_data/len(models)*100:.1f}%)")
    
else:
    print("Brak pliku all_models.json")

# 2. Sprawdź strukturę folderu
print(f"\nSTRUKTURA FOLDERU {PROCESSED_FOLDER.name}:")
txt_files = list(PROCESSED_FOLDER.rglob("*.txt"))
json_files = list(PROCESSED_FOLDER.rglob("*_meta.json"))
html_files = list(PROCESSED_FOLDER.rglob("*.html"))

print(f"  .txt files: {len(txt_files)}")
print(f"  _meta.json files: {len(json_files)}")
print(f"  .html files: {len(html_files)}")

# Sprawdź czy liczba się zgadza
if txt_files and json_files:
    if len(txt_files) == len(json_files):
        print("  ✓ Liczba .txt i _meta.json się zgadza")
    else:
        print(f"  ✗ Rozbieżność: .txt={len(txt_files)}, _meta.json={len(json_files)}")

# 3. Sprawdź przykładowy plik HTML i jego metadane
print("\nPRZYKŁADOWY HTML Z METADANAMI:")
if html_files:
    html_file = html_files[0]
    print(f"  Plik: {html_file.name}")
    
    # Znajdź odpowiadający mu meta.json
    meta_file = None
    for mf in json_files:
        if html_file.stem in mf.stem:
            meta_file = mf
            break
    
    if meta_file and meta_file.exists():
        with open(meta_file, 'r', encoding='utf-8') as f:
            meta = json.load(f)
        print(f"  Meta data:")
        for key, value in meta.items():
            print(f"    {key}: {value}")
        
        # Sprawdź czy ten model jest w all_models.json
        if ALL_MODELS_FILE.exists():
            model_name = meta.get('model_name')
            found = False
            for model in models:
                if model.get('model_name') == model_name:
                    found = True
                    print(f"  ✓ Znaleziono w all_models.json")
                    break
            if not found:
                print(f"  ✗ Nie znaleziono w all_models.json")