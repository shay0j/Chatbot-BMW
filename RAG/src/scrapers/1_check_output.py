import json
from pathlib import Path
from collections import Counter
import pandas as pd

def analyze_output():
    """Analizuje zebrane dane"""
    
    # Znajdź najnowszy folder z danymi
    output_base = Path("C:/Users/hellb/Documents/Chatbot_BMW/RAG/output")
    folders = sorted([f for f in output_base.iterdir() if f.is_dir() and f.name.startswith("bmw")])
    
    if not folders:
        print("❌ Nie znaleziono folderów z danymi!")
        return
    
    latest_folder = folders[-1]
    print(f"📁 Analizuję najnowszy folder: {latest_folder.name}")
    
    # Ścieżki do plików
    models_file = latest_folder / "models_data.json"
    all_file = latest_folder / "all_data.jsonl"
    stats_file = latest_folder / "stats.json"
    
    # 1. Sprawdź statystyki
    if stats_file.exists():
        with open(stats_file, 'r', encoding='utf-8') as f:
            stats = json.load(f)
        
        print("\n" + "="*60)
        print("📊 STATYSTYKI ZBIERANIA")
        print("="*60)
        print(f"Stron ogółem: {stats.get('total_pages', 0)}")
        print(f"Stron z modelami: {stats.get('model_pages', 0)}")
        print(f"Innych stron: {stats.get('other_pages', 0)}")
        print(f"Łącznie słów: {stats.get('total_words', 0):,}")
        print(f"Średnio słów/stronę: {stats.get('avg_words_per_page', 0):.0f}")
        
        if 'categories' in stats:
            print("\n📂 KATEGORIE:")
            for cat, count in sorted(stats['categories'].items(), key=lambda x: x[1], reverse=True):
                print(f"  {cat}: {count} stron")
        
        if 'models_found' in stats:
            print("\n🚗 WYKRYTE MODELE:")
            for model, count in list(stats['models_found'].items())[:20]:
                print(f"  {model}: {count} stron")
    
    # 2. Sprawdź dane modeli
    if models_file.exists():
        with open(models_file, 'r', encoding='utf-8') as f:
            models_data = json.load(f)
        
        print(f"\n🎯 PRZYKŁADOWE DANE MODELI ({len(models_data)} stron):")
        
        for i, model in enumerate(models_data[:3]):
            print(f"\n{i+1}. {model.get('title', 'Brak tytułu')[:80]}...")
            print(f"   URL: {model.get('url', 'Brak URL')}")
            print(f"   Modele: {model.get('detected_models', [])}")
            print(f"   Słowa: {model.get('word_count', 0)}")
            print(f"   Kategorie: {model.get('categories', [])}")
            
            # Sprawdź specyfikacje
            specs = model.get('specifications', {})
            if specs:
                print("   Specyfikacje:")
                if specs.get('detected_models'):
                    print(f"     - Modele: {specs['detected_models']}")
                if specs.get('engine'):
                    print(f"     - Silnik: {specs['engine']}")
                if specs.get('prices'):
                    print(f"     - Ceny: {specs['prices'][:2]}")  # Pierwsze 2 ceny
    
    # 3. Sprawdź jakość treści
    if all_file.exists():
        word_counts = []
        with open(all_file, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                word_counts.append(data.get('word_count', 0))
        
        if word_counts:
            print(f"\n📈 ANALIZA JAKOŚCI:")
            print(f"   Min słów: {min(word_counts)}")
            print(f"   Max słów: {max(word_counts)}")
            print(f"   Avg słów: {sum(word_counts)/len(word_counts):.0f}")
            print(f"   Strony < 50 słów: {sum(1 for w in word_counts if w < 50)}")
            print(f"   Strony > 500 słów: {sum(1 for w in word_counts if w > 500)}")
    
    print("\n" + "="*60)

if __name__ == "__main__":
    analyze_output()