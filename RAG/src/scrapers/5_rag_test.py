"""
📊 TESTY RETRIEVAL DLA BMW RAG - Z CHROMADB I COHERE (NOWA WERSJA)
======================================================================
Automatyczne testy jakości retrieval dla chatbot-a BMW doradcy klienta
Zaktualizowane do najnowszego ChromaDB API
"""

import sys
from pathlib import Path
import json
from typing import List, Dict, Any, Optional
import chromadb
import pandas as pd
from datetime import datetime

# Dodaj ścieżkę do src jeśli potrzebujesz
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root / "src"))

class BMWRAGTest:
    def __init__(self, chroma_path: str, collection_name: str = "bmw_docs"):
        """
        Inicjalizuje tester RAG z ChromaDB (NOWE API)
        
        Args:
            chroma_path: Ścieżka do bazy ChromaDB
            collection_name: Nazwa kolekcji
        """
        self.chroma_path = Path(chroma_path)
        self.collection_name = collection_name
        self.client = None
        self.collection = None
        
        print("======================================================================")
        print("🤖 BMW RAG TESTER - AUTOMATYCZNE TESTY JAKOŚCI (NEW CHROMA API)")
        print("======================================================================")
        
        self._initialize_client()
    
    def _initialize_client(self):
        """Inicjalizuje klienta ChromaDB z nowym API"""
        try:
            # NOWE API: PersistentClient zamiast Client z Settings
            self.client = chromadb.PersistentClient(path=str(self.chroma_path))
            print(f"✅ Połączono z ChromaDB (new API): {self.chroma_path}")
            
            # Pobierz lub utwórz kolekcję
            try:
                self.collection = self.client.get_collection(name=self.collection_name)
                print(f"✅ Załadowano istniejącą kolekcję: {self.collection_name}")
            except Exception as e:
                print(f"⚠️  Nie znaleziono kolekcji {self.collection_name}: {e}")
                print("   Tworzę nową kolekcję...")
                self.collection = self.client.create_collection(name=self.collection_name)
                print(f"✅ Utworzono nową kolekcję: {self.collection_name}")
            
            count = self.collection.count()
            print(f"   📊 Liczba dokumentów: {count}")
            
            if count == 0:
                print("⚠️  UWAGA: Kolekcja jest pusta!")
                print("   Uruchom najpierw 4_embeddings.py z opcją Cohere")
                return False
            
            return True
            
        except Exception as e:
            print(f"❌ Błąd inicjalizacji ChromaDB: {e}")
            print("\n🔧 ROZWIĄZANIE PROBLEMU:")
            print("1. Jeśli masz starą wersję bazy, zainstaluj narzędzie migracyjne:")
            print("   pip install chroma-migrate")
            print("2. Uruchom migrację:")
            print("   chroma-migrate")
            print("3. Lub usuń starą bazę i stwórz nową:")
            print(f"   rmdir /s /q {self.chroma_path}")
            return False
    
    def simple_query(self, query: str, n_results: int = 5, filter_models: List[str] = None):
        """
        Proste zapytanie do bazy z opcjonalnym filtrowaniem modeli
        
        Args:
            query: Tekst zapytania
            n_results: Liczba wyników
            filter_models: Lista modeli do filtrowania (np. ['X3', 'X5'])
        
        Returns:
            Lista wyników
        """
        print(f"\n🔍 ZAPYTANIE: '{query}'")
        print(f"   Filtry: {filter_models if filter_models else 'brak'}")
        print("-" * 80)
        
        try:
            # Przygotuj filtr dla metadanych
            where_filter = None
            if filter_models:
                # Filtruj po modelach - szukaj dokumentów zawierających którykolwiek z modeli
                # W ChromaDB nowe API: użyj operatora $in dla listy
                where_filter = {"models": {"$in": filter_models}}
            
            # Wykonaj zapytanie
            results = self.collection.query(
                query_texts=[query],
                n_results=n_results,
                where=where_filter
            )
            
            # Wyświetl wyniki
            if results and results['documents']:
                for i, (doc, meta, dist) in enumerate(zip(
                    results['documents'][0],
                    results['metadatas'][0],
                    results['distances'][0]
                )):
                    models = meta.get('models', [])
                    if isinstance(models, str):
                        models = [models] if models else []
                    
                    print(f"#{i+1} (dystans: {dist:.3f})")
                    print(f"   Modele: {', '.join(models) if models else 'brak'}")
                    print(f"   Priorytet: {meta.get('priority', 'brak')}")
                    print(f"   Tagi: {', '.join(meta.get('tags', [])) if meta.get('tags') else 'brak'}")
                    print(f"   Fragment: {doc[:200]}...")
                    print()
            else:
                print("❌ Brak wyników")
            
            return results
            
        except Exception as e:
            print(f"❌ Błąd zapytania: {e}")
            return None
    
    def smart_query(self, query: str, n_results: int = 5):
        """
        Inteligentne zapytanie z automatycznym wykrywaniem modeli
        """
        # Wykrywaj modele z zapytania
        bmw_models = ['X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7', 
                      'Serii 1', 'Serii 2', 'Serii 3', 'Serii 4', 'Serii 5', 'Serii 6', 'Serii 7',
                      'i3', 'i4', 'i5', 'i7', 'iX', 'iX1', 'iX3',
                      'M2', 'M3', 'M4', 'M5', 'M8', 'XM',
                      'Z4', 'M240i']
        
        detected_models = []
        query_lower = query.lower()
        
        for model in bmw_models:
            model_lower = model.lower()
            if model_lower in query_lower:
                detected_models.append(model)
        
        # Mapowanie potocznych nazw
        if "seria 1" in query_lower or "serii 1" in query_lower:
            detected_models.append("Serii 1")
        if "seria 3" in query_lower or "serii 3" in query_lower:
            detected_models.append("Serii 3")
        if "seria 5" in query_lower or "serii 5" in query_lower:
            detected_models.append("Serii 5")
        if "seria 7" in query_lower or "serii 7" in query_lower:
            detected_models.append("Serii 7")
        
        # Usuń duplikaty
        detected_models = list(set(detected_models))
        
        print(f"🤖 Wykryto modele: {detected_models if detected_models else 'żadnego'}")
        
        return self.simple_query(query, n_results, detected_models if detected_models else None)
    
    def run_test_suite(self):
        """
        Uruchamia pełną suitę testową
        """
        print("\n" + "="*80)
        print("🧪 PEŁNA SUITA TESTOWA RAG")
        print("="*80)
        
        test_cases = [
            # (zapytanie, oczekiwane_modele, opis)
            ("Ile kosztuje BMW X3?", ["X3"], "Cena konkretnego modelu"),
            ("Jaka jest moc silnika BMW X5?", ["X5"], "Specyfikacje techniczne"),
            ("Jakie są opcje leasingu dla i4?", ["i4"], "Finansowanie"),
            ("Gdzie mogę zrobić jazdę próbną X1?", ["X1"], "Test drive"),
            ("Jakie modele BMW są elektryczne?", [], "Modele elektryczne"),
            ("Ile wynosi rata miesięczna za X3?", ["X3"], "Finansowanie miesięczne"),
            ("Jaki jest zasięg BMW iX3?", ["iX3", "X3"], "Zasięg elektryczny"),
            ("Czy BMW X5 ma pakiet M Sport?", ["X5"], "Opcje wyposażenia"),
            ("Jakie kolory są dostępne dla Serii 3?", ["Serii 3"], "Personalizacja"),
            ("Jaka jest gwarancja na nowe BMW?", [], "Gwarancja"),
            ("Ile kosztuje serwis BMW X5?", ["X5"], "Koszty serwisowe"),
            ("Czy BMW ma program lojalnościowy?", [], "Programy klienta"),
            ("Jakie akcesoria mogę dokupić do X3?", ["X3"], "Akcesoria"),
            ("Jaka jest pojemność bagażnika X1?", ["X1"], "Specyfikacje praktyczne"),
            ("Czy BMW X3 jest dostępne jako hybrid?", ["X3"], "Warianty napędu"),
        ]
        
        results = []
        for i, (query, expected_models, description) in enumerate(test_cases, 1):
            print(f"\n📋 TEST {i}: {description}")
            print(f"   📝 Zapytanie: '{query}'")
            
            # Wykonaj zapytanie
            query_results = self.smart_query(query, n_results=3)
            
            # Analiza wyników
            if query_results and query_results['metadatas']:
                returned_models = []
                for meta in query_results['metadatas'][0]:
                    models = meta.get('models', [])
                    if isinstance(models, str) and models:
                        returned_models.append(models)
                    elif isinstance(models, list):
                        returned_models.extend([m for m in models if m])
                
                # Unikalne modele
                returned_models = list(set(returned_models))
                
                # Ocena trafności
                if expected_models:
                    # Sprawdź czy zwrócono oczekiwane modele
                    match_count = sum(1 for model in expected_models if model in returned_models)
                    relevance = "✅ TRAFNE" if match_count > 0 else "❌ NIETRAFNE"
                else:
                    # Dla pytań ogólnych oceniamy czy zwrócono jakiekolwiek wyniki
                    relevance = "✅ ZWRÓCONO" if returned_models else "❌ BRAK"
                
                results.append({
                    "test": i,
                    "zapytanie": query,
                    "opis": description,
                    "oczekiwane_modele": expected_models,
                    "zwrócone_modele": returned_models,
                    "trafność": relevance,
                    "liczba_wyników": len(query_results['metadatas'][0]),
                    "dystans": query_results['distances'][0][0] if query_results['distances'][0] else None
                })
                
                print(f"   {relevance}")
                if expected_models:
                    print(f"   Oczekiwane modele: {expected_models}")
                print(f"   Zwrócone modele: {returned_models}")
            else:
                print(f"   ❌ BRAK WYNIKÓW")
                results.append({
                    "test": i,
                    "zapytanie": query,
                    "opis": description,
                    "oczekiwane_modele": expected_models,
                    "zwrócone_modele": [],
                    "trafność": "❌ BRAK WYNIKÓW",
                    "liczba_wyników": 0,
                    "dystans": None
                })
        
        # Podsumowanie
        print("\n" + "="*80)
        print("📊 PODSUMOWANIE TESTOW")
        print("="*80)
        
        # Oblicz statystyki
        total_tests = len(results)
        successful = sum(1 for r in results if "✅" in r["trafność"])
        success_rate = (successful / total_tests) * 100 if total_tests > 0 else 0
        
        print(f"\n🎯 WYNIKI: {successful}/{total_tests} ({success_rate:.1f}%)")
        
        # Szczegółowe podsumowanie
        if results:
            df = pd.DataFrame(results)
            print("\n📋 SZCZEGÓŁOWA TABELA WYNIKÓW:")
            print(df[['test', 'opis', 'trafność', 'zwrócone_modele']].to_string(index=False))
        
        # Zapisz wyniki do pliku
        output_file = project_root / "output" / "rag_test_results.json"
        output_file.parent.mkdir(exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump({
                "data_generacji": datetime.now().isoformat(),
                "chroma_path": str(self.chroma_path),
                "collection": self.collection_name,
                "statystyki": {
                    "liczba_testów": total_tests,
                    "udane": successful,
                    "współczynnik_sukcesu": success_rate
                },
                "wyniki": results
            }, f, ensure_ascii=False, indent=2)
        
        print(f"\n💾 Zapisano wyniki do: {output_file}")
        
        return results
    
    def test_specific_scenarios(self):
        """
        Testuje specyficzne scenariusze dla doradcy klienta
        """
        print("\n" + "="*80)
        print("🎯 SPECJALNE SCENARIUSZE DLA DORADCY KLIENTA")
        print("="*80)
        
        scenarios = [
            {
                "name": "KLIENT PYTA O CENĘ",
                "queries": [
                    "Ile kosztuje BMW X3?",
                    "Jaka jest cena podstawowa X5?",
                    "Ile muszę zapłacić za i4?"
                ]
            },
            {
                "name": "KLIENT PYTA O FINANSOWANIE",
                "queries": [
                    "Jakie są raty leasingowe?",
                    "Czy jest opcja kredytu?",
                    "Ile wynosi opłata wstępna?"
                ]
            },
            {
                "name": "KLIENT PYTA O SPECYFIKACJE",
                "queries": [
                    "Jaka jest moc silnika?",
                    "Ile pali BMW X3?",
                    "Jaki jest zasięg elektryczny?"
                ]
            },
            {
                "name": "KLIENT PYTA O SERWIS",
                "queries": [
                    "Ile kosztuje przegląd?",
                    "Jaka jest gwarancja?",
                    "Czy jest assistance?"
                ]
            },
            {
                "name": "KLIENT PYTA O DOSTĘPNOŚĆ",
                "queries": [
                    "Kiedy dostawa nowego X5?",
                    "Czy X3 jest na miejscu?",
                    "Jak długo czeka się na i4?"
                ]
            }
        ]
        
        for scenario in scenarios:
            print(f"\n📁 SCENARIUSZ: {scenario['name']}")
            for query in scenario['queries']:
                print(f"  🔍 {query}")
                results = self.smart_query(query, n_results=2)
                
                if results and results['documents']:
                    # Analizuj czy wyniki są relewantne
                    doc = results['documents'][0][0][:100] + "..."
                    print(f"    → {doc}")
                else:
                    print(f"    → ❌ Brak wyników")
    
    def find_similar_chunks(self, text: str, n_results: int = 5):
        """
        Znajduje podobne chunki do podanego tekstu
        Przydatne do debugowania
        """
        print(f"\n🔍 SZUKAM PODOBNYCH DO TEKSTU:")
        print(f"   '{text[:100]}...'")
        
        results = self.collection.query(
            query_texts=[text],
            n_results=n_results
        )
        
        for i, (doc, meta, dist) in enumerate(zip(
            results['documents'][0],
            results['metadatas'][0],
            results['distances'][0]
        )):
            print(f"\n#{i+1} (dystans: {dist:.3f})")
            print(f"Modele: {meta.get('models', [])}")
            print(f"Tekst: {doc[:200]}...")
    
    def get_collection_stats(self):
        """Pokazuje statystyki kolekcji"""
        print("\n" + "="*80)
        print("📊 STATYSTYKI KOLEKCJI")
        print("="*80)
        
        count = self.collection.count()
        print(f"📁 Liczba dokumentów: {count}")
        
        if count == 0:
            print("❌ Kolekcja jest pusta!")
            return
        
        # Pobierz próbkę dokumentów do analizy
        try:
            sample = self.collection.peek(limit=5)
            
            if sample['metadatas']:
                print("\n📋 PRZYKŁADOWE METADANE:")
                for i, meta in enumerate(sample['metadatas']):
                    print(f"\nDokument #{i+1}:")
                    for key, value in meta.items():
                        if key != 'id':  # Pomijaj ID
                            print(f"  {key}: {value}")
            
            # Analiza tagów - pobierz wszystkie dokumenty
            all_results = self.collection.get(limit=min(100, count))
            tags_count = {}
            models_count = {}
            
            for meta in all_results['metadatas']:
                tags = meta.get('tags', [])
                if isinstance(tags, list):
                    for tag in tags:
                        if tag:  # Pomijaj puste
                            tags_count[tag] = tags_count.get(tag, 0) + 1
                elif tags:  # Jeśli to string
                    tags_count[tags] = tags_count.get(tags, 0) + 1
                
                models = meta.get('models', [])
                if isinstance(models, list):
                    for model in models:
                        if model:  # Pomijaj puste
                            models_count[model] = models_count.get(model, 0) + 1
                elif models:  # Jeśli to string
                    models_count[models] = models_count.get(models, 0) + 1
            
            print(f"\n🏷️ NAJCZĘSTSZE TAGI:")
            for tag, count in sorted(tags_count.items(), key=lambda x: x[1], reverse=True)[:10]:
                print(f"  {tag}: {count}")
            
            print(f"\n🚗 NAJCZĘSTSZE MODELE:")
            for model, count in sorted(models_count.items(), key=lambda x: x[1], reverse=True)[:10]:
                print(f"  {model}: {count}")
                
        except Exception as e:
            print(f"❌ Błąd pobierania statystyk: {e}")

def main():
    """Główna funkcja testująca"""
    # Ścieżka do bazy ChromaDB - DOSTOSUJ DO SWOJEJ ŚCIEŻKI!
    chroma_path = r"C:\Users\hellb\Documents\Chatbot_BMW\RAG\chroma_db_working"
    collection_name = "bmw_docs"
    
    # Inicjalizuj tester
    tester = BMWRAGTest(chroma_path, collection_name)
    
    if tester.collection is None or tester.collection.count() == 0:
        print("\n⚠️  PROBLEM: Kolekcja jest pusta lub nie można się połączyć")
        print("   Rozwiązania:")
        print("   1. Uruchom migrację: pip install chroma-migrate && chroma-migrate")
        print("   2. Lub usuń starą bazę i uruchom ponownie embeddery:")
        print(f"      rmdir /s /q {chroma_path}")
        print("      python 4_embeddings.py")
        return
    
    print("\n" + "="*80)
    print("🎮 MENU TESTOW RAG")
    print("="*80)
    print("1. Pojedyncze zapytanie (ręczne)")
    print("2. Pełna suita testów (15 pytań)")
    print("3. Scenariusze dla doradcy klienta")
    print("4. Znajdź podobne chunki")
    print("5. Statystyki kolekcji")
    print("6. Test z filtrowaniem modeli")
    print("0. Wyjście")
    
    while True:
        try:
            choice = input("\n📝 Wybierz opcję (0-6): ").strip()
            
            if choice == "0":
                print("👋 Zamykanie testera...")
                break
            
            elif choice == "1":
                query = input("🎯 Wpisz zapytanie: ").strip()
                if query:
                    tester.smart_query(query, n_results=5)
            
            elif choice == "2":
                tester.run_test_suite()
            
            elif choice == "3":
                tester.test_specific_scenarios()
            
            elif choice == "4":
                text = input("📝 Wpisz tekst do porównania: ").strip()
                if text:
                    tester.find_similar_chunks(text)
            
            elif choice == "5":
                tester.get_collection_stats()
            
            elif choice == "6":
                query = input("🎯 Wpisz zapytanie: ").strip()
                models_input = input("🚗 Wpisz modele do filtru (oddziel przecinkiem): ").strip()
                if query:
                    models = [m.strip() for m in models_input.split(",")] if models_input else None
                    tester.simple_query(query, n_results=5, filter_models=models)
            
            else:
                print("❌ Nieprawidłowy wybór")
                
        except KeyboardInterrupt:
            print("\n👋 Przerwano przez użytkownika")
            break
        except Exception as e:
            print(f"❌ Błąd: {e}")

if __name__ == "__main__":
    main()