"""
Pośrednik dla importu 6_rag_test.py
"""
import importlib.util
import sys
from pathlib import Path

def import_rag_system():
    """Importuje RAGSystem z 6_rag_test.py"""
    # Ścieżka do pliku
    rag_path = Path(__file__).parent / "6_rag_test.py"
    
    if not rag_path.exists():
        raise FileNotFoundError(f"Nie znaleziono pliku RAG: {rag_path}")
    
    # Dynamiczny import
    spec = importlib.util.spec_from_file_location("rag_system", rag_path)
    
    if spec is None:
        raise ImportError(f"Nie można załadować modułu z {rag_path}")
    
    module = importlib.util.module_from_spec(spec)
    sys.modules["rag_system"] = module
    
    try:
        spec.loader.exec_module(module)
        return module
    except Exception as e:
        raise ImportError(f"Błąd ładowania modułu RAG: {e}")

# Importuj przy załadowaniu modułu
try:
    rag_module = import_rag_system()
    RAGSystem = rag_module.RAGSystem
    find_latest_vector_db = rag_module.find_latest_vector_db
    print(f"✅ Załadowano RAG system z {Path(__file__).parent / '6_rag_test.py'}")
except Exception as e:
    print(f"❌ Błąd ładowania RAG: {e}")
    # Fallback classes
    class RAGSystem:
        def __init__(self, vector_db_path=None):
            pass
    def find_latest_vector_db():
        return None