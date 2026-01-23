# run.py
import sys
from pathlib import Path

# Dodaj główny katalog do Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

if __name__ == "__main__":
    import uvicorn
    print("🚗 Starting BMW Assistant...")
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)