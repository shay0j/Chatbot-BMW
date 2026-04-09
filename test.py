import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(r'C:\Users\oborowiec\Desktop\Chatbot-BMW')))

from app.services.rag_service import initialize_vector_store_from_csv
from app.services.rag_service import settings

async def init():
    csv_path = r'C:\Users\oborowiec\Desktop\Chatbot-BMW\RAG\RAG_sources\BMW_models.csv'
    print(f'📁 Plik: {csv_path}')
    
    success = await initialize_vector_store_from_csv(
        csv_path=csv_path,
        collection_name=settings.CHROMA_COLLECTION_NAME,
        force_recreate=True
    )
    
    if success:
        print('\n✅ BAZA WEKTOROWA GOTOWA!')
    else:
        print('\n❌ BŁĄD INICJALIZACJI')

asyncio.run(init())
