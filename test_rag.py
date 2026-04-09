import asyncio
from app.services.rag_service_faiss import get_rag_service

async def test():
    rag = await get_rag_service()
    results = await rag.retrieve_with_intent_check('katalogi modeli BMW', top_k=3)
    print('Znaleziono dokumentow:', len(results.get('documents', [])))
    for doc in results.get('documents', []):
        metadata = doc.get('metadata', {})
        print(f'  - Plik: {metadata.get("filename", "brak")}')
        print(f'    Tresc: {doc.get("content", "")[:100]}')

asyncio.run(test())