import asyncio
import sys
sys.path.append('.')
from app.services.rag_service import RAGService

async def test():
    rag = RAGService()
    
    test_queries = [
        ("Cześć!", True),
        ("BMW X3", False),
        ("BMW X5 cena", False),
        ("Seria 3", False),
        ("Ile kosztuje?", False),
        ("Moc silnika BMW X5", False),
    ]
    
    print("🧪 Testing fixed RAGService")
    print("=" * 60)
    
    for query, should_skip in test_queries:
        print(f"\nQuery: '{query}'")
        print("-" * 40)
        
        result = await rag.retrieve_with_intent_check(query)
        
        print(f"✓ Skip RAG: {result['skip_rag']} (expected: {should_skip})")
        
        if not result['skip_rag']:
            print(f"✓ Has data: {result['has_data']}")
            print(f"✓ Confidence: {result['confidence']:.3f}")
            print(f"✓ Threshold used: {result.get('confidence_threshold_used', 0.6):.3f}")
            print(f"✓ Intent: {result['intent']}")
            print(f"✓ Detected models: {result['detected_models']}")
            print(f"✓ Docs found: {result.get('documents_retrieved', 0)}")
            print(f"✓ Docs returned: {len(result['documents'])}")
            
            if result['documents']:
                print(f"\nTop document:")
                doc = result['documents'][0]
                print(f"  Score: {doc['score']:.3f}")
                print(f"  Preview: {doc['content'][:100]}...")
    
    print("\n" + "=" * 60)
    print("✅ Test completed!")

if __name__ == "__main__":
    asyncio.run(test())