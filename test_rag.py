"""Quick test script for RAG query scenarios"""
import httpx
import json
import sys

BASE = "http://localhost:8000"

queries = [
    "opowiedz mi o BMW X3",
    "Jakie wersje M4?",
    "M5 Touring jako auto rodzinne",
    "Countryman silniki",
    "opowiedz mi o leasingu BMW",
]

for q in queries:
    print(f"\n{'='*60}")
    print(f"QUERY: {q}")
    print(f"{'='*60}")
    try:
        r = httpx.get(f"{BASE}/test/query", params={"q": q}, timeout=60)
        data = r.json()
        
        response = data.get("response", "NO RESPONSE")
        rag_info = data.get("rag_info", {})
        
        print(f"RAG: has_data={rag_info.get('has_data')}, confidence={rag_info.get('confidence', 0):.3f}, docs={rag_info.get('documents_retrieved', 0)}")
        print(f"Models: {rag_info.get('detected_models', [])}")
        print(f"\nRESPONSE:\n{response[:500]}")
    except Exception as e:
        print(f"ERROR: {e}")