# quick_check.py
import json
from pathlib import Path

# Sprawdź największy chunk
chunks_path = Path(r"C:\Users\hellb\Documents\Chatbot_BMW\RAG\output\rag_ready_fixed_20260125_131230\all_chunks.jsonl")

sizes = []
with open(chunks_path, 'r', encoding='utf-8') as f:
    for line in f:
        chunk = json.loads(line)
        sizes.append((len(chunk['text']), chunk['id'], chunk['text'][:100]))

# Znajdź 5 największych
largest = sorted(sizes, reverse=True)[:5]
print("5 NAJWIĘKSZYCH CHUNKÓW:")
for size, chunk_id, preview in largest:
    print(f"\nID: {chunk_id}")
    print(f"Rozmiar: {size} znaków")
    print(f"Podgląd: {preview}...")