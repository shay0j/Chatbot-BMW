# RAG/src/test.py
from query_chroma import qa

questions = [
    "Jaki silnik ma BMW X5?",
    "Ile koni ma BMW 330i?",
]

for q in questions:
    print(f"Pytanie: {q}")
    print("Odpowiedź:", qa.run(q))
    print("---")
