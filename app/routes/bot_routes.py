# app/routes/bot_routes.py

from flask import Blueprint, request, jsonify
import sys, os

# Dodaj RAG do sys.path
RAG_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../RAG"))
if RAG_DIR not in sys.path:
    sys.path.append(RAG_DIR)

from rag_service import run_rag
from app.services.cohere_service import cohere_chat

bot_bp = Blueprint("bot_bp", __name__, url_prefix="/api")

@bot_bp.route("/ask", methods=["POST"])
def ask():
    try:
        data = request.get_json(force=True)
        if not data or "question" not in data:
            return jsonify({"error": "Missing 'question' field"}), 400

        user_question = data["question"].strip()
        if not user_question:
            return jsonify({"error": "Empty question"}), 400

        # ---------------------------
        # 1️⃣ RAG – próbujemy pobrać kontekst
        # ---------------------------
        rag_prompt = run_rag(user_question)  # zwraca prompt lub None

        # ---------------------------
        # 2️⃣ Końcowy prompt do LLM
        # ---------------------------
        # jeśli mamy RAG → używamy go w cohere_chat jako rag_prompt
        answer = cohere_chat(user_question, rag_prompt=rag_prompt)

        return jsonify({"answer": answer})

    except Exception as e:
        print(f"❌ ERROR in /api/ask: {e}")
        return jsonify({"error": str(e)}), 500
