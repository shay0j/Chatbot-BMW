# RAG/src/query_chroma.py
from pathlib import Path
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
import json

BASE_DIR = Path(__file__).resolve().parent.parent
DB_PATH = BASE_DIR / "db"

# Load vector store
vectorstore = Chroma(persist_directory=str(DB_PATH), embedding_function=OpenAIEmbeddings())

retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

qa = RetrievalQA.from_chain_type(
    llm=OpenAI(temperature=0),
    chain_type="stuff",
    retriever=retriever
)

# Przykładowe pytanie
question = "Jaki silnik ma BMW X5?"
answer = qa.run(question)
print(answer)
