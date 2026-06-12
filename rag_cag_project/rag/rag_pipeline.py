from langchain_community.document_loaders import WebBaseLoader
from sentence_transformers import CrossEncoder
from langchain_community.document_loaders import PyPDFLoader

import os
from dotenv import load_dotenv

from rag.chunking import chunk_documents
from embeddings.embedder import get_embedding_model_rag
from embeddings.vector_store import create_or_load_vector_store, get_retriever
from llm.generation import generate_answer

load_dotenv()  
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

docs = WebBaseLoader("https://en.wikipedia.org/wiki/Quantum_computing").load()

def load_pdfs(pdf_dir: str):
    docs = []
    for file in os.listdir(pdf_dir):
        if file.endswith(".pdf"):
            path = os.path.join(pdf_dir, file)
            loader = PyPDFLoader(path)
            docs.extend(loader.load())
    return docs

# from pathlib import Path

# BASE_DIR = Path(__file__).resolve().parent
# PDF_DIR = BASE_DIR / "data" / "raw" / "pdfs"

# PDF_DIR.mkdir(parents=True, exist_ok=True)
# pdf_docs = load_pdfs(str(PDF_DIR))

# docs = web_docs + pdf_docs

splits = chunk_documents(docs)

embeddings = get_embedding_model_rag()

vector_db = create_or_load_vector_store(
    docs=splits,
    embeddings=embeddings
)

retriever = get_retriever(vector_db, k=10)

def cross_encoder_rerank(query, chunks, top_k=5):
        """
        chunks: list of dicts
        [{ "id": ..., "text": ... }]
        """
        pairs = [(query, c["text"]) for c in chunks]
        scores = reranker.predict(pairs)

        reranked = sorted(
            zip(chunks, scores),
            key=lambda x: x[1],
            reverse=True
        )

        return [c for c, s in reranked[:top_k]]

reranker = CrossEncoder("BAAI/bge-reranker-base")
def run_rag(query: str) -> str:

    retrieved_docs = retriever.invoke(query)

    chunks = [
        {
            "id": i,
            "text": doc.page_content
        }
        for i, doc in enumerate(retrieved_docs)
    ]

    top_chunks = cross_encoder_rerank( query=query, chunks=chunks, top_k=5 )

    context = "\n".join(c["text"] for c in top_chunks)

    response = generate_answer(query, context)

    return response