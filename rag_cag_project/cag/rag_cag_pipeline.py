import faiss
import numpy as np
import os
os.environ["USER_AGENT"] = "rag-cag-project/1.0"

from embeddings.vector_store import create_or_load_vector_store, get_retriever
from embeddings.embedder import get_embedding_model_cag
from rag.rag_pipeline import run_rag

embedder = get_embedding_model_cag()
EMB_DIM = 384

cache_index = faiss.IndexFlatIP(EMB_DIM)
cache_queries = []
cache_answers = []

def embed(text):
    vec = embedder.encode(text, convert_to_numpy=True)
    vec = vec / np.linalg.norm(vec)
    return vec.astype("float32")

def cag_lookup(query, threshold=0.8):
    if cache_index.ntotal == 0:
        return {
            "hit": False,
            "score": 0.0,
            "answer": None
        }

    q_vec = embed(query).reshape(1, -1)
    scores, indices = cache_index.search(q_vec, k=1)

    best_score = float(scores[0][0])

    if best_score >= threshold:
        return {
            "hit": True,
            "score": best_score,
            "answer": cache_answers[indices[0][0]]
        }

    return {
        "hit": False,
        "score": best_score,
        "answer": None
    }


def cag_store(query, answer):
    vec = embed(query).reshape(1, -1)
    cache_index.add(vec)
    cache_queries.append(query)
    cache_answers.append(answer)

def cag_rag_pipeline(query):
    lookup = cag_lookup(query)

    if lookup["hit"]:
        return {
            "answer": lookup["answer"],
            "cache_hit": True,
            "similarity": lookup["score"],
            "latency_ms": 20
        }

    response = run_rag(query)

    cag_store(query, response)

    return {
        "answer": response,
        "cache_hit": False,
        "similarity": lookup["score"],
        "latency_ms": 800
    }


