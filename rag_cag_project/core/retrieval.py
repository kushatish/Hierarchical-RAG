"""Word-based parent/child retrieval with optional local neural models."""
from collections import Counter
from functools import lru_cache
import math
import re

STOP = set("a an the is are was were be been of to in on for and or with what which who how does do can please tell me about this that it as by from".split())

def tokens(text):
    return re.findall(r"[\w]+", text.lower())

TERM_ALIASES = {
    "minimum": "least", "maximum": "most",
    "requirement": "require", "requirements": "require", "required": "require", "requires": "require",
    "approval": "approve", "approves": "approve", "approved": "approve",
    "permission": "permit", "permissions": "permit",
}

def terms(text):
    return [TERM_ALIASES.get(word, word) for word in tokens(text) if word not in STOP]

def cosine(left, right):
    a, b = Counter(left), Counter(right)
    denominator = math.sqrt(sum(v*v for v in a.values()) * sum(v*v for v in b.values()))
    return sum(v * b.get(k, 0) for k, v in a.items()) / denominator if denominator else 0.0

def chunk_page(text, parent_size, child_size, overlap):
    # Preserve original whitespace/punctuation inside each span for verifiable citations.
    spans = list(re.finditer(r"\S+", text))
    for start in range(0, len(spans), parent_size):
        group = spans[start:start + parent_size]
        parent = text[group[0].start():group[-1].end()]
        for offset in range(0, len(group), child_size - overlap):
            child = group[offset:offset + child_size]
            yield start // parent_size, parent, text[child[0].start():child[-1].end()]
            if offset + child_size >= len(group):
                break

@lru_cache(maxsize=2)
def embedding_model():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

@lru_cache(maxsize=1)
def reranker_model():
    from sentence_transformers import CrossEncoder
    return CrossEncoder("BAAI/bge-reranker-base")

@lru_cache(maxsize=4)
def corpus_vectors(texts, backend="sentence-transformers"):
    return encoder_for(backend).encode(list(texts), normalize_embeddings=True)

def encoder_for(backend):
    if backend == "onnx":
        from core.onnx_models import get_model
        return get_model("embedding")
    return embedding_model()

def reranker_for(backend):
    if backend == "onnx":
        from core.onnx_models import get_model
        return get_model("reranker")
    return reranker_model()

def rank(query, rows, backend="lexical", rerank=False):
    if not rows:
        return []
    if backend in {"sentence-transformers", "onnx"}:
        model = encoder_for(backend)
        query_vector = model.encode(query, normalize_embeddings=True)
        vectors = corpus_vectors(tuple(r["text"] for r in rows), backend)
        scores = [float(query_vector @ vector) for vector in vectors]
    else:
        # BM25 gives rare terms more weight than ubiquitous words in the collection.
        corpus = [terms(row["text"]) for row in rows]
        query_terms = set(terms(query))
        avg_len = sum(map(len, corpus)) / len(corpus) or 1
        document_frequency = Counter(word for document in corpus for word in set(document))
        scores = []
        for document in corpus:
            counts = Counter(document)
            score = 0.0
            for word in query_terms:
                frequency = counts[word]
                idf = math.log(1 + (len(corpus) - document_frequency[word] + 0.5) / (document_frequency[word] + 0.5))
                score += idf * frequency * 2.5 / (frequency + 1.5 * (0.25 + 0.75 * len(document) / avg_len))
            scores.append(score)
    ranked = sorted([{**r, "score": s} for r, s in zip(rows, scores) if s > 0], key=lambda r: r["score"], reverse=True)
    if rerank and ranked:
        candidates = ranked[:20]
        scores = reranker_for(backend).predict([(query, row["text"]) for row in candidates])
        ranked = sorted([{**r, "score": float(s)} for r, s in zip(candidates, scores)], key=lambda r: r["score"], reverse=True)
    return ranked

def semantic_similarity(left, right, backend):
    if backend in {"sentence-transformers", "onnx"}:
        vecs = encoder_for(backend).encode([left, right], normalize_embeddings=True)
        return float(vecs[0] @ vecs[1])
    return cosine(terms(left), terms(right))

# A deliberately conservative guard, in addition to a calibrated similarity threshold.
# It is a heuristic, not a proof that questions are equivalent.
GUARDED = set("who what when where why how true false not no never without except only before after more less least most minimum maximum eligible ineligible allowed prohibited must may can cannot increase decrease".split())

def compatible_questions(left, right):
    a, b = set(tokens(left)), set(tokens(right))
    # Keep decimal and time literals intact: 8.5 and 5.8 must differ.
    numeric_pattern = r"[-+]?\d+(?:[.,:]\d+)*"
    numbers_a = set(re.findall(numeric_pattern, left))
    numbers_b = set(re.findall(numeric_pattern, right))
    return a & GUARDED == b & GUARDED and numbers_a == numbers_b
