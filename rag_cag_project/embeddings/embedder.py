"""Optional embedding model; downloaded only when explicitly selected."""
from core.retrieval import embedding_model

def get_embedding_model_cag():
    return embedding_model()

def get_embedding_model_rag():
    return embedding_model()
