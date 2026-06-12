from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import SentenceTransformer


embedding_model_rag = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)

embedding_model_cag = SentenceTransformer("all-MiniLM-L6-v2")

def get_embedding_model_rag():
    return embedding_model_rag

def get_embedding_model_cag():
    return embedding_model_cag