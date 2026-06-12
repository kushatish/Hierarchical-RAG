import os
from langchain_chroma import Chroma

def create_or_load_vector_store(docs, embeddings, persist_dir="./chroma_db"):
    print("CAG/RAG STEP A: entered create_or_load_vector_store")

    print("CAG/RAG STEP B: creating Chroma object")
    vector_store = Chroma(
        persist_directory=persist_dir,
        embedding_function=embeddings
    )
    print("CAG/RAG STEP C: Chroma object created")

    print("CAG/RAG STEP D: checking collection count")
    count = vector_store._collection.count()
    print(f"CAG/RAG STEP E: collection count = {count}")

    if count == 0:
        print("CAG/RAG STEP F: No vectors found. Indexing documents...")
        vector_store = Chroma.from_documents(
            documents=docs,
            embedding=embeddings,
            persist_directory=persist_dir
        )
        print("CAG/RAG STEP G: Indexing completed")
    else:
        print(f"CAG/RAG STEP F2: Loaded {count} vectors from Chroma")

    print("CAG/RAG STEP H: returning vector_store")
    return vector_store


def get_retriever(vector_store, k=10):
    return vector_store.as_retriever(
        search_kwargs={"k": k}
    )
