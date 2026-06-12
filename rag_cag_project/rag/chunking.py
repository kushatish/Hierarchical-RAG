from langchain_text_splitters import RecursiveCharacterTextSplitter

def chunk_documents(docs, chunk_size=100, overlap=20):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap
    )
    return splitter.split_documents(docs)