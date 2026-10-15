"""Public parent/child splitter; sizes are words, not characters."""
from core.retrieval import chunk_page

def chunk_documents(pages, parent_size=240, child_size=60, overlap=12):
    if not 0 <= overlap < child_size <= parent_size:
        raise ValueError("Invalid chunk sizes")
    return [
        {"page": page, "parent_id": parent_id, "parent": parent, "text": child}
        for page, text in enumerate(pages, 1)
        for parent_id, parent, child in chunk_page(text, parent_size, child_size, overlap)
    ]
