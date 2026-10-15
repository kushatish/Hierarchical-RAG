"""Compatibility entry point; initialization is lazy and never fetches web pages."""
from functools import lru_cache
from core.engine import Assistant

@lru_cache(maxsize=1)
def get_assistant():
    return Assistant()

def cag_rag_pipeline(query):
    return get_assistant().ask(query)
