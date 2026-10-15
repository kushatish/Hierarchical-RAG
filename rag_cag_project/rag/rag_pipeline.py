from cag.rag_cag_pipeline import get_assistant

def run_rag(query: str) -> str:
    return get_assistant().ask(query, use_cache=False)["answer"]
