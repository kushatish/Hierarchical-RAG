from pathlib import Path
from tempfile import TemporaryDirectory
import streamlit as st
from core.engine import Assistant

st.set_page_config(page_title="Evidence QA", page_icon="📚", layout="wide")

@st.cache_resource
def assistant():
    return Assistant()

engine = assistant()
st.title("Evidence QA")
st.caption("Document answers with page-level evidence and version-aware caching")
st.info(f"Retrieval: {engine.settings.backend} · Answers: {engine.settings.generator}. "
        "Extractive mode returns matching source sentences; it does not use an LLM.")

with st.sidebar:
    st.header("Document library")
    uploaded = st.file_uploader("PDF, Markdown or text", type=["pdf", "md", "txt"])
    if st.button("Add / replace document", disabled=uploaded is None):
        try:
            if uploaded.size > 10 * 1024 * 1024:
                raise ValueError("Maximum file size is 10 MiB")
            with st.spinner("Reading document…"):
                with TemporaryDirectory() as directory:
                    path = Path(directory) / Path(uploaded.name).name
                    path.write_bytes(uploaded.getvalue())
                    result = engine.ingest_file(path)
            st.success("Indexed" if result["changed"] else "Document already up to date")
        except Exception as exc:
            st.error(f"Could not index document: {exc}")
    if st.button("Load fictional demo handbook"):
        engine.ingest_file(Path(__file__).parent / "data/demo/handbook.md")
        st.success("Demo loaded")
    for doc in engine.list_documents():
        st.write(doc["name"])
        if st.button("Delete", key=doc["id"]):
            engine.delete_document(doc["id"])
            st.rerun()
    st.caption("Same filename replaces the existing document and invalidates cached answers.")

query = st.text_input("Ask about your documents", max_chars=2000, placeholder="What is the minimum attendance requirement?")
semantic = st.checkbox("Enable experimental semantic cache", value=False,
                       help="Exact caching is enabled by default. Semantic similarity can confuse different questions; calibrate before enabling.")
if st.button("Ask", type="primary"):
    try:
        with st.spinner("Searching evidence…"):
            result = engine.ask(query, semantic_cache=semantic)
        st.session_state["last_answer"] = result
    except Exception as exc:
        st.error(f"Could not answer: {exc}")
if "last_answer" in st.session_state:
    result = st.session_state["last_answer"]
    current_revision = engine.revision()
    if result["revision"] != current_revision:
        st.warning("The document library changed. Ask again to refresh this answer.")
    st.write(result["answer"])
    cols = st.columns(3)
    cols[0].metric("Request latency", f'{result["latency_ms"]:.1f} ms')
    cols[1].metric("Cache", result["cache_type"])
    cols[2].metric("Document revision", result["revision"])
    for source in result["citations"]:
        with st.expander(f'[{source["id"]}] {source["source"]} · page {source["page"]}'):
            st.text(source["text"])
            st.caption("Retrieved evidence; citation identifiers alone do not prove that every claim is supported.")
