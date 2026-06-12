import streamlit as st
import time

from cag.rag_cag_pipeline import cag_rag_pipeline   

st.set_page_config(
    page_title="RAG + CAG Assistant",
    page_icon="🧠",
    layout="centered"
)

st.markdown("<h1 style='text-align:center;'>🧠 RAG + CAG Assistant</h1>", unsafe_allow_html=True)
st.caption("FAISS-based semantic cache + RAG fallback")

st.divider()

query = st.text_input(
    "Ask a question",
    placeholder="e.g. Explain quantum computing"
)

if st.button("Submit", use_container_width=True):
    if not query.strip():
        st.warning("Please enter a query")
    else:
        with st.spinner("Thinking..."):
            result = cag_rag_pipeline(query)

        st.divider()

        # CACHE STATUS
        if result["cache_hit"]:
            st.success("⚡ Cache HIT")
        else:
            st.error("🐌 Cache MISS → RAG Triggered")

        col1, col2 = st.columns(2)
        col1.metric("Similarity", f"{result['similarity']:.2f}")
        col2.metric("Latency (ms)", result["latency_ms"])

        st.divider()
        st.subheader("Answer")
        st.write(result["answer"])

