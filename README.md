# Hierarchical RAG with Cache-Augmented Generation (CAG)

## Overview
This project implements a **Hierarchical Retrieval-Augmented Generation (RAG)** system enhanced with **Cache-Augmented Generation (CAG)** to reduce latency and cost for repeated or semantically similar queries.

Unlike standard RAG pipelines, this system:
- Uses hierarchical chunking for better context retrieval
- Introduces a semantic cache to reuse past LLM responses
- Falls back to full RAG only when cache misses occur

This makes the system faster, cheaper, and more scalable for real-world document question-answering use cases.


## Key Concepts Used
- Retrieval-Augmented Generation (RAG)
- Hierarchical Chunking
- Semantic Caching (CAG)
- Vector Similarity Search
- FAISS
- ChromaDB
- Large Language Models (LLMs)



## Swagger UI:
<img width="1848" height="859" alt="image" src="https://github.com/user-attachments/assets/0793b342-d81b-4622-bd8b-5605dee9b8bd" />

<img width="1804" height="863" alt="image" src="https://github.com/user-attachments/assets/5a8e540c-1be6-4aa4-af18-b19894391b25" />

---
## Streamlit UI:

<img width="1007" height="779" alt="image" src="https://github.com/user-attachments/assets/4352f162-012c-43f9-8030-dd43c75e2dbf" />

---

## Cache miss:
<img width="1074" height="659" alt="image" src="https://github.com/user-attachments/assets/19d9ff95-e239-48ce-86b2-a3ffe704e3c6" />

---

## Cache hit:
<img width="1143" height="784" alt="image" src="https://github.com/user-attachments/assets/1f1a9387-73f4-4ee7-984e-3bcdd2e83aee" />
