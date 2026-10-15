# Resume and interview notes

## Suggested project title

**Evidence QA — Hierarchical Document Retrieval and Version-Aware Caching**

## Resume bullets supported by the current implementation

- Built a document QA service with persistent SQLite FTS5 search, parent–child retrieval, source citations, PDF ingestion, and FastAPI/Streamlit interfaces.
- Engineered version-aware answer caching with TTL, atomic writes, duplicate-request coalescing and bounded concurrency; verified 56 automated tests including real CPU inference and API/UI interactions.
- Improved document Recall@1 from 90.7% to 100% using a CPU cross-encoder on 43 answerable development questions across 39 CPython documentation snapshots; published latency, abstention and answer-coverage tradeoffs.

Keep the final resume entry to two or three bullets. The ONNX embedding/reranking path has real inference results. The 100% figure describes labelled source ranking on development data, not answer accuracy or held-out performance. Gemini and the separate PyTorch adapters remain unverified.

## Interview demonstration

1. Load the fictional handbook and ask the attendance question. Show the answer and source passage.
2. Repeat the question to show an exact-cache hit and measured latency.
3. Replace the handbook using the same filename with an updated attendance rule. Repeat the question and show that the old cache entry is invalidated.
4. Show that an old API source link returns a revision-conflict response instead of silently displaying new text.
5. Ask an unanswerable question and discuss the limitations of lexical abstention.
6. Explain why the tiny development dataset does not prove hierarchical retrieval is better than the baseline.

## Decisions worth explaining

- Small chunks support matching; parent expansion restores surrounding context under a fixed budget.
- SQLite stores query/answer/evidence atomically, avoiding separate mutable vector/answer arrays.
- A revision check prevents a slow generation from reintroducing stale answers after a document update.
- Exact matching is the default; semantic reuse is opt-in because similarity does not imply answer equivalence.
- Offline extractive mode keeps the core reproducible without provider credentials; neural models and generation load only when selected.
- Lexical retrieval uses a persistent FTS5 index and neural embeddings persist. Neural scoring still scans vectors; ANN search, tenant isolation and shared controls across replicas remain work for a larger service.

## Evidence still needed for stronger numerical bullets

Collect a real corpus and independent test annotations. Measure retrieval quality, per-claim citation support, cache false hits, token consumption, and mixed-query latency. Publish the baseline, configuration and failure cases alongside any improvement claim.
