# Local validation

Updated on 15 October 2026. Tests use isolated temporary databases. The CPython reference corpus is independently authored source text; its question labels are development annotations, not an independently reviewed held-out set.

## Verified locally

- **56 tests pass with no skips on Python 3.12.14**, with pinned CPU models enabled. The suite includes core retrieval, persistent FTS5 indexing, vector persistence, concurrent access, duplicate-request coalescing, rate limits, API authentication, PDF extraction, UI interactions and real-model API upload/query/cache/delete operations.
- The standard tests also ran on Python 3.14.6; optional model tests skipped in that environment because no compatible ONNX wheel was available. Python 3.11 remains a configured CI target without a local run.
- The final Python 3.12 environment passed `pip check`. Project packages were installed into a temporary virtual environment; bundled runtime packages were reused read-only.
- Both pinned ONNX models ran real CPU inference. Artifact revisions, checksums, runtime versions and raw benchmark outputs are recorded in `evaluation/reference/onnx_report.json` and `core/models.lock.json`.
- A 53-question benchmark over 39 real documentation snapshots ran lexical, neural, parent–child and reranked ablations. It exposed answer-quality failures alongside strong source retrieval; see `BENCHMARKS.md`.
- A 32-pair semantic-cache sweep ran with real embeddings and recorded false-hit/recall tradeoffs.
- Cached and uncached mixed HTTP workloads each completed 106/106 requests at concurrency eight. Basic answerability expectations matched on 96/106; these were not factual-correctness judgments. WAL mode was used, and the test rate limit was explicitly raised to 10,000/minute.
- The local API and UI were restarted using the CPU embedding/reranking path. The user-facing document library remains empty until a document or the fictional demo is loaded. Real-model API tests use a separate temporary library.

## Not yet verified

- Gemini generation and the separate PyTorch/SentenceTransformer inference path have not been called with real models/providers. Provider contract tests use fakes.
- Docker is unavailable in the local environment. Container build/readiness/upload checks are configured in CI but have not run remotely.
- The manual model workflow is configured, including cacheable pinned downloads and a benchmark artifact; it has not run remotely.
- Public hosting has not been performed because no hosting target/account was supplied.
- Independent label review, a held-out corpus, per-claim entailment assessment, repeated performance trials and multi-replica operations remain future validation work.

The clean source ZIP includes scripts, tests, documentation, licensed source snapshots and reports. It excludes runtime databases, downloaded weights, virtual environments and secrets. Models can be downloaded explicitly using the committed lock file.
