# What the project can demonstrate

These are measured local development results, not a production or held-out accuracy claim. The source collection contains **39 real CPython documentation snapshots** and **53 questions**: 43 answerable and ten unanswerable. Questions were written during development; the labels still need independent review.

## Retrieval and answer quality

| Configuration | Document Recall@1 | MRR@5 | Answer phrase coverage | Unanswerable abstention | p95 request ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| FTS5 / flat | 93.0% | 0.965 | 76.7% | 70% | 2.1 |
| FTS5 / parent–child | 93.0% | 0.965 | 76.7% | 70% | 2.5 |
| CPU embeddings / flat | 90.7% | 0.953 | 72.1% | 100% | 9.3 |
| CPU embeddings / parent–child | 90.7% | 0.953 | 74.4% | 100% | 8.9 |
| CPU embeddings + cross-encoder | 100.0% | 1.000 | 79.1% | 90% | 1065.2 |

The reranker finds the labelled source first for all 43 answerable cases. That does **not** mean every answer is correct: expected phrase coverage is about 79%, one of ten unanswerable questions still receives an answer, and source-ID precision does not verify claim entailment.

Reranking is substantially slower. The lexical path remains a useful default when short response times matter. Parent expansion has only a modest effect on this small collection; a larger corpus with cross-section questions is needed to assess it properly.

Both retrieval reports use Python 3.12 on the same host, with identical source files, questions, generator and word budgets. CPU inference uses two intra-op threads per ONNX session. Model warmup is included in the first relevant call; timings are single runs without confidence intervals. The mixed HTTP tests below use a different interpreter/runtime and should not be directly compared with these per-engine timings.

## Cached and uncached HTTP workload

The mixed workload cycles through all 53 questions twice, using eight client threads and SQLite WAL mode. Both runs completed **106/106 HTTP requests** without transport failures. The basic answerability check matched on **96/106** requests; this only checks whether a cited answer or abstention was expected, not factual correctness.

The cached run started empty. Cache writes still raised tail latency relative to uncached lexical search, even though median latency was lower. This fast offline workload is a reminder to measure caching on the actual request mix; a cache is not automatically faster. The test rate limit was raised to 10,000/minute, while the default 120/minute limiter is covered separately by tests.

## Semantic cache tradeoff

A separate set contains **32 labelled query pairs**, including genuine paraphrases, function-name changes, negations, boolean changes and numeric changes. The real embedding model produced false hits at lower thresholds. At threshold 0.8, precision was about 73%; at the conservative default 0.97, only one of 16 positive pairs was accepted and none of 16 negative pairs was accepted. Those small counts do not establish general safety.

Exact caching remains the default. Semantic reuse is opt-in, and threshold selection needs a larger, independently reviewed dataset.

## Engineering checks

All **56 automated tests pass on Python 3.12 with real models enabled**, including persistent-index reuse, embedding persistence, document replacement, concurrent cache access, duplicate-miss coalescing, bounded request admission, rate-limit expiry, API authentication, real-model API calls, and headless UI interactions.

Docker health checks, required API keys and container CI smoke tests are configured. Docker execution, remote workflows, Gemini calls and public hosting have not been validated locally.

## Reproduce and inspect

See [the reference evaluation guide](rag_cag_project/evaluation/reference/README.md) for commands and definitions.

- [Lexical results](rag_cag_project/evaluation/reference/report.json)
- [Real CPU model results and artifact identities](rag_cag_project/evaluation/reference/onnx_report.json)
- [Semantic threshold sweep](rag_cag_project/evaluation/reference/cache_calibration_report.json)
- [Uncached mixed HTTP results](rag_cag_project/evaluation/reference/mixed_uncached_load.json)
- [Cached mixed HTTP results](rag_cag_project/evaluation/reference/mixed_cached_load.json)

For resume claims, name the metric and dataset size explicitly. Use a new, independently reviewed test collection before presenting these development scores as evidence of generalization.
