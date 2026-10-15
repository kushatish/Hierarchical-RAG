# Evaluation protocol

The checked-in dataset is a **development smoke set**, not a held-out benchmark. It covers a fictional campus handbook. Its purpose is to make regressions and limitations observable before collecting a real dataset.

## Reproduce

From `rag_cag_project`:

```bash
python -m evaluation.benchmark
```

`demo_report.json` records all answers, evidence sources, metrics and runtime. `before_abstention_report.json` preserves the initial run without the query-term abstention gate. That run abstained on only one of four unanswerable questions. The gate was added after examining those cases, so the later score is development-set performance, not independent validation.

The configurations are flat child retrieval, parent–child retrieval, and parent–child retrieval with a primed exact cache. Optional `--rerank` adds a cross-encoder configuration. The warm-cache experiment primes each question immediately before timing it. Unanswerable responses are not cached. Timing uses wall-clock request duration with a small sample, no concurrent load, and no provider generation.

## Local HTTP smoke load

Start the API after ingesting the demo, then run:

```bash
python -m evaluation.load_test --requests 100 --concurrency 8
```

The runner uses one repeated question and requires a cited response for success. This mostly exercises warm exact-cache access; it is not a mixed-query retrieval benchmark. `local_load_report.json` records the local run. Public-network performance and provider latency are not measured.

## Metrics

| Metric | Definition | What it does not prove |
| --- | --- | --- |
| Evidence phrase recall | Average fraction of expected answer phrases in retrieved context | Correct document ranking or Recall@k |
| Answer phrase coverage | Average fraction of expected phrases in the answer | Factual correctness, reasoning or entailment |
| Unanswerable abstention | Fraction of unanswerable questions receiving insufficient-evidence response | General hallucination resistance |
| p50 / p95 latency | Median and nearest-rank 95th percentile of measured request times | Production latency under load |
| Cache-hit precision | Equivalent labelled pairs among accepted cache hits | Safety on real paraphrases |

Run `python -m evaluation.calibrate_cache` to sweep thresholds independently of answer generation; results are written to `cache_calibration_report.json`. Precision/recall in that report measure labelled question equivalence, not answer correctness. The service threshold is not automatically changed.

The cache smoke pairs are in `cache_pairs.json`. Positive examples are only word reorderings. Add genuine paraphrases, entity changes, negations, temporal changes and numeric changes before threshold tuning.

## Build a defensible held-out benchmark

1. Choose a collection of documents you have permission to use and publish its document count, page count and content hashes.
2. Write 100–200 reviewed questions with expected answers and supporting document/page/passage IDs. Include at least 20% unanswerable cases and difficult multi-section questions.
3. Split development and test cases before tuning. Include a document-level split when testing generalization to unseen sources. Never tune against held-out scores.
4. Keep the corpus, generator, generation settings and context budget constant across ablations. Separate cold-model startup, cold answer-cache and warm answer-cache measurements.
5. Measure true passage Recall@k using supporting passage annotations; separately review factual correctness, per-claim citation support and abstention errors. LLM judges can supplement, not replace, reviewed cases.
6. Measure cache-hit precision and false-hit rate on non-equivalent pairs. Record threshold, workload hit mix, token consumption, model identifiers and pricing date for any cost estimate.
7. Run a concurrent load test and publish request count, concurrency, hardware, failures, throughput, and p50/p95 latency. Repeat trials and report variability.
8. Publish failure examples and raw outputs along with aggregate results.

The benchmark accepts an alternate dataset and document directory. Dataset format:

```json
[
  {
    "id": "q1",
    "question": "What is the attendance requirement?",
    "expected_phrases": ["75 percent"],
    "answerable": true
  }
]
```

This format supports phrase-based smoke metrics only. Add passage-ID annotations and a dedicated evaluator before claiming true Recall@k.

## Resume evidence checklist

Safe to describe after verification:

- Parent–child retrieval, optional cross-encoder reranking, source-page evidence.
- Persistent SQLite answer cache with TTL, capacity limits, corpus-version invalidation and atomic writes.
- FastAPI upload/query/delete endpoints, Streamlit UI, Docker packaging and deterministic tests.

Do not claim an accuracy percentage from substring checks, production scale from this demo, or cost savings without provider usage measurements.

Bullet templates for later measured results:

- Built a document QA API over **[N] documents / [P] pages**, combining parent–child retrieval, reranking and page-level citations.
- Improved **[defined retrieval metric]** from **[A] to [B]** versus **[baseline]** on **[N] held-out questions**.
- Reduced **[workload] p95 request latency by [X]%**, with **[Y]% cache-hit precision** over **[N] labelled pairs / requests**.
