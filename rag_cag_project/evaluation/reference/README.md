# Reference-document benchmark

This collection contains 39 real CPython 3.12 standard-library docstrings, exported from a local Python 3.12.14 runtime. The original authors wrote the documentation independently of this project. `manifest.json` records each symbol and document hash; `PYTHON_LICENSE.txt` includes the interpreter's license and notices.

The 53 questions were written during project development: 43 have labelled source documents and expected answer phrases, and ten are deliberately unanswerable from this collection. The labels were checked against the source text automatically, but have not received independent human review. This is **development data, not a held-out test set**.

## Reproduce lexical retrieval

From `rag_cag_project`:

```bash
python -m evaluation.benchmark \
  --dataset evaluation/reference/questions.json \
  --documents evaluation/reference/documents \
  --output evaluation/reference/report.json
```

To rebuild the source snapshots with Python 3.12:

```bash
python evaluation/reference/build_corpus.py
```

Keep the original manifest and questions under version control. Docstrings from other interpreter releases may differ.

## Reproduce real CPU inference

Use Python 3.12:

```bash
pip install -r requirements-onnx.txt
python -m core.onnx_models --download
RAG_RUN_MODEL_TESTS=1 python -m unittest tests.test_onnx -v
python -m evaluation.benchmark --backend onnx --rerank \
  --dataset evaluation/reference/questions.json \
  --documents evaluation/reference/documents \
  --output evaluation/reference/onnx_report.json
```

The setup uses pinned quantized ONNX exports of `sentence-transformers/all-MiniLM-L6-v2` and `cross-encoder/ms-marco-MiniLM-L6-v2`. Model/tokenizer bytes are verified against committed revision hashes. The runtime never loads custom model Python code. The models are Apache-2.0 licensed; their source model-card links and artifact identities are in `core/models.lock.json` and the inference report.

## What the metrics mean

- **Document Recall@k:** fraction of labelled relevant source documents represented in the first k returned contexts. Multiple contexts from one document do not increase recall. This is document relevance, not passage recall or answer accuracy.
- **MRR@5:** reciprocal rank of the first labelled relevant source, averaged over answerable questions.
- **Answer phrase coverage:** fraction of expected phrases appearing in the answer after normalizing whitespace. It does not detect contradictions, extra unsupported claims or correct alternative wording.
- **Citation source precision:** fraction of cited sources matching the document labels, conditional on returning citations. This is not per-claim entailment.
- **Abstention:** fraction of labelled unanswerable questions receiving an insufficient-evidence response.

All ablations keep the source collection, question set, generator and word budget fixed. The reranker processes up to 20 child candidates; other configurations do not use it. Cold model initialization is included in the first relevant call, not reported as a separate startup measure. Single-run latency numbers are descriptive and do not have confidence intervals.

## Load-test protocol

The saved mixed-load reports used an isolated SQLite database populated with all 39 documents, WAL mode, 106 requests cycling through these 53 questions, and eight client threads. The uncached phase disabled answer caching; the cached phase began with an empty answer cache. The process-local rate limit was raised to 10,000/minute for the test so the limiter would not dominate the workload. The default 120/minute limit is tested separately.

With an API serving this collection:

```bash
python -m evaluation.load_test --dataset evaluation/reference/questions.json \
  --requests 106 --concurrency 8 --no-cache --output evaluation/reference/mixed_uncached_load.json
python -m evaluation.load_test --dataset evaluation/reference/questions.json \
  --requests 106 --concurrency 8 --output evaluation/reference/mixed_cached_load.json
```

Both runs completed without HTTP failures. Basic answerability expectations matched on 96 of 106 requests: a cited answer is expected for answerable questions, and abstention for unanswerable ones. That check does not judge the factual content of a cited answer. Cache writes can increase tail latency on this fast offline workload; caching is not an automatic performance win.

## Known failure patterns

`before_sentence_fix_onnx_report.json` preserves an earlier development run. It exposed sentence truncation at wrapped lines, which was fixed with a regression test. The current reports retain failures, including missed expected phrases, overly broad context and false abstentions.

The next useful evaluation is an independently reviewed question set on new documents, with per-claim support annotations and several timed trials. Do not tune parameters on that held-out set.
