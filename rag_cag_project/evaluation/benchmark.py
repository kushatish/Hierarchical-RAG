"""Run: python -m evaluation.benchmark --output evaluation/demo_report.json.

Smoke metrics use labelled substring matches, not an LLM judge or a correctness
claim. The bundled fictional set is development data, NOT a held-out benchmark.
"""
import argparse
from dataclasses import asdict, replace
from pathlib import Path
from tempfile import TemporaryDirectory
import json
import math
import platform
import statistics
from hashlib import sha256

from core.config import Settings
from core.engine import Assistant, ENGINE_VERSION, PROMPT_VERSION

HERE = Path(__file__).resolve().parent

def percentile(values, fraction):
    ordered = sorted(values)
    return ordered[max(0, math.ceil(len(ordered) * fraction) - 1)]

def evaluate(engine, cases, hierarchy, cache=False):
    outputs, latencies, evidence_recall, answer_coverage, abstentions = [], [], [], [], []
    usage = {"input_tokens": 0, "output_tokens": 0}
    recall = {1: [], 3: [], 5: []}
    reciprocal_ranks, citation_precision, citation_coverage = [], [], []
    for case in cases:
        if cache:
            engine.ask(case["question"], hierarchy=hierarchy)
        result = engine.ask(case["question"], hierarchy=hierarchy, use_cache=cache)
        latencies.append(result["latency_ms"])
        for kind in usage:
            usage[kind] += result["usage"][kind]
        normalize = lambda text: " ".join(text.lower().split())
        evidence = normalize(" ".join(c["text"] for c in result["citations"]))
        if case["answerable"]:
            phrases = case["expected_phrases"]
            evidence_recall.append(sum(normalize(p) in evidence for p in phrases) / len(phrases))
            answer_coverage.append(sum(normalize(p) in normalize(result["answer"]) for p in phrases) / len(phrases))
            relevant = set(case.get("relevant_sources", []))
            if relevant:
                ranked = [c["source"] for c in result["citations"]]
                for k in recall:
                    recall[k].append(len(relevant & set(ranked[:k])) / len(relevant))
                first = next((i for i, name in enumerate(ranked[:5], 1) if name in relevant), None)
                reciprocal_ranks.append(1 / first if first else 0)
                cited = [c for c in result["citations"] if c["id"] in result["cited_ids"]]
                citation_coverage.append(bool(cited))
                if cited:
                    citation_precision.append(sum(c["source"] in relevant for c in cited) / len(cited))
        else:
            abstentions.append("insufficient evidence" in result["answer"].lower())
        outputs.append({"id": case["id"], "answer": result["answer"], "cache_type": result["cache_type"],
                        "sources": [{"source": c["source"], "page": c["page"]} for c in result["citations"]]})
    return {
        "document_recall_at": {str(k): statistics.mean(values) if values else None for k, values in recall.items()},
        "document_mrr_at_5": statistics.mean(reciprocal_ranks) if reciprocal_ranks else None,
        "citation_source_precision_when_cited": statistics.mean(citation_precision) if citation_precision else None,
        "answerable_citation_coverage": statistics.mean(citation_coverage) if citation_coverage else None,
        "queries": len(cases), "measured_request_generation_usage": usage,
        "evidence_phrase_recall": statistics.mean(evidence_recall) if evidence_recall else None,
        "answer_phrase_coverage": statistics.mean(answer_coverage) if answer_coverage else None,
        "unanswerable_abstention_rate": statistics.mean(abstentions) if abstentions else None,
        "latency_p50_ms": statistics.median(latencies), "latency_p95_ms": percentile(latencies, .95),
        "cache_hit_rate": sum(o["cache_type"] in {"exact", "semantic"} for o in outputs) / len(outputs),
        "outputs": outputs,
    }

def cache_evaluation(base_settings):
    pairs = json.loads((HERE / "cache_pairs.json").read_text())
    results = []
    for pair in pairs:
        with TemporaryDirectory() as directory:
            engine = Assistant(replace(base_settings, db_path=Path(directory) / "pairs.sqlite3"))
            engine.ingest_file(HERE.parent / "data/demo/handbook.md")
            engine.ask(pair["seed"])
            result = engine.ask(pair["candidate"], semantic_cache=True)
            results.append({**pair, "hit": result["cache_hit"], "similarity": result["similarity"]})
    hits = [p for p in results if p["hit"]]
    equivalent = [p for p in results if p["equivalent"]]
    return {"pairs": results, "hit_precision": sum(p["equivalent"] for p in hits) / len(hits) if hits else None,
            "equivalent_pair_hit_recall": sum(p["hit"] for p in equivalent) / len(equivalent),
            "note": "Tiny development smoke set. Two positive pairs are word reorderings, not a realistic paraphrase benchmark."}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=HERE / "demo_report.json")
    parser.add_argument("--dataset", type=Path, default=HERE / "demo_questions.json")
    parser.add_argument("--documents", type=Path, default=HERE.parent / "data/demo")
    parser.add_argument("--backend", choices=["lexical", "sentence-transformers", "onnx"], default="lexical")
    parser.add_argument("--generator", choices=["extractive", "gemini"], default="extractive")
    parser.add_argument("--cache-threshold", type=float, default=0.97)
    parser.add_argument("--rerank", action="store_true", help="Also run cross-encoder ablation; downloads optional models")
    args = parser.parse_args()
    cases = json.loads(args.dataset.read_text())
    if not cases:
        parser.error("Dataset must contain questions")
    report = {"purpose": "Development smoke benchmark; not held-out evidence for resume claims", "python": platform.python_version(),
              "backend": args.backend, "generator": args.generator, "dataset": args.dataset.name, "dataset_sha256": sha256(args.dataset.read_bytes()).hexdigest(),
              "engine_version": ENGINE_VERSION, "prompt_version": PROMPT_VERSION, "platform": platform.platform(),
              "metric_definitions": {"document_recall_at": "Fraction of annotated relevant source documents present in the first k returned contexts; only for cases with source labels",
                                     "document_mrr_at_5": "Mean reciprocal rank of the first annotated relevant source in the first five contexts",
                                     "citation_source_precision_when_cited": "Fraction of cited source IDs matching document labels, conditional on a cited answer; NOT per-claim entailment",
                                     "evidence_phrase_recall": "Mean fraction of gold phrases present in retrieved context for answerable questions; NOT document Recall@k",
                                     "answer_phrase_coverage": "Mean fraction of gold phrases in answer text; NOT factual correctness",
                                     "unanswerable_abstention_rate": "Fraction of labelled unanswerable questions returning insufficient evidence",
                                     "latency": "Local request wall time; model warmup not separated; small sample, not a load test. Token usage excludes cache priming requests."}}
    with TemporaryDirectory() as directory:
        settings = Settings(db_path=Path(directory) / "benchmark.sqlite3", backend=args.backend, generator=args.generator, cache_threshold=args.cache_threshold)
        if args.backend == "onnx":
            from core.onnx_models import LOCK, artifact_identity
            report["neural_artifacts"] = {"lock_sha256": artifact_identity(), "models": json.loads(LOCK.read_text())}
            from importlib.metadata import version
            report["model_runtime"] = {name: version(name) for name in ["onnxruntime", "tokenizers", "numpy"]}
        report["settings"] = {k: v for k, v in asdict(settings).items() if k != "db_path"}
        engine = Assistant(settings)
        paths = sorted(p for p in args.documents.iterdir() if p.suffix.lower() in {".pdf", ".txt", ".md"})
        if not paths:
            parser.error("No supported documents found")
        for path in paths:
            engine.ingest_file(path)
        _, snapshot = engine.snapshot()
        sources = {doc["name"]: " ".join(json.loads(doc["pages"])).lower() for doc in snapshot}
        for case in cases:
            if case["answerable"] and not case["expected_phrases"]:
                parser.error(f"Answerable case {case['id']} needs expected phrases")
            for source in case.get("relevant_sources", []):
                if source not in sources:
                    parser.error(f"Unknown source label {source}")
            if case.get("relevant_sources"):
                gold = " ".join(" ".join(sources[name].split()) for name in case["relevant_sources"])
                for phrase in case["expected_phrases"]:
                    if " ".join(phrase.lower().split()) not in gold:
                        parser.error(f"Expected phrase missing from labelled source for {case['id']}")
        report["documents"] = len(paths)
        report["corpus"] = engine.list_documents()
        report["configurations"] = {
            "flat": evaluate(engine, cases, hierarchy=False),
            "hierarchical": evaluate(engine, cases, hierarchy=True),
            "hierarchical_warm_exact_cache": evaluate(engine, cases, hierarchy=True, cache=True),
        }
        if args.rerank:
            reranked = Assistant(replace(settings, rerank=True))
            report["configurations"]["hierarchical_reranked"] = evaluate(reranked, cases, hierarchy=True)
        else:
            report["reranking"] = "Not run; pass --rerank after installing requirements-ml.txt"
        report["cache_smoke"] = cache_evaluation(settings)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    summary = {name: {k: v for k, v in result.items() if k != "outputs"} for name, result in report["configurations"].items()}
    print(json.dumps(summary, indent=2))
    print(f"Detailed report: {args.output}")

if __name__ == "__main__":
    main()
