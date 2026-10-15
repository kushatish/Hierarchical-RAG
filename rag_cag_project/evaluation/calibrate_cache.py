"""Sweep semantic-cache thresholds on labelled development query pairs.

This evaluates the equivalence gate independently of whether a seed answer was
cacheable. Do not select thresholds using the held-out test set.
"""
import argparse
import json
from pathlib import Path
from core.retrieval import compatible_questions, semantic_similarity

HERE = Path(__file__).resolve().parent

def sweep(pairs, backend, thresholds):
    scored = [{**pair, "compatible": compatible_questions(pair["seed"], pair["candidate"]),
               "score": semantic_similarity(pair["seed"], pair["candidate"], backend)} for pair in pairs]
    results = []
    for threshold in thresholds:
        tp = fp = fn = tn = 0
        for pair in scored:
            hit = pair["compatible"] and pair["score"] >= threshold
            if hit and pair["equivalent"]:
                tp += 1
            elif hit:
                fp += 1
            elif pair["equivalent"]:
                fn += 1
            else:
                tn += 1
        results.append({"threshold": threshold, "true_positives": tp, "false_positives": fp,
                        "false_negatives": fn, "true_negatives": tn,
                        "precision": tp / (tp + fp) if tp + fp else None,
                        "recall": tp / (tp + fn) if tp + fn else None,
                        "false_hit_rate": fp / (fp + tn) if fp + tn else None})
    return scored, results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", type=Path, default=HERE / "cache_pairs.json")
    parser.add_argument("--backend", choices=["lexical", "sentence-transformers", "onnx"], default="lexical")
    parser.add_argument("--output", type=Path, default=HERE / "cache_calibration_report.json")
    args = parser.parse_args()
    pairs = json.loads(args.pairs.read_text())
    if not pairs or not any(p["equivalent"] for p in pairs) or not any(not p["equivalent"] for p in pairs):
        parser.error("Include both equivalent and non-equivalent labelled pairs")
    scored, results = sweep(pairs, args.backend, [.75, .8, .85, .9, .95, .97, .99, 1.0])
    report = {"note": "Development-set threshold sweep only; not a recommendation to deploy any threshold. Corpus/generator cache scoping remains mandatory.",
              "backend": args.backend, "pairs": scored, "thresholds": results}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(results, indent=2))

if __name__ == "__main__":
    main()
