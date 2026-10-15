"""HTTP smoke workload runner. Start the API separately; defaults to one repeated query."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
import platform
from pathlib import Path
import statistics
import time
from urllib.request import Request, urlopen
from evaluation.benchmark import percentile

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--requests", type=int, default=100)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--dataset", type=Path, help="Cycle through a benchmark question file")
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--question", default="What is the minimum attendance requirement?")
    parser.add_argument("--output", type=Path, default=Path("evaluation/local_load_report.json"))
    args = parser.parse_args()
    if not 1 <= args.requests <= 10000 or not 1 <= args.concurrency <= 64:
        parser.error("Use 1–10000 requests and 1–64 workers")
    headers = {"Content-Type": "application/json"}
    if os.getenv("RAG_API_KEY"):
        headers["X-API-Key"] = os.environ["RAG_API_KEY"]
    cases = json.loads(args.dataset.read_text()) if args.dataset else [{"question": args.question, "answerable": True}]
    if not cases:
        parser.error("Workload must contain questions")
    def request_once(index):
        case = cases[index % len(cases)]
        data = json.dumps({"query": case["question"], "use_cache": not args.no_cache}).encode()
        start = time.perf_counter()
        try:
            request = Request(args.url.rstrip("/") + "/query", data=data, headers=headers)
            with urlopen(request, timeout=60) as response:
                result = json.load(response)
            return {"ok": True, "content_matches_expectation": bool(result.get("cited_ids")) if case["answerable"] else "insufficient evidence" in result.get("answer", "").lower(), "latency_ms": (time.perf_counter() - start) * 1000,
                    "cache_hit": result.get("cache_hit", False)}
        except Exception as exc:
            return {"ok": False, "error_type": type(exc).__name__, "latency_ms": (time.perf_counter() - start) * 1000, "cache_hit": False}
    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        results = list(pool.map(request_once, range(args.requests)))
    duration = time.perf_counter() - start
    latencies = [r["latency_ms"] for r in results]
    report = {"note": "Local HTTP smoke workload; not a production capacity claim. HTTP success and basic answerability expectations are reported separately; neither is factual accuracy.",
              "workload_question_count": len(cases), "cache_enabled": not args.no_cache,
              "content_expectation_matches": sum(r.get("content_matches_expectation", False) for r in results),
              "requests": args.requests, "concurrency": args.concurrency, "url": args.url,
              "platform": platform.platform(), "python": platform.python_version(),
              "successes": sum(r["ok"] for r in results), "failures": sum(not r["ok"] for r in results),
              "elapsed_seconds": duration, "throughput_requests_per_second": args.requests / duration,
              "latency_p50_ms": statistics.median(latencies), "latency_p95_ms": percentile(latencies, .95),
              "cache_hits": sum(r["cache_hit"] for r in results), "errors": [r for r in results if not r["ok"]]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "errors"}, indent=2))
    if report["failures"]:
        raise SystemExit(1)

if __name__ == "__main__":
    main()
