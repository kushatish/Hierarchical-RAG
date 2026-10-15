"""Export CPython's independently authored docstrings as a versioned search corpus.

Use Python 3.12. Sources are Python standard-library docstrings under the PSF
license, not generated policy text. Questions remain a development set until
independently reviewed; this script never generates questions or expected answers.
"""
import importlib
import inspect
import json
from pathlib import Path
import platform
import sys
from hashlib import sha256

SYMBOLS = [
("json", "loads"), ("json", "dumps"),
("pathlib", "Path.glob"), ("pathlib", "Path.read_text"), ("pathlib", "Path.resolve"), ("pathlib", "Path.relative_to"),
("sqlite3", "Connection.backup"), ("sqlite3", "Connection.iterdump"),
("concurrent.futures", "wait"), ("concurrent.futures", "as_completed"),
("itertools", "groupby"), ("itertools", "islice"),
("collections", "Counter"), ("collections", "deque"),
("functools", "lru_cache"), ("functools", "reduce"),
("heapq", "nlargest"), ("heapq", "merge"),
("bisect", "bisect_left"), ("bisect", "insort_right"),
("shutil", "copy2"), ("shutil", "copytree"), ("shutil", "rmtree"),
("tempfile", "TemporaryDirectory"), ("tempfile", "NamedTemporaryFile"),
("statistics", "median"), ("statistics", "mean"), ("statistics", "quantiles"),
("hmac", "compare_digest"), ("urllib.parse", "urlparse"), ("urllib.parse", "urljoin"),
("re", "findall"), ("re", "finditer"), ("re", "sub"),
("textwrap", "dedent"), ("textwrap", "fill"),
("difflib", "get_close_matches"), ("time", "monotonic"), ("os", "replace"),
]

def main():
    if sys.version_info[:2] != (3,12):
        raise SystemExit("Rebuild this corpus with Python 3.12 to keep its version stable")
    here = Path(__file__).resolve().parent
    manifest = {"source": "CPython standard-library docstrings", "python": platform.python_version(),
                "license": "PSF License Agreement for Python", "license_url": "https://docs.python.org/3.12/license.html", "documents": []}
    for module_name, name in SYMBOLS:
        symbol = importlib.import_module(module_name)
        for part in name.split("."):
            symbol = getattr(symbol, part)
        doc = inspect.getdoc(symbol)
        if not doc:
            raise RuntimeError(f"No documentation for {module_name}.{name}")
        filename = f"{module_name}.{name}.txt"
        text = f"{module_name}.{name}\n\n{doc}\n"
        (here / "documents" / filename).write_text(text)
        manifest["documents"].append({"file": filename, "symbol": f"{module_name}.{name}", "sha256": sha256(text.encode()).hexdigest()})
    (here / "manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    print(f"Exported {len(SYMBOLS)} documented symbols")

if __name__ == "__main__":
    main()
