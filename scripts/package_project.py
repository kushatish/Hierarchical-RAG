"""Build a clean source folder and ZIP without databases, model weights or secrets."""
from hashlib import sha256
from pathlib import Path
import shutil
import zipfile

ROOT = Path(__file__).resolve().parents[1]
EXCLUDED = {"deliverables", ".git", ".venv", "venv", "__pycache__", ".pytest_cache", "chroma_db", "node_modules", ".model-cache", ".codex", ".agents"}

def main():
    folder = ROOT/"deliverables/Hierarchical-RAG-updated"
    folder.mkdir(parents=True, exist_ok=True)
    included = []
    for source in sorted(ROOT.rglob("*")):
        relative = source.relative_to(ROOT)
        if any(part in EXCLUDED for part in relative.parts) or not source.is_file() or source.is_symlink():
            continue
        if relative.parts[:3] == ("rag_cag_project", "data", "models"):
            continue
        if (source.name.startswith(".env") and source.name != ".env.example") or source.suffix in {".pyc", ".pyo", ".zip", ".db", ".onnx", ".pem", ".key"} or ".sqlite" in source.name:
            continue
        target = folder/relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        included.append(relative)
    archive = folder.with_suffix(".zip")
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as output:
        for relative in included:
            output.write(folder/relative, Path(folder.name)/relative)
    with zipfile.ZipFile(archive) as output:
        assert output.testzip() is None
        for relative in included:
            assert sha256(output.read(str(Path(folder.name)/relative))).digest() == sha256((ROOT/relative).read_bytes()).digest()
    print(f"Verified {len(included)} source/documentation files")
    print(f"ZIP: {archive} ({archive.stat().st_size:,} bytes)")

if __name__ == "__main__":
    main()
