from dataclasses import dataclass
from pathlib import Path
import os

ROOT = Path(__file__).resolve().parents[1]

@dataclass(frozen=True)
class Settings:
    db_path: Path = ROOT / "data" / "assistant.sqlite3"
    backend: str = "lexical"
    generator: str = "extractive"
    rerank: bool = False
    parent_size: int = 240
    child_size: int = 60
    overlap: int = 12
    top_k: int = 5
    context_words: int = 1200
    cache_threshold: float = 0.97
    cache_ttl: int = 3600
    cache_capacity: int = 1000

    def __post_init__(self):
        if self.backend not in {"lexical", "sentence-transformers", "onnx"}:
            raise ValueError("Unsupported retrieval backend")
        if self.generator not in {"extractive", "gemini"}:
            raise ValueError("Unsupported generator")
        if not 0 <= self.overlap < self.child_size <= self.parent_size:
            raise ValueError("Require 0 <= overlap < child_size <= parent_size")
        if min(self.top_k, self.context_words, self.cache_ttl, self.cache_capacity) <= 0:
            raise ValueError("Limits must be positive")
        if not 0 <= self.cache_threshold <= 1:
            raise ValueError("Invalid cache threshold")

    @classmethod
    def from_env(cls):
        return cls(
            db_path=Path(os.getenv("RAG_DB_PATH", str(ROOT / "data" / "assistant.sqlite3"))),
            backend=os.getenv("RAG_BACKEND", "lexical"),
            generator=os.getenv("RAG_GENERATOR", "extractive"),
            rerank=os.getenv("RAG_RERANK", "false").lower() == "true",
            parent_size=int(os.getenv("RAG_PARENT_WORDS", "240")),
            child_size=int(os.getenv("RAG_CHILD_WORDS", "60")),
            overlap=int(os.getenv("RAG_OVERLAP_WORDS", "12")),
            top_k=int(os.getenv("RAG_TOP_K", "5")),
            context_words=int(os.getenv("RAG_CONTEXT_WORDS", "1200")),
            cache_threshold=float(os.getenv("RAG_CACHE_THRESHOLD", "0.97")),
            cache_ttl=int(os.getenv("RAG_CACHE_TTL", "3600")),
            cache_capacity=int(os.getenv("RAG_CACHE_CAPACITY", "1000")),
        )
