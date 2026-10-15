"""The legacy Chroma snapshot is not used by the revision-aware engine.

Document and cache persistence lives in core.engine.Assistant; retrieval operates
on an immutable document snapshot so updates cannot mix corpus revisions.
"""
from core.engine import Assistant

__all__ = ["Assistant"]
