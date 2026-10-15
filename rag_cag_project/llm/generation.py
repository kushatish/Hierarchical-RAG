"""Generation is configured with RAG_GENERATOR and invoked by Assistant.generate.

Gemini loads lazily; offline extractive answers require no provider credentials.
"""
from core.config import Settings

__all__ = ["Settings"]
