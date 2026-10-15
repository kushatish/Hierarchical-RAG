"""Adapter contract checks without network or paid provider requests."""
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from types import ModuleType, SimpleNamespace
import os
import unittest
from unittest.mock import patch
from core.config import Settings
from core.engine import Assistant

class ProviderContractTests(unittest.TestCase):
    def test_missing_gemini_key_fails_explicitly(self):
        # A fake SDK lets this test validate configuration without the optional SDK.
        google = ModuleType("google")
        google.genai = SimpleNamespace()
        with TemporaryDirectory() as directory, patch.dict("sys.modules", {"google": google}), patch.dict(os.environ, {"GEMINI_API_KEY": ""}):
            engine = Assistant(Settings(db_path=Path(directory)/"test.db", generator="gemini"))
            with self.assertRaisesRegex(ValueError, "GEMINI_API_KEY"):
                engine.generate("question", [{"source": "test", "page": 1, "text": "evidence"}])

    def test_invalid_provider_citation_is_rejected_and_usage_recorded(self):
        class FakeClient:
            def __init__(self, **kwargs):
                self.models = self
            def __enter__(self):
                return self
            def __exit__(self, *args):
                return False
            def generate_content(self, **kwargs):
                return SimpleNamespace(text="Unsupported claim [99]", usage_metadata=SimpleNamespace(prompt_token_count=40, candidates_token_count=8))
        google = ModuleType("google")
        google.genai = SimpleNamespace(Client=FakeClient)
        with TemporaryDirectory() as directory, patch.dict("sys.modules", {"google": google}), patch.dict(os.environ, {"GEMINI_API_KEY": "fake-test-key"}):
            engine = Assistant(Settings(db_path=Path(directory)/"test.db", generator="gemini"))
            answer, usage, ids = engine.generate("question", [{"source": "test", "page": 1, "text": "evidence"}])
            self.assertIn("Insufficient evidence", answer)
            self.assertEqual(ids, [])
            self.assertEqual(usage, {"input_tokens": 40, "output_tokens": 8})
