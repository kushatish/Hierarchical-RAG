"""Headless Streamlit interaction test; no browser or provider required."""
import importlib.util
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

@unittest.skipUnless(importlib.util.find_spec("streamlit"), "Install requirements-dev.txt to run UI tests")
class UITests(unittest.TestCase):
    def test_demo_answer_cache_and_stale_revision_warning(self):
        import streamlit as st
        from streamlit.testing.v1 import AppTest
        from core.engine import Assistant
        with TemporaryDirectory() as directory, patch.dict(os.environ, {
            "RAG_DB_PATH": str(Path(directory) / "ui.sqlite3"),
            "RAG_BACKEND": "lexical", "RAG_GENERATOR": "extractive", "RAG_RERANK": "false",
        }):
            st.cache_resource.clear()
            app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py"), default_timeout=15).run()
            self.assertEqual(len(app.exception), 0)
            next(b for b in app.button if b.label == "Load fictional demo handbook").click().run()
            app.text_input[0].set_value("What is the minimum attendance requirement?").run()
            next(b for b in app.button if b.label == "Ask").click().run()
            self.assertEqual(len(app.exception), 0)
            self.assertTrue(any("75 percent" in m.value for m in app.markdown))
            next(b for b in app.button if b.label == "Ask").click().run()
            self.assertTrue(any(m.value == "exact" for m in app.metric))
            Assistant().ingest("handbook.md", ["Attendance must be at least 80 percent."])
            app.run()
            self.assertTrue(any("library changed" in warning.value for warning in app.warning))
            st.cache_resource.clear()
