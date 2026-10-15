from dataclasses import replace
import importlib.util
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
from core.config import Settings
from core.engine import Assistant

class IndexTests(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.settings = Settings(db_path=Path(self.tmp.name)/"index.db", parent_size=30, child_size=8, overlap=2)
        self.engine = Assistant(self.settings)
        self.engine.ingest("policy.md", ["Scholarship requires CGPA 8.5. Library loans last fourteen days. Applications close in September."])
    def tearDown(self):
        self.tmp.cleanup()

    def test_persistent_fts_index_avoids_rechunking_after_restart(self):
        with patch("core.index.chunk_page", side_effect=AssertionError("Rebuilt existing chunks")):
            reopened = Assistant(self.settings)
            result = reopened.ask("library loans", use_cache=False)
        self.assertIn("fourteen", result["answer"])

    def test_replacement_and_delete_remove_old_search_terms(self):
        self.engine.ingest("policy.md", ["Scholarship requires CGPA 9.5."])
        self.assertEqual(self.engine.ask("library loans", use_cache=False)["citations"], [])
        self.engine.delete_document(self.engine.list_documents()[0]["id"])
        with self.engine.connect() as db:
            self.assertEqual(db.execute("SELECT count(*) FROM chunks").fetchone()[0], 0)
            self.assertEqual(db.execute("SELECT count(*) FROM chunk_search WHERE chunk_search MATCH 'library'").fetchone()[0], 0)

    def test_chunk_configuration_isolated(self):
        small = Assistant(replace(self.settings, child_size=4, overlap=1))
        self.assertIn("8.5", small.ask("scholarship CGPA", use_cache=False)["answer"])
        self.assertIn("8.5", self.engine.ask("scholarship CGPA", use_cache=False)["answer"])
        with self.engine.connect() as db:
            self.assertEqual(db.execute("SELECT count(DISTINCT config) FROM chunks").fetchone()[0], 2)

    def test_chunk_ids_not_reused(self):
        with self.engine.connect() as db:
            previous = db.execute("SELECT max(id) FROM chunks").fetchone()[0]
        self.engine.ingest("policy.md", ["Library loans now last seven days."])
        with self.engine.connect() as db:
            self.assertGreater(db.execute("SELECT min(id) FROM chunks").fetchone()[0], previous)

    def test_search_syntax_is_treated_as_question_text(self):
        result = self.engine.ask('library " OR * NOT (loans)', use_cache=False)
        self.assertTrue(result["citations"])

    @unittest.skipUnless(importlib.util.find_spec("numpy"), "NumPy needed for neural-vector contract")
    def test_neural_vectors_survive_restart_and_are_deleted_with_document(self):
        import numpy as np
        class FakeModel:
            batches = 0
            def encode(self, text, **kwargs):
                def vector(s):
                    value = np.array([1 + s.lower().count("scholarship"), 1 + s.lower().count("library")], dtype=np.float32)
                    return value / np.linalg.norm(value)
                if isinstance(text, str):
                    return vector(text)
                self.batches += 1
                return np.array([vector(s) for s in text])
        model = FakeModel()
        with patch("core.retrieval.embedding_model", return_value=model):
            neural = Assistant(replace(self.settings, backend="sentence-transformers"))
            neural.ask("scholarship CGPA", use_cache=False)
            first_batches = model.batches
            Assistant(neural.settings).ask("library loans", use_cache=False)
            self.assertEqual(model.batches, first_batches)
        with neural.connect() as db:
            self.assertGreater(db.execute("SELECT count(*) FROM vectors").fetchone()[0], 0)
        neural.delete_document(neural.list_documents()[0]["id"])
        with neural.connect() as db:
            self.assertEqual(db.execute("SELECT count(*) FROM vectors").fetchone()[0], 0)
