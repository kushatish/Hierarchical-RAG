"""Opt-in real model tests; never substituted with mocked inference."""
from pathlib import Path
from tempfile import TemporaryDirectory
import os
import unittest
from core.config import Settings
from core.engine import Assistant

@unittest.skipUnless(os.getenv("RAG_RUN_MODEL_TESTS") == "1", "Set RAG_RUN_MODEL_TESTS=1 after downloading pinned CPU models")
class RealModelTests(unittest.TestCase):
    def test_embeddings_have_expected_dimension_and_semantic_order(self):
        import numpy as np
        from core.onnx_models import get_model
        vectors = get_model("embedding").encode(["The cat is sleeping on the sofa.", "A kitten rests on a couch.", "The database transaction was rolled back."])
        self.assertEqual(vectors.shape, (3, 384))
        np.testing.assert_allclose(np.linalg.norm(vectors, axis=1), 1, atol=1e-5)
        self.assertGreater(float(vectors[0] @ vectors[1]), float(vectors[0] @ vectors[2]))

    def test_reranker_prefers_relevant_passage(self):
        from core.onnx_models import get_model
        scores = get_model("reranker").predict([
            ("What is the capital of France?", "Paris is the capital of France."),
            ("What is the capital of France?", "Whales live in the ocean."),
        ])
        self.assertGreater(scores[0], scores[1])

    def test_real_neural_pipeline_and_persistent_vectors(self):
        with TemporaryDirectory() as directory:
            settings = Settings(db_path=Path(directory)/"model.db", backend="onnx", rerank=True)
            engine = Assistant(settings)
            engine.ingest("library.md", ["Library loans last fourteen days. Students can borrow four books."])
            engine.ingest("hostel.md", ["Hostel quiet hours start at ten in the evening."])
            first = engine.ask("How long do library loans last?", use_cache=False)
            self.assertEqual(first["citations"][0]["source"], "library.md")
            self.assertIn("fourteen", first["answer"])
            with engine.connect() as db:
                count = db.execute("SELECT count(*) FROM vectors").fetchone()[0]
            reopened = Assistant(settings)
            self.assertIn("fourteen", reopened.ask("How long do library loans last?", use_cache=False)["answer"])
            with reopened.connect() as db:
                self.assertEqual(db.execute("SELECT count(*) FROM vectors").fetchone()[0], count)

    def test_real_model_api_upload_query_cache_and_delete(self):
        import importlib.util
        if not all(importlib.util.find_spec(n) for n in ["fastapi", "httpx", "multipart"]):
            self.skipTest("Install requirements-dev.txt for real-model API test")
        from fastapi.testclient import TestClient
        from unittest.mock import patch
        from app.main import app, get_assistant
        with TemporaryDirectory() as directory, patch.dict(os.environ, {
            "RAG_DB_PATH": str(Path(directory)/"api-model.db"),
            "RAG_BACKEND": "onnx", "RAG_RERANK": "true", "RAG_GENERATOR": "extractive", "RAG_API_KEY": "model-test-key",
        }):
            get_assistant.cache_clear()
            try:
                with TestClient(app) as client:
                    headers = {"X-API-Key": "model-test-key"}
                    uploaded = client.post("/documents", headers=headers, files={"file": ("library.md", b"Library loans last fourteen days. Students can borrow four books.")})
                    self.assertEqual(uploaded.status_code, 200)
                    result = client.post("/query", headers=headers, json={"query": "How long do library loans last?"}).json()
                    self.assertIn("fourteen", result["answer"])
                    self.assertEqual(result["backend"], "onnx")
                    self.assertFalse(result["cache_hit"])
                    self.assertEqual(client.get(result["citations"][0]["source_url"], headers=headers).status_code, 200)
                    self.assertTrue(client.post("/query", headers=headers, json={"query": "How long do library loans last?"}).json()["cache_hit"])
                    client.delete("/documents/"+uploaded.json()["id"], headers=headers)
                    missing = client.post("/query", headers=headers, json={"query": "How long do library loans last?"}).json()
                    self.assertFalse(missing["cache_hit"])
                    self.assertIn("Insufficient evidence", missing["answer"])
            finally:
                get_assistant.cache_clear()
