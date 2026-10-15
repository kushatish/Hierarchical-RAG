import importlib.util
from pathlib import Path
from tempfile import TemporaryDirectory
import os
import unittest
from unittest.mock import patch

HAS_API = all(importlib.util.find_spec(name) for name in ["fastapi", "httpx", "multipart"])

@unittest.skipUnless(HAS_API, "Install requirements-dev.txt to run API integration tests")
class APITests(unittest.TestCase):
    def setUp(self):
        from fastapi.testclient import TestClient
        from app.main import app, get_assistant
        self.tmp = TemporaryDirectory()
        self.env = patch.dict(os.environ, {"RAG_DB_PATH": str(Path(self.tmp.name) / "api.sqlite3"), "RAG_API_KEY": "test-key", "RAG_BACKEND": "lexical", "RAG_GENERATOR": "extractive", "RAG_RERANK": "false"})
        self.env.start()
        get_assistant.cache_clear()
        self.client = TestClient(app)
        self.headers = {"X-API-Key": "test-key"}

    def tearDown(self):
        from app.main import get_assistant
        self.client.close()
        get_assistant.cache_clear()
        self.env.stop()
        self.tmp.cleanup()

    def test_authentication_and_health(self):
        self.assertEqual(self.client.get("/health").status_code, 200)
        self.assertEqual(self.client.get("/documents").status_code, 401)

    def test_document_lifecycle_and_source(self):
        uploaded = self.client.post("/documents", headers=self.headers, files={"file": ("policy.md", b"Library loans last fourteen days.")})
        self.assertEqual(uploaded.status_code, 200)
        doc_id = uploaded.json()["id"]
        answer = self.client.post("/query", headers=self.headers, json={"query": "library loans"})
        self.assertIn("fourteen", answer.json()["answer"])
        self.assertTrue(self.client.post("/query", headers=self.headers, json={"query": "library loans"}).json()["cache_hit"])
        self.assertEqual(self.client.get(f"/documents/{doc_id}/pages/1", headers=self.headers).json()["text"], "Library loans last fourteen days.")
        self.assertEqual(self.client.delete(f"/documents/{doc_id}", headers=self.headers).status_code, 200)
        self.assertEqual(self.client.get(f"/documents/{doc_id}/pages/1", headers=self.headers).status_code, 404)

    def test_old_citation_link_rejects_new_revision(self):
        uploaded = self.client.post("/documents", headers=self.headers, files={"file": ("policy.md", b"Library loans last fourteen days.")})
        result = self.client.post("/query", headers=self.headers, json={"query": "library loans"}).json()
        source_url = result["citations"][0]["source_url"]
        self.assertEqual(self.client.get(source_url, headers=self.headers).status_code, 200)
        self.client.post("/documents", headers=self.headers, files={"file": ("policy.md", b"Library loans last seven days.")})
        self.assertEqual(self.client.get(source_url, headers=self.headers).status_code, 409)

    def test_readable_pdf_upload(self):
        import io
        from pypdf import PdfWriter
        from pypdf.generic import NameObject, DictionaryObject, DecodedStreamObject
        writer = PdfWriter()
        page = writer.add_blank_page(width=300, height=300)
        font = DictionaryObject({NameObject("/Type"): NameObject("/Font"), NameObject("/Subtype"): NameObject("/Type1"), NameObject("/BaseFont"): NameObject("/Helvetica")})
        page[NameObject("/Resources")] = DictionaryObject({NameObject("/Font"): DictionaryObject({NameObject("/F1"): font})})
        stream = DecodedStreamObject()
        stream.set_data(b"BT /F1 12 Tf 10 100 Td (Library loans last fourteen days.) Tj ET")
        page[NameObject("/Contents")] = writer._add_object(stream)
        data = io.BytesIO()
        writer.write(data)
        response = self.client.post("/documents", headers=self.headers, files={"file": ("policy.pdf", data.getvalue(), "application/pdf")})
        self.assertEqual(response.status_code, 200)
        result = self.client.post("/query", headers=self.headers, json={"query": "library loans"}).json()
        self.assertIn("fourteen", result["answer"])
        self.assertEqual(result["citations"][0]["page"], 1)

    def test_rate_limit_request_id_metrics_and_readiness(self):
        from core.operations import RateLimiter
        ready = self.client.get("/ready")
        self.assertEqual(ready.status_code, 200)
        self.assertEqual(len(ready.headers["X-Request-ID"]), 32)
        with patch("app.main.limiter", RateLimiter(limit=1)):
            self.assertEqual(self.client.get("/documents", headers=self.headers).status_code, 200)
            blocked = self.client.get("/documents", headers=self.headers)
            self.assertEqual(blocked.status_code, 429)
            self.assertIn("Retry-After", blocked.headers)
        report = self.client.get("/metrics", headers=self.headers).json()
        self.assertGreater(report["requests"], 0)
        self.assertNotIn("query", report)

    def test_deployment_refuses_missing_required_key(self):
        from fastapi.testclient import TestClient
        from app.main import app
        with patch.dict(os.environ, {"RAG_REQUIRE_API_KEY": "true", "RAG_API_KEY": ""}):
            with self.assertRaisesRegex(RuntimeError, "RAG_API_KEY"):
                with TestClient(app):
                    pass

    def test_generation_error_hides_internal_details(self):
        from app.main import get_assistant
        with patch.object(get_assistant(), "ask", side_effect=RuntimeError("private-provider-detail")):
            result = self.client.post("/query", headers=self.headers, json={"query": "valid question"})
        self.assertEqual(result.status_code, 503)
        self.assertNotIn("private-provider-detail", result.text)
        self.assertIn("X-Request-ID", result.headers)

    def test_validation_and_upload_limit(self):
        self.assertEqual(self.client.post("/query", headers=self.headers, json={"query": " "}).status_code, 400)
        self.assertEqual(self.client.post("/query", headers=self.headers, json={"query": "x" * 2001}).status_code, 422)
        self.assertEqual(self.client.post("/documents", headers=self.headers, files={"file": ("bad.exe", b"bad")}).status_code, 400)
        with patch("app.main.MAX_UPLOAD", 3):
            self.assertEqual(self.client.post("/documents", headers=self.headers, files={"file": ("big.md", b"four")}).status_code, 413)
