"""Deterministic core regression tests: no network, model downloads or API keys."""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
import json
import time
import unittest
from unittest.mock import patch

from core.config import Settings
from core.engine import Assistant
from core.retrieval import chunk_page, compatible_questions

class EngineTests(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.engine = Assistant(Settings(db_path=Path(self.tmp.name) / "test.sqlite3", parent_size=30, child_size=8, overlap=2))
        self.engine.ingest("policy.md", ["Scholarship policy. The scholarship requires a CGPA of 8.5. Students with disciplinary penalties are not eligible. Applications close in September.", "Library books may be borrowed for fourteen days."])

    def tearDown(self):
        self.tmp.cleanup()

    def test_exact_hit_and_actual_timing(self):
        first = self.engine.ask("What CGPA does the scholarship require?")
        second = self.engine.ask("What CGPA does the scholarship require?")
        self.assertFalse(first["cache_hit"])
        self.assertTrue(second["cache_hit"])
        self.assertEqual(second["cache_type"], "exact")
        self.assertEqual(first["answer"], second["answer"])
        self.assertGreater(first["latency_ms"], 0)
        self.assertEqual(second["usage"]["input_tokens"], 0)

    def test_exact_hit_avoids_loading_document_text(self):
        self.engine.ask("scholarship CGPA")
        with patch.object(self.engine, "snapshot", side_effect=AssertionError("Should not load corpus on cache hit")):
            self.assertTrue(self.engine.ask("scholarship CGPA")["cache_hit"])

    def test_persistence(self):
        self.engine.ask("scholarship CGPA")
        reopened = Assistant(self.engine.settings)
        self.assertTrue(reopened.ask("scholarship CGPA")["cache_hit"])

    def test_update_invalidates(self):
        self.engine.ask("scholarship CGPA")
        self.engine.ingest("policy.md", ["Scholarship requires a CGPA of 9.0."])
        result = self.engine.ask("scholarship CGPA")
        self.assertFalse(result["cache_hit"])
        self.assertIn("9.0", result["answer"])
        self.assertNotIn("8.5", result["answer"])

    def test_identical_upload_preserves_revision(self):
        before, docs = self.engine.snapshot()
        result = self.engine.ingest("policy.md", json.loads(docs[0]["pages"]))
        self.assertFalse(result["changed"])
        self.assertEqual(before, self.engine.snapshot()[0])

    def test_delete_invalidates(self):
        self.engine.ask("scholarship CGPA")
        self.assertTrue(self.engine.delete_document(self.engine.list_documents()[0]["id"]))
        result = self.engine.ask("scholarship CGPA")
        self.assertFalse(result["cache_hit"])
        self.assertEqual(result["citations"], [])
        self.assertFalse(self.engine.delete_document("missing"))

    def test_citations_are_original_page_spans(self):
        result = self.engine.ask("library books")
        _, docs = self.engine.snapshot()
        pages = json.loads(docs[0]["pages"])
        for citation in result["citations"]:
            self.assertIn(citation["text"], pages[citation["page"] - 1])
        self.assertEqual(result["citations"][0]["page"], 2)
        self.assertTrue(set(result["cited_ids"]).issubset({c["id"] for c in result["citations"]}))

    def test_common_requirement_wording(self):
        self.engine.ingest("attendance.md", ["Students must maintain at least 75 percent attendance to sit for examinations."])
        result = self.engine.ask("What is the minimum attendance requirement?")
        self.assertIn("75 percent", result["answer"])
        self.assertNotIn("scholarship", result["answer"].lower())

    def test_wrapped_prose_keeps_the_complete_answer_sentence(self):
        self.engine.ingest("forms.md", ["Applications require a signed\npermission form from the office."])
        result = self.engine.ask("What do applications require?")
        self.assertIn("signed permission form", " ".join(result["answer"].split()))

    def test_related_but_unanswerable_question(self):
        result = self.engine.ask("What is the library monthly membership price?")
        self.assertIn("Insufficient evidence", result["answer"])
        self.assertEqual(result["cited_ids"], [])

    def test_parent_expansion(self):
        _, docs = self.engine.snapshot()
        parents = self.engine.retrieve("CGPA", docs, True)
        children = self.engine.retrieve("CGPA", docs, False)
        self.assertGreater(len(parents[0]["text"].split()), len(children[0]["text"].split()))
        self.assertEqual(len({c["parent_id"] for c in parents}), len(parents))

    def test_context_budget(self):
        engine = Assistant(replace(self.engine.settings, context_words=9))
        result = engine.ask("scholarship CGPA library books", use_cache=False)
        self.assertLessEqual(sum(len(c["text"].split()) for c in result["citations"]), 9)

    def test_invalid_questions(self):
        for question in ["", "   ", "x" * 2001, None]:
            with self.assertRaises(ValueError):
                self.engine.ask(question)

    def test_empty_and_invalid_documents(self):
        for pages in [[], [" "], [123]]:
            with self.assertRaises(ValueError):
                self.engine.ingest("bad.md", pages)

    def test_unsupported_file(self):
        with self.assertRaises(ValueError):
            self.engine.ingest_file("bad.exe")

    def test_no_evidence_not_cached(self):
        for _ in range(2):
            result = self.engine.ask("interstellar zucchini")
            self.assertFalse(result["cache_hit"])
            self.assertIn("Insufficient evidence", result["answer"])

    def test_expiry(self):
        self.engine.ask("scholarship CGPA")
        with self.engine.connect() as db:
            db.execute("UPDATE cache SET created=0")
        self.assertFalse(self.engine.ask("scholarship CGPA")["cache_hit"])

    def test_capacity(self):
        engine = Assistant(replace(self.engine.settings, cache_capacity=2))
        for query in ["scholarship CGPA", "library books", "scholarship September"]:
            engine.ask(query)
        with engine.connect() as db:
            self.assertEqual(db.execute("SELECT count(*) FROM cache").fetchone()[0], 2)

    def test_configuration_separates_cache(self):
        self.engine.ask("scholarship CGPA")
        engine = Assistant(replace(self.engine.settings, top_k=2))
        self.assertFalse(engine.ask("scholarship CGPA")["cache_hit"])
        self.assertFalse(self.engine.ask("scholarship CGPA", hierarchy=False)["cache_hit"])

    def test_concurrent_answers_never_mismatch(self):
        queries = ["scholarship CGPA", "library books"] * 12
        with ThreadPoolExecutor(max_workers=6) as pool:
            results = list(pool.map(self.engine.ask, queries))
        for question, result in zip(queries, results):
            self.assertIn("8.5" if "CGPA" in question else "fourteen", result["answer"])

    def test_disabling_cache_still_respects_generation_capacity(self):
        from core.operations import SingleFlight
        from threading import Event
        entered, release = Event(), Event()
        self.engine.flights = SingleFlight(capacity=1)
        original = self.engine.generate
        def blocked(query, citations):
            entered.set()
            if not release.wait(timeout=5):
                raise TimeoutError("Test did not release generation")
            return original(query, citations)
        with patch.object(self.engine, "generate", side_effect=blocked):
            with ThreadPoolExecutor(max_workers=1) as pool:
                first = pool.submit(self.engine.ask, "scholarship CGPA", use_cache=False)
                self.assertTrue(entered.wait(timeout=5))
                try:
                    with self.assertRaises(RuntimeError):
                        self.engine.ask("library loans", use_cache=False)
                finally:
                    release.set()
                self.assertIn("8.5", first.result()["answer"])

    def test_simultaneous_identical_misses_generate_once(self):
        from threading import Barrier
        start = Barrier(6)
        original = self.engine.generate
        def slow_generate(query, citations):
            time.sleep(.1)
            return original(query, citations)
        def worker(_):
            start.wait(timeout=5)
            return self.engine.ask("scholarship CGPA")
        with patch.object(self.engine, "generate", side_effect=slow_generate) as generate:
            with ThreadPoolExecutor(max_workers=6) as pool:
                results = list(pool.map(worker, range(6)))
        self.assertEqual(generate.call_count, 1)
        self.assertTrue(any(r["cache_type"] == "coalesced" for r in results))
        self.assertEqual(len({r["answer"] for r in results}), 1)

    def test_stale_generation_cannot_repopulate_cache(self):
        original = self.engine.generate
        def update_during_generation(query, citations):
            self.engine.ingest("policy.md", ["Scholarship now requires 9.5 CGPA."])
            return original(query, citations)
        with patch.object(self.engine, "generate", side_effect=update_during_generation):
            old = self.engine.ask("scholarship CGPA")
        new = self.engine.ask("scholarship CGPA")
        self.assertGreater(new["revision"], old["revision"])
        self.assertFalse(new["cache_hit"])
        self.assertIn("9.5", new["answer"])

    def test_semantic_cache_opt_in(self):
        self.engine.ask("scholarship CGPA requirement")
        self.assertFalse(self.engine.ask("CGPA scholarship requirement")["cache_hit"])
        self.assertTrue(self.engine.ask("requirement scholarship CGPA", semantic_cache=True)["cache_hit"])

    def test_guard_against_negation_numbers_and_question_type(self):
        for left, right in [("Who is eligible?", "Who is not eligible?"), ("CGPA 8.5", "CGPA 9.5"), ("CGPA 8.5", "CGPA 5.8"), ("typed=True", "typed=False"), ("fee -10", "fee 10"), ("When scholarship?", "Who scholarship?")]:
            self.assertFalse(compatible_questions(left, right))

    def test_negation_does_not_hit_even_with_high_similarity(self):
        self.engine.ask("Who is eligible for scholarship?")
        with patch("core.engine.semantic_similarity", return_value=1.0):
            result = self.engine.ask("Who is not eligible for scholarship?", semantic_cache=True)
        self.assertFalse(result["cache_hit"])

    def test_split_preserves_whitespace_and_parent_links(self):
        text = "one  two\nthree four five six seven eight nine ten"
        chunks = list(chunk_page(text, 6, 3, 1))
        for _, parent, child in chunks:
            self.assertIn(parent, text)
            self.assertIn(child, parent)
        self.assertEqual({n for n, _, _ in chunks}, {0, 1})

    def test_invalid_config(self):
        for config in [{"child_size": 0}, {"overlap": 60}, {"top_k": 0}, {"backend": "unknown"}, {"cache_threshold": 2}]:
            with self.assertRaises(ValueError):
                Settings(**config)

if __name__ == "__main__":
    unittest.main()
