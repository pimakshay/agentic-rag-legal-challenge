"""Focused tests for the legal hybrid RAG building blocks."""

from __future__ import annotations

import unittest
from pathlib import Path

from retrieval import (
    IngestedCorpusLoader,
    LegalChunkerConfig,
    LegalHybridRAGPipeline,
    LegalIngestChunker,
    LegalQuestionRouter,
)


ROOT = Path(__file__).resolve().parents[1]
INGEST_ROOT = ROOT / "ingestion" / "docs_corpus_ingest_result"
DOCS_ROOT = ROOT / "docs_corpus"


class LegalHybridRAGTests(unittest.TestCase):
    def test_retrieval_package_exports(self):
        self.assertIsNotNone(IngestedCorpusLoader)
        self.assertIsNotNone(LegalHybridRAGPipeline)
        self.assertIsNotNone(LegalQuestionRouter)

    def test_loader_reads_corpus_and_flags_missing_structure(self):
        loader = IngestedCorpusLoader(ingest_root=INGEST_ROOT, docs_root=DOCS_ROOT)
        documents = loader.load_corpus()
        self.assertGreaterEqual(len(documents), 30)
        doc_ids = {doc.metadata["doc_id"] for doc in documents}
        self.assertIn("03b621728fe29eb6113fcdb57f6458d793fd2d5c5b833ae26d40f04a29c85359", doc_ids)

        missing_structure_docs = [
            doc for doc in documents if not doc.metadata["structure_available"]
        ]
        self.assertTrue(
            any(
                doc.metadata["doc_id"]
                == "4e387152960c1029b3711cacb05b287b13c977bc61f2558059a62b7b427a62eb"
                for doc in missing_structure_docs
            )
        )

    def test_legal_chunker_emits_title_and_page_chunks(self):
        loader = IngestedCorpusLoader(ingest_root=INGEST_ROOT, docs_root=DOCS_ROOT)
        source_doc = loader.load_corpus()[0]
        chunker = LegalIngestChunker(LegalChunkerConfig())
        chunk_result = chunker.chunk([source_doc])
        chunk_kinds = {chunk.metadata["chunk_kind"] for chunk in chunk_result.chunks}
        self.assertIn("title_page", chunk_kinds)
        self.assertIn("page_anchor", chunk_kinds)
        self.assertTrue(any(chunk.metadata["page_numbers"] for chunk in chunk_result.chunks))

    def test_router_detects_case_article_and_page_hints(self):
        router = LegalQuestionRouter()
        route = router.route(
            "Based on page 2 of case CFI 057/2025, under Article 28(1) of the Operating Law 2018 what happened?",
            "free_text",
        )
        self.assertIn("CFI 057/2025", route.case_ids)
        self.assertIn("Article 28(1)", route.article_refs)
        self.assertEqual(route.target_pages, [2])
        self.assertTrue(route.page_specific_mode)

    def test_pipeline_parses_types_without_model_calls(self):
        pipeline = LegalHybridRAGPipeline(llm=object(), embedding_model=object())
        self.assertTrue(pipeline._parse_answer_by_type("true", "boolean"))
        self.assertEqual(pipeline._parse_answer_by_type("2026-01-13", "date"), "2026-01-13")
        self.assertEqual(pipeline._parse_answer_by_type("Alice; Bob", "names"), ["Alice", "Bob"])
        self.assertIsNone(pipeline._parse_answer_by_type("null", "number"))


if __name__ == "__main__":
    unittest.main()
