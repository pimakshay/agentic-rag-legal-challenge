"""Focused tests for the legal hybrid RAG building blocks."""

from __future__ import annotations

import unittest
from pathlib import Path
import json

from langchain_core.documents import Document

from ingestion.legal_metadata import build_case_metadata, build_law_metadata
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
    @classmethod
    def setUpClass(cls):
        cls.loader = IngestedCorpusLoader(ingest_root=INGEST_ROOT, docs_root=DOCS_ROOT)
        cls.documents = cls.loader.load_corpus()

    def test_retrieval_package_exports(self):
        self.assertIsNotNone(IngestedCorpusLoader)
        self.assertIsNotNone(LegalHybridRAGPipeline)
        self.assertIsNotNone(LegalQuestionRouter)

    def test_loader_reads_corpus_and_flags_missing_structure(self):
        documents = self.documents
        self.assertGreaterEqual(len(documents), 30)
        doc_ids = {doc.metadata["doc_id"] for doc in documents}
        self.assertIn("03b621728fe29eb6113fcdb57f6458d793fd2d5c5b833ae26d40f04a29c85359", doc_ids)

        self.assertTrue(
            all(isinstance(doc.metadata["structure_available"], bool) for doc in documents)
        )

    def test_legal_chunker_emits_title_and_page_chunks(self):
        source_doc = self.documents[0]
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

    def test_router_classifies_issue_date_questions(self):
        router = LegalQuestionRouter()
        direct_route = router.route(
            "What is the Date of Issue of the document in case CFI 057/2025?",
            "date",
        )
        self.assertEqual(direct_route.comparison_kind, "issue_date")
        self.assertTrue(direct_route.prefer_title_page)

        compare_route = router.route(
            "Which case has an earlier Date of Issue: CFI 010/2024 or CFI 016/2025?",
            "name",
        )
        self.assertEqual(compare_route.comparison_kind, "issue_date")
        self.assertTrue(compare_route.comparison_mode)
        self.assertTrue(compare_route.prefer_title_page)

    def test_router_classifies_judge_overlap_and_keeps_generic_dates_narrow(self):
        router = LegalQuestionRouter()
        judge_route = router.route(
            "Considering all documents across case CA 005/2025 and case TCD 001/2024, was there any judge who participated in both cases?",
            "boolean",
        )
        self.assertEqual(judge_route.comparison_kind, "judge_overlap")
        self.assertTrue(judge_route.comparison_mode)

        generic_date_route = router.route(
            "Which case has the earlier date: CFI 010/2024 or CFI 016/2025?",
            "name",
        )
        self.assertIsNone(generic_date_route.comparison_kind)

    def test_pipeline_parses_types_without_model_calls(self):
        pipeline = LegalHybridRAGPipeline(llm=object(), embedding_model=object())
        self.assertTrue(pipeline._parse_answer_by_type("true", "boolean"))
        self.assertEqual(pipeline._parse_answer_by_type("2026-01-13", "date"), "2026-01-13")
        self.assertEqual(pipeline._parse_answer_by_type("Alice; Bob", "names"), ["Alice", "Bob"])
        self.assertIsNone(pipeline._parse_answer_by_type("null", "number"))

    def test_issue_date_normalization_helper(self):
        pipeline = LegalHybridRAGPipeline(llm=object(), embedding_model=object())
        self.assertEqual(pipeline._parse_non_iso_date("FEBRUARY 03, 2026"), "2026-02-03")
        self.assertEqual(pipeline._parse_non_iso_date("10 December 2025"), "2025-12-10")
        self.assertEqual(pipeline._parse_non_iso_date("JULY 01, 2025"), "2025-07-01")

    def test_case_metadata_extracts_core_fields_without_llm_or_existing_metadata(self):
        doc_id = "1b446e196b4d1752241c8ff689a31ea705e98ad0c16b9d343c303664f72b64a1"
        txt_dir = INGEST_ROOT / doc_id / "txt"
        content_items = json.loads((txt_dir / f"{doc_id}_content_list.json").read_text(encoding="utf-8"))
        structured_items = json.loads((txt_dir / f"{doc_id}_structure.json").read_text(encoding="utf-8"))

        metadata = build_case_metadata(content_items, structured_items, existing_metadata={}, llm_metadata={})
        self.assertEqual(metadata["claim_number"], "CFI 057/2025")
        self.assertEqual(
            metadata["case_name"],
            "Clyde & Co LLP v (1) Union Properties P.J.S.C. (2) UPP Capital Investment LLC",
        )
        self.assertEqual(metadata["court"], "DUBAI INTERNATIONAL FINANCIAL CENTRE COURTS")
        self.assertEqual(metadata["court_division"], "COURT OF FIRST INSTANCE")
        self.assertEqual(metadata["judgment_date"], "FEBRUARY 02, 2026")
        self.assertEqual(metadata["claimant"], "CLYDE & CO LLP")
        self.assertEqual(
            metadata["defendants"],
            ["(1) UNION PROPERTIES P.J.S.C.", "(2) UPP CAPITAL INVESTMENT LLC"],
        )

    def test_law_metadata_normalizes_law_number_after_llm_merge(self):
        doc_id = "fbdd7f9dd299d83b1f398778da2e6765dfaaed62005667264734a1f76ec09071"
        txt_dir = INGEST_ROOT / doc_id / "txt"
        content_items = json.loads((txt_dir / f"{doc_id}_content_list.json").read_text(encoding="utf-8"))
        structured_items = json.loads((txt_dir / f"{doc_id}_structure.json").read_text(encoding="utf-8"))

        metadata = build_law_metadata(
            content_items,
            structured_items,
            existing_metadata={},
            llm_metadata={"law_number": "DIFC Law No. 2 of 2018", "law_year": "2018"},
        )
        self.assertEqual(metadata["law_number"], "2")
        self.assertEqual(metadata["law_year"], "2018")

    def test_direct_issue_date_bypass_uses_metadata(self):
        pipeline = LegalHybridRAGPipeline(
            llm=object(),
            embedding_model=object(),
            ingest_root=str(INGEST_ROOT),
            docs_root=str(DOCS_ROOT),
        )
        pipeline.source_documents = self.documents
        pipeline._build_metadata_indexes(self.documents)

        result = pipeline.answer_question(
            {
                "question": "What is the Date of Issue of the document in case CFI 057/2025?",
                "answer_type": "date",
            }
        )
        self.assertEqual(result.answer, "2026-02-02")
        self.assertEqual(result.debug_metadata["reason"], "metadata_bypass")

    def test_issue_date_comparison_regressions_use_metadata_bypass(self):
        pipeline = LegalHybridRAGPipeline(
            llm=object(),
            embedding_model=object(),
            ingest_root=str(INGEST_ROOT),
            docs_root=str(DOCS_ROOT),
        )
        pipeline.source_documents = self.documents
        pipeline._build_metadata_indexes(self.documents)

        cases = [
            (
                "Between SCT 295/2025 and SCT 514/2025, which document has the earlier issue date?",
                "SCT 295/2025",
            ),
            (
                "Between CFI 010/2024 and SCT 169/2025, which document has the earlier issue date?",
                "SCT 169/2025",
            ),
            (
                "Which case has an earlier Date of Issue: CFI 010/2024 or CFI 016/2025?",
                "CFI 010/2024",
            ),
            (
                "Which case has an earlier Date of Issue: ENF 269/2023 or SCT 514/2025?",
                "ENF 269/2023",
            ),
            (
                "Which case has an earlier Date of Issue: SCT 169/2025 or SCT 295/2025?",
                "SCT 295/2025",
            ),
            (
                "Which case has an earlier Date of Issue: CFI 016/2025 or ENF 269/2023?",
                "ENF 269/2023",
            ),
            (
                "Which case has an earlier Date of Issue: CA 004/2025 or SCT 295/2025?",
                "CA 004/2025",
            ),
            (
                "Between CFI 016/2025 and CFI 057/2025, which document has the earlier issue date?",
                "CFI 057/2025",
            ),
            (
                "Which case was issued earlier by the Court of First Instance: CFI 057/2025 or CFI 067/2025?",
                "CFI 057/2025",
            ),
            (
                "Which case has an earlier Date of Issue: ENF 269/2023 or SCT 169/2025?",
                "ENF 269/2023",
            ),
        ]

        for question, expected in cases:
            with self.subTest(question=question):
                result = pipeline.answer_question(
                    {"question": question, "answer_type": "name"}
                )
                self.assertEqual(result.answer, expected)
                self.assertEqual(result.debug_metadata["reason"], "metadata_bypass")

    def test_issue_date_bypass_declines_ambiguous_case_metadata(self):
        pipeline = LegalHybridRAGPipeline(llm=object(), embedding_model=object())
        synthetic_docs = [
            Document(page_content="doc a", metadata={"doc_id": "doc-a", "claim_number": "CFI 999/2025", "judgment_date": "JANUARY 01, 2026"}),
            Document(page_content="doc b", metadata={"doc_id": "doc-b", "claim_number": "CFI 999/2025", "judgment_date": "FEBRUARY 01, 2026"}),
        ]
        pipeline.source_documents = synthetic_docs
        pipeline._build_metadata_indexes(synthetic_docs)

        route = pipeline.router.route(
            "What is the Date of Issue of the document in case CFI 999/2025?",
            "date",
        )
        resolution = pipeline._resolve_route_context(route)
        self.assertIsNone(
            pipeline._try_bypass_llm(
                "What is the Date of Issue of the document in case CFI 999/2025?",
                "date",
                route,
                resolution,
            )
        )

    def test_case_fact_index_resolves_identical_issue_dates(self):
        pipeline = LegalHybridRAGPipeline(llm=object(), embedding_model=object())
        synthetic_docs = [
            Document(page_content="doc a", metadata={"doc_id": "doc-a", "claim_number": "CFI 777/2025", "judgment_release_date": "JANUARY 05, 2026"}),
            Document(page_content="doc b", metadata={"doc_id": "doc-b", "claim_number": "CFI 777/2025", "judgment_release_date": "JANUARY 05, 2026"}),
        ]
        pipeline.source_documents = synthetic_docs
        pipeline._build_metadata_indexes(synthetic_docs)
        resolved = pipeline._resolve_unique_case_fact("CFI 777/2025", "issue_dates")
        self.assertIsNotNone(resolved)
        self.assertEqual(resolved[0], "2026-01-05")

    def test_judge_extraction_uses_header_blocks_only(self):
        pipeline = LegalHybridRAGPipeline(llm=object(), embedding_model=object())
        metadata = {
            "blocks": [
                {"block_index": 0, "page_number": 1, "text": "ORDER WITH REASONS OF H.E. JUSTICE ROGER STEWART KC"},
                {"block_index": 1, "page_number": 1, "text": "UPON the Order of H.E. Justice Michael Black KC dated 1 January 2026"},
                {"block_index": 2, "page_number": 3, "text": "The claimant relied on Justice Wayne Martin in another case"},
            ]
        }
        evidences = pipeline._extract_judge_evidences(metadata, "doc-1")
        self.assertEqual([e.value for e in evidences], ["ROGER STEWART KC"])

    def test_party_overlap_bypass_aggregates_multiple_docs(self):
        pipeline = LegalHybridRAGPipeline(llm=object(), embedding_model=object())
        synthetic_docs = [
            Document(page_content="doc a", metadata={"doc_id": "doc-a", "claim_number": "CFI 111/2025", "claimant": "ALPHA LLC", "defendants": ["BETA LLC"]}),
            Document(page_content="doc b", metadata={"doc_id": "doc-b", "claim_number": "CFI 111/2025", "claimant": "ALPHA LLC", "defendants": ["GAMMA LLC"]}),
            Document(page_content="doc c", metadata={"doc_id": "doc-c", "claim_number": "SCT 222/2025", "claimant": "DELTA LLC", "defendants": ["GAMMA LLC"]}),
        ]
        pipeline.source_documents = synthetic_docs
        pipeline._build_metadata_indexes(synthetic_docs)
        route = pipeline.router.route(
            "Do cases CFI 111/2025 and SCT 222/2025 involve any of the same legal entities or individuals as main parties?",
            "boolean",
        )
        resolution = pipeline._resolve_route_context(route)
        result = pipeline._try_bypass_llm(
            "Do cases CFI 111/2025 and SCT 222/2025 involve any of the same legal entities or individuals as main parties?",
            "boolean",
            route,
            resolution,
        )
        self.assertIsNotNone(result)
        self.assertTrue(result[0])

    def test_judge_overlap_regressions_use_case_fact_bypass(self):
        pipeline = LegalHybridRAGPipeline(
            llm=object(),
            embedding_model=object(),
            ingest_root=str(INGEST_ROOT),
            docs_root=str(DOCS_ROOT),
        )
        pipeline.source_documents = self.documents
        pipeline._build_metadata_indexes(self.documents)

        cases = [
            (
                "Based on the full case files for DEC 001/2025 and CFI 057/2025, was there any judge involved in both cases?",
                False,
            ),
            (
                "Is there a judge who presided over both case CFI 057/2025 and case ENF 269/2023?",
                False,
            ),
            (
                "Considering all documents across case CA 005/2025 and case TCD 001/2024, was there any judge who participated in both cases?",
                True,
            ),
            (
                "Review the full case files for DEC 001/2025 and TCD 001/2024 — did any judge preside over proceedings in both cases?",
                False,
            ),
            (
                "Is there a judge who presided over both case ARB 034/2025 and case CFI 067/2025?",
                True,
            ),
        ]

        for question, expected in cases:
            with self.subTest(question=question):
                result = pipeline.answer_question({"question": question, "answer_type": "boolean"})
                self.assertEqual(result.answer, expected)
                self.assertEqual(result.debug_metadata["reason"], "metadata_bypass")
                self.assertTrue(result.retrieval_refs)

    def test_judge_overlap_bypass_declines_when_case_lacks_high_confidence_judge(self):
        pipeline = LegalHybridRAGPipeline(
            llm=object(),
            embedding_model=object(),
            ingest_root=str(INGEST_ROOT),
            docs_root=str(DOCS_ROOT),
        )
        pipeline.source_documents = self.documents
        pipeline._build_metadata_indexes(self.documents)

        route = pipeline.router.route(
            "Was the same judge involved in both case CFI 010/2024 and case CFI 016/2025?",
            "boolean",
        )
        resolution = pipeline._resolve_route_context(route)
        self.assertIsNone(
            pipeline._try_bypass_llm(
                "Was the same judge involved in both case CFI 010/2024 and case CFI 016/2025?",
                "boolean",
                route,
                resolution,
            )
        )


if __name__ == "__main__":
    unittest.main()
