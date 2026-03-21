"""Legal-challenge hybrid RAG pipeline built on ingest outputs."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
import logging
import math
import re
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from langchain_core.documents import Document

from arlc import RetrievalRef, normalize_retrieved_pages
from retrieval.chunkers import LegalChunkerConfig, LegalIngestChunker
from retrieval.chunkers.legal_chunk_types import (
    decode_page_numbers,
    sanitize_metadata_for_vectorstore,
)
from retrieval.free_text_prompts import build_free_text_prompt, detect_free_text_subtype
from retrieval.legal_question_router import LegalQuestionRouter, RoutePlan
from retrieval.loaders import IngestedCorpusLoader
from retrieval.retrievers.base import BaseRAGRetriever
from retrieval.turbopuffer_store import TurbopufferStore
from retrieval.utils.rerankers import BaseReranker, MiniLMReranker

logger = logging.getLogger(__name__)


@dataclass
class NormalizedQuery:
    """Compatibility container for dense and sparse query variants."""
    original_query: str
    dense_query: str
    sparse_query: str
    used_llm: bool = False


@dataclass
class AnswerResult:
    """Answer payload returned by the legal pipeline."""
    answer: Any
    supporting_docs: List[Document]
    retrieval_refs: List[RetrievalRef]
    debug_metadata: Dict[str, Any] = field(default_factory=dict)
    raw_response: str = ""


@dataclass
class ResolutionContext:
    """Resolved metadata constraints for one query."""
    candidate_doc_ids: Set[str]
    confident_match: bool = False
    article_entries: Dict[str, List[Dict[str, Any]]] = field(default_factory=dict)


@dataclass
class FactEvidence:
    """Resolved fact with provenance."""

    value: str
    doc_id: str
    page_numbers: List[int]
    source_kind: str
    confidence: str = "high"


@dataclass
class CaseFactRecord:
    """Aggregated facts for one claim number across all matched documents."""

    claim_number: str
    doc_ids: Set[str] = field(default_factory=set)
    issue_dates: List[FactEvidence] = field(default_factory=list)
    judges: List[FactEvidence] = field(default_factory=list)
    claimants: List[FactEvidence] = field(default_factory=list)
    defendants: List[FactEvidence] = field(default_factory=list)


class LegalHybridRAGPipeline:
    """Challenge-specific hybrid RAG pipeline backed by ingestion output."""

    def __init__(
        self,
        llm: Any = None,
        embedding_model: Any = None,
        ingest_root: str = "ingestion/docs_corpus_ingest_result",
        docs_root: str = "docs_corpus",
        top_k_docs: int = 10,
        dense_candidate_k: int = 14,
        sparse_candidate_k: int = 24,
        dense_weight: float = 0.5,
        sparse_weight: float = 0.5,
        rrf_k: int = 60,
        enable_reranking: bool = True,
        reranker: Optional[BaseReranker] = None,
        loader: Optional[IngestedCorpusLoader] = None,
        chunker: Optional[LegalIngestChunker] = None,
        router: Optional[LegalQuestionRouter] = None,
        store: Optional[TurbopufferStore] = None,
        use_turbopuffer: bool = False,
        skip_indexing: bool = False,
    ) -> None:
        self.llm = llm
        self.embedding_model = embedding_model
        self.ingest_root = ingest_root
        self.docs_root = docs_root
        self.top_k_docs = top_k_docs
        self.dense_candidate_k = dense_candidate_k
        self.sparse_candidate_k = sparse_candidate_k
        self.dense_weight = dense_weight
        self.sparse_weight = sparse_weight
        self.rrf_k = rrf_k

        self.skip_indexing = skip_indexing
        self.use_turbopuffer = use_turbopuffer
        self.store = store
        if self.store is None and self.use_turbopuffer:
            self.store = TurbopufferStore(namespace="legal-challenge", region="gcp-europe-west3")
        self.loader = loader or IngestedCorpusLoader(ingest_root=ingest_root, docs_root=docs_root)
        self.chunker = chunker or LegalIngestChunker(LegalChunkerConfig())
        self.router = router or LegalQuestionRouter()
        self.reranker = reranker or (MiniLMReranker() if enable_reranking else None)

        self.source_documents: List[Document] = []
        self.raw_chunks: List[Document] = []
        self._all_doc_ids: set[str] = set()
        self._claim_number_index: Dict[str, set[str]] = {}
        self._neutral_citation_index: Dict[str, set[str]] = {}
        self._law_number_index: Dict[str, set[str]] = {}
        self._law_alias_index: Dict[str, set[str]] = {}
        self._article_index: Dict[str, List[Dict[str, Any]]] = {}
        self._doc_metadata: Dict[str, Dict[str, Any]] = {}
        self._case_fact_index: Dict[str, CaseFactRecord] = {}

        # Local fallback indices (used when Turbopuffer is disabled).
        # Dense: normalized embedding vectors for cosine similarity.
        self._local_dense_vectors_normed: Optional[List[List[float]]] = None
        self._local_dense_doc_ids: List[str] = []
        # Sparse: in-process BM25 retriever over chunk texts.
        self._local_bm25_retriever: Optional[Any] = None

    def load_corpus(self) -> List[Document]:
        self.source_documents = self.loader.load_corpus(self.ingest_root, self.docs_root)
        self._build_metadata_indexes(self.source_documents)
        return self.source_documents

    def build_indexes(self) -> Any:
        if self.embedding_model is None:
            raise ValueError("embedding_model must be provided")

        if not self.source_documents:
            self.load_corpus()

        chunk_result = self.chunker.chunk(self.source_documents)
        self.raw_chunks = [chunk for chunk in chunk_result.chunks if chunk.page_content.strip()]

        if self.store is not None:
            if self.skip_indexing:
                logger.info("Skipping Turbopuffer index (skip_indexing=True); using existing namespace.")
                return self.store

            def _embed(texts: List[str]) -> List[List[float]]:
                return self.embedding_model.embed_documents(texts)

            logger.info("Indexing %s legal chunks into Turbopuffer", len(self.raw_chunks))
            self.store.index_chunks(self.raw_chunks, _embed)
            return self.store

        # Local fallback: build dense vectors + BM25 without relying on Turbopuffer.
        logger.info("Building local fallback indexes for %s legal chunks", len(self.raw_chunks))
        self._local_dense_doc_ids = [str((c.metadata or {}).get("doc_id") or "") for c in self.raw_chunks]

        # Dense index (cosine similarity on normalized vectors).
        dense_vectors = self.embedding_model.embed_documents([c.page_content for c in self.raw_chunks])
        self._local_dense_vectors_normed = []
        for vec in dense_vectors:
            norm = math.sqrt(sum(float(x) * float(x) for x in vec)) or 1.0
            self._local_dense_vectors_normed.append([float(x) / norm for x in vec])

        # Sparse index (BM25).
        from langchain_community.retrievers import BM25Retriever

        self._local_bm25_retriever = BM25Retriever.from_documents(documents=self.raw_chunks)
        self._local_bm25_retriever.k = self.sparse_candidate_k  # default; retrieval will override per-query
        return None

    def retrieve(
        self,
        question_text: str,
        answer_type: str = "free_text",
        route: Optional[RoutePlan] = None,
        timing_out: Optional[Dict[str, float]] = None,
    ) -> Tuple[List[Document], RoutePlan]:
        import time as _time
        if not self.raw_chunks:
            self.build_indexes()

        active_route = route or self.router.route(question_text, answer_type)
        resolution = self._resolve_route_context(active_route)
        sparse_pool = self._filter_chunks(self.raw_chunks, resolution.candidate_doc_ids)

        def run_dense() -> tuple[List[Document], float]:
            t0 = _time.perf_counter()
            docs = self._dense_search(question_text, resolution, active_route)
            return docs, (_time.perf_counter() - t0) * 1000

        def run_sparse() -> tuple[List[Document], float]:
            t0 = _time.perf_counter()
            docs = self._sparse_search(question_text, sparse_pool, active_route, resolution)
            return docs, (_time.perf_counter() - t0) * 1000

        with ThreadPoolExecutor(max_workers=2) as executor:
            fut_dense = executor.submit(run_dense)
            fut_sparse = executor.submit(run_sparse)
            dense_docs, dense_ms = fut_dense.result()
            sparse_docs, sparse_ms = fut_sparse.result()

        if timing_out is not None:
            timing_out["retrieve_dense_ms"] = dense_ms
            timing_out["retrieve_sparse_ms"] = sparse_ms

        t0 = _time.perf_counter()
        fused_docs = self._fuse_results(dense_docs, sparse_docs, self.top_k_docs * 4)
        reranked_docs = self._rerank(question_text, fused_docs)
        if timing_out is not None:
            timing_out["retrieve_fuse_rerank_ms"] = (_time.perf_counter() - t0) * 1000

        return reranked_docs[: self.top_k_docs], active_route

    def answer_question(
        self,
        question_item: Dict[str, Any],
        telemetry_timer: Optional[Any] = None,
        timing_breakdown: Optional[Dict[str, float]] = None,
    ) -> AnswerResult:
        import time as _time
        question_text = question_item["question"]
        answer_type = question_item.get("answer_type", "free_text")

        t_start = _time.perf_counter()
        route = self.router.route(question_text, answer_type)
        if timing_breakdown is not None:
            timing_breakdown["route_ms"] = (_time.perf_counter() - t_start) * 1000

        t0 = _time.perf_counter()
        resolution = self._resolve_route_context(route)
        bypass_result = self._try_bypass_llm(question_text, answer_type, route, resolution)
        if timing_breakdown is not None:
            timing_breakdown["resolve_and_bypass_check_ms"] = (_time.perf_counter() - t0) * 1000

        if bypass_result is not None:
            if telemetry_timer:
                telemetry_timer.mark_token()
            ans, bypass_docs = bypass_result
            retrieval_refs = normalize_retrieved_pages(
                RetrievalRef(doc_id=str(doc.metadata["doc_id"]), page_numbers=self._chunk_page_numbers(doc))
                for doc in bypass_docs
            )
            return AnswerResult(
                answer=ans,
                supporting_docs=bypass_docs,
                retrieval_refs=retrieval_refs,
                debug_metadata={"route": asdict(route), "reason": "metadata_bypass"},
                raw_response=str(ans),
            )

        if route.comparison_mode and len(route.case_ids) > 1:
            t0 = _time.perf_counter()
            supporting_docs = self._retrieve_for_comparison(question_text, answer_type, route)
            if timing_breakdown is not None:
                timing_breakdown["retrieve_total_ms"] = (_time.perf_counter() - t0) * 1000
        else:
            t0 = _time.perf_counter()
            supporting_docs, route = self.retrieve(
                question_text, answer_type=answer_type, route=route, timing_out=timing_breakdown
            )
            if timing_breakdown is not None:
                timing_breakdown["retrieve_total_ms"] = (_time.perf_counter() - t0) * 1000

        if not supporting_docs:
            if telemetry_timer:
                telemetry_timer.mark_token()
            answer = self._default_absent_answer(answer_type)
            return AnswerResult(
                answer=answer,
                supporting_docs=[],
                retrieval_refs=[],
                debug_metadata={"route": asdict(route), "reason": "no_supporting_docs"},
                raw_response="",
            )

        if self.llm is None:
            raise ValueError("llm must be provided to answer questions")

        t0 = _time.perf_counter()
        prompt = self._build_prompt(question_text, answer_type, supporting_docs, route)
        if timing_breakdown is not None:
            timing_breakdown["build_prompt_ms"] = (_time.perf_counter() - t0) * 1000

        raw_response = self._invoke_llm(
            prompt,
            telemetry_timer=telemetry_timer,
            timing_out=timing_breakdown,
        )
        if timing_breakdown is not None and "llm_total_ms" in timing_breakdown:
            timing_breakdown["llm_ms"] = timing_breakdown["llm_total_ms"]
        
        used_indices = []
        parsed_answer = raw_response

        if answer_type.lower() != "free_text":
            parsed_answer = self._extract_deterministic_candidate(raw_response)
        elif answer_type.lower() == "free_text":
            ans_match = re.search(r"Answer:\s*(.*?)(?:\nIndices:|$)", raw_response, re.IGNORECASE | re.DOTALL)
            idx_match = re.search(r"Indices:\s*(.*)", raw_response, re.IGNORECASE | re.DOTALL)
            if ans_match:
                parsed_answer = ans_match.group(1).strip()
            if idx_match:
                idx_str = idx_match.group(1).strip()
                for token in re.split(r"[^\d]+", idx_str):
                    if token.isdigit():
                        used_indices.append(int(token))
                        
        answer = self._parse_answer_by_type(parsed_answer, answer_type)

        is_absent = answer is None or (
            answer_type == "free_text" and answer == self._default_absent_answer("free_text")
        )

        if is_absent:
            if answer is None and answer_type == "free_text":
                answer = self._default_absent_answer("free_text")
            retrieval_refs = []
        else:
            filtered_docs = list(supporting_docs)
            if answer_type.lower() == "free_text" and used_indices:
                filtered_docs = []
                for idx in used_indices:
                    try:
                        doc_idx = int(idx) - 1
                        if 0 <= doc_idx < len(supporting_docs):
                            filtered_docs.append(supporting_docs[doc_idx])
                    except (ValueError, TypeError):
                        pass
                if not filtered_docs:
                    filtered_docs = list(supporting_docs)
            elif answer_type.lower() != "free_text" and len(supporting_docs) > 0:
                # For deterministic answers, usually the top 1 or 2 docs contain the fact.
                filtered_docs = supporting_docs[:2]

            retrieval_refs = normalize_retrieved_pages(
                RetrievalRef(doc_id=str(doc.metadata["doc_id"]), page_numbers=self._chunk_page_numbers(doc))
                for doc in filtered_docs
            )

            if answer_type == "free_text" and isinstance(answer, str):
                answer = self._clean_free_text_answer(answer).strip()

            if answer is None:
                retrieval_refs = []

        return AnswerResult(
            answer=answer,
            supporting_docs=supporting_docs,
            retrieval_refs=retrieval_refs,
            debug_metadata={"route": asdict(route), "prompt": prompt[:4000]},
            raw_response=raw_response,
        )

    def _try_bypass_llm(self, question_text: str, answer_type: str, route: RoutePlan, resolution: ResolutionContext) -> Optional[Tuple[Any, List[Document]]]:
        if answer_type.lower() == "free_text":
            return None

        lowered = question_text.lower()
        if not resolution.confident_match or not resolution.candidate_doc_ids:
            return None

        doc_ids = list(resolution.candidate_doc_ids)

        if answer_type == "boolean":
            if route.comparison_kind == "judge_overlap" and len(route.case_ids) == 2:
                return self._execute_judge_overlap_bypass(route)
            if route.comparison_kind == "party_overlap" and len(route.case_ids) == 2:
                return self._execute_party_overlap_bypass(route)

        if answer_type == "date":
            if route.comparison_kind == "issue_date" and len(route.case_ids) == 1:
                return self._execute_issue_date_lookup_bypass(route)

        if answer_type == "number":
            if ("official difc law number" in lowered or "law number" in lowered) and len(doc_ids) == 1:
                meta = self._doc_metadata.get(doc_ids[0], {})
                num = meta.get("law_number")
                if num:
                    ans = self._parse_answer_by_type(str(num), "number")
                    if ans is not None:
                        return (ans, self._build_evidence_docs([
                            FactEvidence(
                                value=str(num),
                                doc_id=doc_ids[0],
                                page_numbers=[1],
                                source_kind="metadata",
                            )
                        ]))

        if answer_type == "name":
            if route.comparison_kind == "issue_date" and len(route.case_ids) == 2:
                return self._execute_issue_date_comparison_bypass(route)
            if len(route.case_ids) == 1:
                result = self._execute_single_fact_name_bypass(route, lowered)
                if result is not None:
                    return result

        if answer_type == "names":
            if len(route.case_ids) == 1:
                result = self._execute_multi_fact_names_bypass(route, lowered)
                if result is not None:
                    return result

        return None

    def _build_evidence_docs(self, evidences: Sequence[FactEvidence]) -> List[Document]:
        docs: List[Document] = []
        seen: Set[Tuple[str, Tuple[int, ...], str]] = set()
        for evidence in evidences:
            page_numbers = sorted({int(p) for p in evidence.page_numbers if int(p) > 0})
            if not page_numbers:
                continue
            key = (evidence.doc_id, tuple(page_numbers), evidence.source_kind)
            if key in seen:
                continue
            seen.add(key)
            meta = dict(self._doc_metadata.get(evidence.doc_id, {}))
            meta["doc_id"] = evidence.doc_id
            meta["page_numbers"] = page_numbers
            docs.append(Document(page_content=f"[Fact Bypass] {evidence.source_kind}", metadata=meta))
        return docs

    def _resolve_case_record(self, case_id: str) -> Optional[CaseFactRecord]:
        return self._case_fact_index.get(self._normalize_key(case_id))

    def _resolve_unique_case_fact(
        self,
        case_id: str,
        attr_name: str,
    ) -> Optional[Tuple[str, List[FactEvidence]]]:
        record = self._resolve_case_record(case_id)
        if record is None:
            return None
        evidences = [e for e in getattr(record, attr_name, []) if e.confidence == "high"]
        if not evidences:
            return None
        values = {e.value for e in evidences}
        if len(values) != 1:
            return None
        value = next(iter(values))
        return value, evidences

    def _case_party_sets(self, case_id: str) -> Tuple[Dict[str, str], List[FactEvidence]]:
        record = self._resolve_case_record(case_id)
        if record is None:
            return {}, []
        normalized_to_surface: Dict[str, str] = {}
        evidences: List[FactEvidence] = []
        for evidence in [*record.claimants, *record.defendants]:
            normalized = self._normalize_key(evidence.value)
            if not normalized:
                continue
            normalized_to_surface.setdefault(normalized, evidence.value)
            evidences.append(evidence)
        return normalized_to_surface, evidences

    def _case_judge_sets(self, case_id: str) -> Tuple[Set[str], List[FactEvidence]]:
        record = self._resolve_case_record(case_id)
        if record is None:
            return set(), []
        evidences = [e for e in record.judges if e.confidence == "high"]
        judges = {self._normalize_key(e.value) for e in evidences if self._normalize_key(e.value)}
        return judges, evidences

    def _execute_issue_date_lookup_bypass(self, route: RoutePlan) -> Optional[Tuple[Any, List[Document]]]:
        resolved = self._resolve_unique_case_fact(route.case_ids[0], "issue_dates")
        if resolved is None:
            return None
        value, evidences = resolved
        return value, self._build_evidence_docs(evidences)

    def _execute_issue_date_comparison_bypass(self, route: RoutePlan) -> Optional[Tuple[Any, List[Document]]]:
        resolved_cases: List[Tuple[str, str, List[FactEvidence]]] = []
        for case_id in route.case_ids:
            resolved = self._resolve_unique_case_fact(case_id, "issue_dates")
            if resolved is None:
                return None
            value, evidences = resolved
            resolved_cases.append((case_id, value, evidences))
        winner = min(resolved_cases, key=lambda item: item[1])
        evidences = [evidence for _, _, case_evidences in resolved_cases for evidence in case_evidences]
        return winner[0], self._build_evidence_docs(evidences)

    def _execute_party_overlap_bypass(self, route: RoutePlan) -> Optional[Tuple[Any, List[Document]]]:
        if len(route.case_ids) != 2:
            return None
        parties_a, evidences_a = self._case_party_sets(route.case_ids[0])
        parties_b, evidences_b = self._case_party_sets(route.case_ids[1])
        if not parties_a or not parties_b:
            return None
        common = set(parties_a).intersection(parties_b)
        evidences = evidences_a + evidences_b
        return bool(common), self._build_evidence_docs(evidences)

    def _execute_judge_overlap_bypass(self, route: RoutePlan) -> Optional[Tuple[Any, List[Document]]]:
        if len(route.case_ids) != 2:
            return None
        judges_a, evidences_a = self._case_judge_sets(route.case_ids[0])
        judges_b, evidences_b = self._case_judge_sets(route.case_ids[1])
        if not judges_a or not judges_b:
            return None
        overlap = judges_a.intersection(judges_b)
        if overlap:
            matched = overlap.copy()
            evidences = [
                evidence
                for evidence in [*evidences_a, *evidences_b]
                if self._normalize_key(evidence.value) in matched
            ]
            return True, self._build_evidence_docs(evidences)
        return False, self._build_evidence_docs(evidences_a + evidences_b)

    _FACT_BYPASS_SKIP_KEYWORDS = (
        "counsel", "lawyer", "barrister", "solicitor", "representative", "law firm",
    )
    _FACT_ATTR_MAP: Sequence[Tuple[Tuple[str, ...], str]] = (
        (("judge", "justice", "presiding"), "judges"),
        (("claimant", "plaintiff", "applicant"), "claimants"),
        (("defendant", "respondent"), "defendants"),
    )

    def _detect_case_fact_attr(self, lowered: str) -> Optional[str]:
        for keywords, attr in self._FACT_ATTR_MAP:
            if any(kw in lowered for kw in keywords):
                return attr
        return None

    def _execute_single_fact_name_bypass(
        self, route: RoutePlan, lowered: str,
    ) -> Optional[Tuple[Any, List[Document]]]:
        if not route.case_ids or len(route.case_ids) != 1:
            return None
        if any(kw in lowered for kw in self._FACT_BYPASS_SKIP_KEYWORDS):
            return None
        attr_name = self._detect_case_fact_attr(lowered)
        if attr_name is None:
            return None
        resolved = self._resolve_unique_case_fact(route.case_ids[0], attr_name)
        if resolved is None:
            return None
        value, evidences = resolved
        return value, self._build_evidence_docs(evidences)

    def _execute_multi_fact_names_bypass(
        self, route: RoutePlan, lowered: str,
    ) -> Optional[Tuple[Any, List[Document]]]:
        if not route.case_ids or len(route.case_ids) != 1:
            return None
        if any(kw in lowered for kw in self._FACT_BYPASS_SKIP_KEYWORDS):
            return None
        attr_name = self._detect_case_fact_attr(lowered)
        if attr_name is None:
            return None
        record = self._resolve_case_record(route.case_ids[0])
        if record is None:
            return None
        evidences = [e for e in getattr(record, attr_name, []) if e.confidence == "high"]
        if not evidences:
            return None
        names = list(dict.fromkeys(e.value for e in evidences))
        return names, self._build_evidence_docs(evidences)

    def _build_metadata_indexes(self, source_documents: Sequence[Document]) -> None:
        self._all_doc_ids = set()
        self._claim_number_index = {}
        self._neutral_citation_index = {}
        self._law_number_index = {}
        self._law_alias_index = {}
        self._article_index = {}
        self._doc_metadata = {}
        self._case_fact_index = {}

        for document in source_documents:
            metadata = dict(document.metadata or {})
            doc_id = str(metadata.get("doc_id") or "")
            if not doc_id:
                continue
            self._all_doc_ids.add(doc_id)
            self._doc_metadata[doc_id] = metadata

            claim_number = self._normalize_key(
                str(metadata.get("claim_number_normalized") or metadata.get("claim_number") or "")
            )
            if claim_number:
                self._claim_number_index.setdefault(claim_number, set()).add(doc_id)
                record = self._case_fact_index.setdefault(
                    claim_number,
                    CaseFactRecord(claim_number=str(metadata.get("claim_number") or "")),
                )
                record.doc_ids.add(doc_id)
                self._append_case_facts(record, metadata, doc_id)

            neutral_citation = self._normalize_key(
                str(metadata.get("neutral_citation_normalized") or metadata.get("neutral_citation") or "")
            )
            if neutral_citation:
                self._neutral_citation_index.setdefault(neutral_citation, set()).add(doc_id)

            law_number = self._normalize_key(
                str(metadata.get("official_citation_normalized") or metadata.get("official_citation") or "")
            )
            if law_number:
                self._law_number_index.setdefault(law_number, set()).add(doc_id)

            for alias in metadata.get("alias_keys") or []:
                normalized_alias = self._normalize_key(str(alias))
                if normalized_alias:
                    self._law_alias_index.setdefault(normalized_alias, set()).add(doc_id)

            for article_ref, article_entry in (metadata.get("article_index") or {}).items():
                normalized_article = self._normalize_article_ref(str(article_ref))
                if not normalized_article or not isinstance(article_entry, dict):
                    continue
                entry = dict(article_entry)
                entry["doc_id"] = doc_id
                self._article_index.setdefault(normalized_article, []).append(entry)

    def _append_case_facts(self, record: CaseFactRecord, metadata: Dict[str, Any], doc_id: str) -> None:
        issue_date = self._resolve_issue_date_from_meta(metadata)
        if issue_date:
            record.issue_dates.append(
                FactEvidence(
                    value=issue_date,
                    doc_id=doc_id,
                    page_numbers=self._title_evidence_pages(metadata),
                    source_kind="issue_date",
                )
            )

        claimant = str(metadata.get("claimant") or "").strip()
        if claimant:
            record.claimants.append(
                FactEvidence(
                    value=claimant,
                    doc_id=doc_id,
                    page_numbers=[1],
                    source_kind="claimant",
                )
            )

        for defendant in metadata.get("defendants") or []:
            defendant_text = str(defendant or "").strip()
            if not defendant_text:
                continue
            record.defendants.append(
                FactEvidence(
                    value=defendant_text,
                    doc_id=doc_id,
                    page_numbers=[1],
                    source_kind="defendant",
                )
            )

        record.judges.extend(self._extract_judge_evidences(metadata, doc_id))

    def _resolve_issue_date_from_meta(self, metadata: Dict[str, Any]) -> Optional[str]:
        raw_value = metadata.get("judgment_release_date") or metadata.get("judgment_date")
        if not raw_value:
            return None
        return self._parse_non_iso_date(str(raw_value))

    def _title_evidence_pages(self, metadata: Dict[str, Any]) -> List[int]:
        page_count = int(metadata.get("page_count") or 0)
        if page_count >= 2:
            return [1, 2]
        return [1]

    def _extract_judge_evidences(self, metadata: Dict[str, Any], doc_id: str) -> List[FactEvidence]:
        blocks = metadata.get("blocks") or []
        evidences: List[FactEvidence] = []
        seen: Set[Tuple[str, int]] = set()
        for block in blocks:
            page_number = int(block.get("page_number") or 0)
            block_index = int(block.get("block_index") or 0)
            if page_number <= 0 or page_number > 3 or block_index > 80:
                continue
            text = " ".join(str(block.get("text") or "").split())
            if not text:
                continue
            if not self._is_high_confidence_judge_block(text):
                continue
            for judge_name in self._extract_judge_names_from_text(text):
                key = (judge_name, page_number)
                if key in seen:
                    continue
                seen.add(key)
                evidences.append(
                    FactEvidence(
                        value=judge_name,
                        doc_id=doc_id,
                        page_numbers=[page_number],
                        source_kind="judge_header",
                    )
                )
        return evidences

    _JUDGE_BLOCK_MARKERS = (
        "BEFORE ",
        "BEFORE:",
        "ISSUED BY:",
        "ISSUED BY ",
        "ORDER WITH REASONS OF",
        "AMENDED ORDER WITH REASONS OF",
        "REASONS FOR THE ORDER OF",
        "JUDGMENT OF",
        "ORDER OF H.E.",
        "ORDER OF HIS EXCELLENCY",
        "ORDER OF HER EXCELLENCY",
        "REASONS OF H.E.",
        "REASONS OF HIS EXCELLENCY",
        "REASONS OF HER EXCELLENCY",
        "JUDGMENT OF H.E.",
        "JUDGMENT OF HIS EXCELLENCY",
        "JUDGMENT OF HER EXCELLENCY",
        "BEFORE HIS EXCELLENCY",
        "BEFORE HER EXCELLENCY",
        "BEFORE THE HONOURABLE",
        "BEFORE THE HONORABLE",
    )

    def _is_high_confidence_judge_block(self, text: str) -> bool:
        upper = " ".join(text.upper().split())
        if not any(upper.startswith(m) for m in self._JUDGE_BLOCK_MARKERS):
            return False
        return "JUSTICE" in upper or "JUDGE" in upper

    def _extract_judge_names_from_text(self, text: str) -> List[str]:
        candidates: List[str] = []
        normalized_text = " ".join(text.replace("\n", " ").split())
        for match in re.finditer(
            r"(?:(?:H\.E\.|HIS\s+EXCELLENCY|HER\s+EXCELLENCY|THE\s+HONOU?RABLE)\s+)?"
            r"(?:(?:CHIEF|DEPUTY)\s+)?JUSTICE\s+([A-Z][A-Za-z\s.'\-]+?(?:KC|KBE|CBE|QC)?)"
            r"(?=,| AND | DATED | ON |\s*$)",
            normalized_text,
            re.IGNORECASE,
        ):
            name = " ".join(match.group(1).split()).strip(" ,.")
            if name and len(name) > 2:
                candidates.append(name.upper())
        if "ISSUED BY:" in normalized_text.upper():
            issued_match = re.search(r"Issued by:\s*([A-Z][A-Za-z\s.'-]+)", normalized_text, re.IGNORECASE)
            if issued_match:
                name = " ".join(issued_match.group(1).split()).strip(" ,.")
                if name and len(name) > 2:
                    candidates.append(name.upper())
        unique: List[str] = []
        seen: Set[str] = set()
        for candidate in candidates:
            if candidate in seen:
                continue
            seen.add(candidate)
            unique.append(candidate)
        return unique

    def _resolve_route_context(self, route: RoutePlan) -> ResolutionContext:
        candidate_doc_ids: Set[str] = set()
        confident_match = False
        article_entries: Dict[str, List[Dict[str, Any]]] = {}

        for case_id in route.case_ids:
            candidate_doc_ids.update(self._claim_number_index.get(self._normalize_key(case_id), set()))
        if candidate_doc_ids:
            confident_match = True

        if not candidate_doc_ids:
            for neutral_citation in route.neutral_citations:
                candidate_doc_ids.update(
                    self._neutral_citation_index.get(self._normalize_key(neutral_citation), set())
                )
            if candidate_doc_ids:
                confident_match = True

        if not candidate_doc_ids:
            for law_number in route.law_numbers:
                candidate_doc_ids.update(self._law_number_index.get(self._normalize_key(law_number), set()))
            if candidate_doc_ids:
                confident_match = True

        if not candidate_doc_ids:
            for law_title in route.law_title_candidates:
                candidate_doc_ids.update(self._law_alias_index.get(self._normalize_key(law_title), set()))
            if not candidate_doc_ids:
                for law_title in route.law_title_candidates:
                    candidate_doc_ids.update(self._fuzzy_match_law_alias(self._normalize_key(law_title)))
            if candidate_doc_ids:
                confident_match = True

        if route.article_refs:
            for article_ref in route.article_refs:
                normalized_article = self._normalize_article_ref(article_ref)
                for entry in self._article_index.get(normalized_article, []):
                    doc_id = str(entry.get("doc_id") or "")
                    if not doc_id:
                        continue
                    article_entries.setdefault(doc_id, []).append(entry)

            if candidate_doc_ids and article_entries:
                restricted = {
                    doc_id: entries
                    for doc_id, entries in article_entries.items()
                    if doc_id in candidate_doc_ids
                }
                if restricted:
                    article_entries = restricted
                    candidate_doc_ids = set(restricted)

        return ResolutionContext(
            candidate_doc_ids=candidate_doc_ids or set(self._all_doc_ids),
            confident_match=confident_match,
            article_entries=article_entries,
        )

    _LAW_FILLER_TOKENS = frozenset({"law", "the", "of", "and", "in", "on", "for", "difc", "a", "an"})

    def _fuzzy_match_law_alias(self, query_key: str) -> Set[str]:
        if not query_key:
            return set()
        query_tokens = set(query_key.split())
        significant = query_tokens - self._LAW_FILLER_TOKENS
        if not significant:
            return set()

        best: Set[str] = set()
        best_score = 0.0

        for alias_key, doc_ids in self._law_alias_index.items():
            if query_key in alias_key or alias_key in query_key:
                if best_score < 1.0:
                    best_score = 1.0
                    best = set(doc_ids)
                else:
                    best.update(doc_ids)
                continue

            alias_significant = set(alias_key.split()) - self._LAW_FILLER_TOKENS
            if not alias_significant:
                continue
            overlap = significant & alias_significant
            if not overlap:
                continue
            q_cov = len(overlap) / len(significant)
            a_cov = len(overlap) / len(alias_significant)
            score = q_cov * 0.7 + a_cov * 0.3
            if score >= 0.55 and score > best_score:
                best_score = score
                best = set(doc_ids)
            elif score >= 0.55 and abs(score - best_score) < 0.01:
                best.update(doc_ids)

        return best

    def _filter_chunks(self, chunks: Sequence[Document], candidate_doc_ids: Set[str]) -> List[Document]:
        return [
            chunk
            for chunk in chunks
            if str(chunk.metadata.get("doc_id") or "") in candidate_doc_ids
        ]

    def _dense_search(
        self,
        question_text: str,
        resolution: ResolutionContext,
        route: RoutePlan,
    ) -> List[Document]:
        query_vec = self.embedding_model.embed_query(question_text)
        base_k = self.dense_candidate_k * 3 if resolution.candidate_doc_ids and resolution.candidate_doc_ids != self._all_doc_ids else self.dense_candidate_k
        if route.prefer_title_page or route.prefer_last_page:
            base_k = min(base_k * 2, 50)
        if self.store is None:
            if not self._local_dense_vectors_normed:
                return []

            # Compute similarities only within the candidate doc-id set.
            candidate_doc_ids = resolution.candidate_doc_ids
            q_norm = math.sqrt(sum(float(x) * float(x) for x in query_vec)) or 1.0
            query_normed = [float(x) / q_norm for x in query_vec]

            similarities: List[Tuple[float, int]] = []
            for i, doc_id in enumerate(self._local_dense_doc_ids):
                if candidate_doc_ids and doc_id not in candidate_doc_ids:
                    continue
                vec = self._local_dense_vectors_normed[i]
                sim = sum(a * b for a, b in zip(query_normed, vec))
                similarities.append((sim, i))

            if not similarities:
                raw = []
            else:
                # Pick top base_k most similar chunks.
                similarities.sort(key=lambda t: t[0], reverse=True)
                raw = [self.raw_chunks[i] for _, i in similarities[:base_k]]
        else:
            raw = self.store.vector_search(query_vec, top_k=base_k)

        filtered = [
            doc
            for doc in raw
            if str(doc.metadata.get("doc_id") or "") in resolution.candidate_doc_ids
        ]
        take = self.dense_candidate_k * 2 if (route.prefer_title_page or route.prefer_last_page) else self.dense_candidate_k
        take = min(take, len(filtered) if filtered else len(raw))
        results = filtered or raw[: self.dense_candidate_k]
        dense_docs = results[:take]
        return self._strictly_filter_route(self._apply_route_bias(dense_docs, route, resolution), route, resolution)

    def _sparse_search(
        self,
        question_text: str,
        sparse_pool: Sequence[Document],
        route: RoutePlan,
        resolution: ResolutionContext,
    ) -> List[Document]:
        if not sparse_pool:
            return []
        
        # We can still use text_search on turbopuffer but it searches globally, then we filter.
        sparse_query = self._build_sparse_query(question_text, route)
        top_k_sparse = self.sparse_candidate_k * 2
        if route.prefer_title_page or route.prefer_last_page:
            top_k_sparse = min(top_k_sparse * 2, 60)

        if self.store is None:
            if self._local_bm25_retriever is None:
                return []
            original_k = getattr(self._local_bm25_retriever, "k", None)
            try:
                if original_k is not None:
                    self._local_bm25_retriever.k = top_k_sparse
                raw = self._local_bm25_retriever.invoke(sparse_query)
            finally:
                if original_k is not None:
                    self._local_bm25_retriever.k = original_k
        else:
            raw = self.store.text_search(sparse_query, top_k=top_k_sparse)

        filtered = [
            doc for doc in raw
            if str(doc.metadata.get("doc_id") or "") in resolution.candidate_doc_ids
        ]
        take = self.sparse_candidate_k * 2 if (route.prefer_title_page or route.prefer_last_page) else self.sparse_candidate_k
        results = filtered or raw[: self.sparse_candidate_k]
        return self._strictly_filter_route(
            self._apply_route_bias(results[:take], route, resolution), route, resolution
        )

    def _build_sparse_query(self, question_text: str, route: RoutePlan) -> str:
        tokens = [question_text]
        tokens.extend(route.case_ids)
        tokens.extend(route.neutral_citations)
        tokens.extend(route.article_refs)
        tokens.extend(route.law_title_candidates)
        tokens.extend(route.law_numbers)
        if route.prefer_title_page:
            tokens.extend(["title page", "cover page", "law number"])
        if route.prefer_last_page:
            tokens.extend(["last page", "order", "outcome"])
        return " ".join(token for token in tokens if token).strip()
    
    def _strictly_filter_route(
        self,
        docs: Sequence[Document],
        route: RoutePlan,
        resolution: ResolutionContext,
    ) -> List[Document]:
        def strictly_filter(doc: Document) -> bool:
            metadata = doc.metadata or {}
            doc_id = str(metadata.get("doc_id") or "")
            chunk_kind = str(metadata.get("chunk_kind") or "")
            page_numbers = self._chunk_page_numbers(doc)
            if route.target_pages and (not any(page in page_numbers for page in route.target_pages)):
                return False
            if route.prefer_title_page and chunk_kind != "title_page":
                return False
            if resolution.confident_match and route.article_refs and resolution.article_entries:
                entries = resolution.article_entries.get(doc_id, [])
                if entries and not any(self._chunk_overlaps_entry(doc, entry) for entry in entries):
                    return False
            if route.prefer_last_page:
                doc_pages = int(self._doc_metadata.get(doc_id, {}).get("page_count") or 0)
                if not (doc_pages and page_numbers and max(page_numbers) == doc_pages):
                    return False
            return True

        filtered = list(filter(strictly_filter, docs))
        if not filtered and docs and (route.prefer_title_page or route.prefer_last_page):
            return list(docs)
        if route.prefer_title_page and 0 < len(filtered) < self.top_k_docs:
            seen = {BaseRAGRetriever.fusion_key(d) for d in filtered}
            doc_ids_from_title = {str(d.metadata.get("doc_id") or "") for d in filtered}
            for doc in docs:
                if BaseRAGRetriever.fusion_key(doc) in seen:
                    continue
                if str(doc.metadata.get("doc_id") or "") not in doc_ids_from_title:
                    continue
                kind = str((doc.metadata or {}).get("chunk_kind") or "")
                if kind in ("title_page", "page_anchor"):
                    filtered.append(doc)
                    seen.add(BaseRAGRetriever.fusion_key(doc))
                    if len(filtered) >= self.top_k_docs * 2:
                        break
            def title_first_key(d: Document) -> tuple:
                k = str((d.metadata or {}).get("chunk_kind") or "")
                return (0 if k == "title_page" else 1, -int((d.metadata or {}).get("page_start") or 0))
            filtered = sorted(filtered, key=title_first_key)[: self.top_k_docs]
        if route.prefer_last_page and 0 < len(filtered) < self.top_k_docs:
            seen = {BaseRAGRetriever.fusion_key(d) for d in filtered}
            for doc in docs:
                if BaseRAGRetriever.fusion_key(doc) in seen:
                    continue
                meta = doc.metadata or {}
                page_numbers = self._chunk_page_numbers(doc)
                doc_pages = int(
                    self._doc_metadata.get(str(meta.get("doc_id") or ""), {}).get("page_count")
                    or 0
                )
                if doc_pages and page_numbers and max(page_numbers) >= doc_pages - 1:
                    filtered.append(doc)
                    seen.add(BaseRAGRetriever.fusion_key(doc))
                    if len(filtered) >= self.top_k_docs * 2:
                        break
            def last_page_sort_key(d: Document) -> tuple:
                meta = d.metadata or {}
                pns = self._chunk_page_numbers(d)
                dp = int(self._doc_metadata.get(str(meta.get("doc_id") or ""), {}).get("page_count") or 0)
                on_last = 0 if (dp and pns and max(pns) == dp) else 1
                return (on_last, -int(meta.get("page_start") or 0))
            filtered = sorted(filtered, key=last_page_sort_key)[: self.top_k_docs]
        return filtered

    def _apply_route_bias(
        self,
        docs: Sequence[Document],
        route: RoutePlan,
        resolution: ResolutionContext,
    ) -> List[Document]:
        def score(doc: Document) -> Tuple[int, int, int, int]:
            metadata = doc.metadata or {}
            doc_id = str(metadata.get("doc_id") or "")
            chunk_kind = str(metadata.get("chunk_kind") or "")
            page_numbers = self._chunk_page_numbers(doc)
            preferred_kind = 1 if chunk_kind in route.preferred_chunk_kinds else 0
            target_page_hit = 0
            article_hit = 0
            if route.target_pages and any(page in page_numbers for page in route.target_pages):
                target_page_hit = 1
            if route.prefer_title_page and chunk_kind == "title_page":
                target_page_hit = 2
            if route.prefer_last_page:
                doc_pages = int(self._doc_metadata.get(doc_id, {}).get("page_count") or 0)
                if doc_pages and page_numbers and max(page_numbers) == doc_pages:
                    target_page_hit = 2
            if route.article_refs and resolution.article_entries.get(doc_id):
                article_hit = 1 if any(self._chunk_overlaps_entry(doc, entry) for entry in resolution.article_entries[doc_id]) else 0
            return (article_hit, preferred_kind, target_page_hit, -int(metadata.get("page_start") or 0))

        return sorted(docs, key=score, reverse=True)

    def _fuse_results(
        self,
        dense_docs: Sequence[Document],
        sparse_docs: Sequence[Document],
        k: int,
    ) -> List[Document]:
        scores: Dict[str, float] = {}
        docs_by_key: Dict[str, Document] = {}

        def accumulate(documents: Sequence[Document], weight: float) -> None:
            for rank, doc in enumerate(documents):
                key = BaseRAGRetriever.fusion_key(doc)
                docs_by_key.setdefault(key, doc)
                scores[key] = scores.get(key, 0.0) + (weight / float(self.rrf_k + rank + 1))

        accumulate(dense_docs, self.dense_weight)
        accumulate(sparse_docs, self.sparse_weight)
        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        return [docs_by_key[key] for key, _ in ranked[:k]]

    def _rerank(self, question_text: str, docs: Sequence[Document]) -> List[Document]:
        if not docs:
            return []
        if self.reranker is None:
            return list(docs[: self.top_k_docs])
        try:
            reranked = self.reranker.rerank(question_text, list(docs), top_k=self.top_k_docs)
            return reranked.documents
        except Exception as exc:
            logger.warning("Reranking failed, falling back to fused order: %s", exc)
            return list(docs[: self.top_k_docs])

    def _retrieve_for_comparison(
        self,
        question_text: str,
        answer_type: str,
        route: RoutePlan,
    ) -> List[Document]:
        combined: List[Document] = []
        for case_id in route.case_ids:
            subroute = RoutePlan(
                question_text=route.question_text,
                answer_type=route.answer_type,
                case_ids=[case_id],
                neutral_citations=list(route.neutral_citations),
                article_refs=list(route.article_refs),
                law_title_candidates=list(route.law_title_candidates),
                law_numbers=list(route.law_numbers),
                target_pages=list(route.target_pages),
                prefer_last_page=route.prefer_last_page,
                prefer_title_page=route.prefer_title_page,
                comparison_mode=False,
                comparison_kind=route.comparison_kind,
                common_entity_mode=route.common_entity_mode,
                page_specific_mode=route.page_specific_mode,
                preferred_chunk_kinds=list(route.preferred_chunk_kinds),
            )
            docs, _ = self.retrieve(question_text, answer_type=answer_type, route=subroute)
            take = docs[: max(2, self.top_k_docs // max(1, len(route.case_ids)))]
            for doc in take:
                meta = dict(doc.metadata or {})
                meta["comparison_case_id"] = case_id
                combined.append(Document(page_content=doc.page_content, metadata=meta, id=doc.id))

        deduped: Dict[str, Document] = {}
        for doc in combined:
            deduped[BaseRAGRetriever.fusion_key(doc)] = doc
        return list(deduped.values())[: self.top_k_docs]

    def _build_prompt(
        self,
        question_text: str,
        answer_type: str,
        supporting_docs: Sequence[Document],
        route: RoutePlan,
    ) -> str:
        if answer_type.lower() == "free_text":
            return self._build_free_text_prompt(question_text, supporting_docs)

        if route.comparison_mode and len(route.case_ids) > 1:
            case_contexts: List[str] = []
            for case_id in route.case_ids:
                case_docs = [
                    doc
                    for doc in supporting_docs
                    if self._normalize_key(str((doc.metadata or {}).get("comparison_case_id") or (doc.metadata or {}).get("claim_number") or "")) == self._normalize_key(case_id)
                ]
                chunks = []
                for index, doc in enumerate(case_docs, start=1):
                    metadata = doc.metadata or {}
                    chunks.append(
                        "\n".join(
                            [
                                f"[Context {index}]",
                                f"doc_id={metadata.get('doc_id', '')}",
                                f"claim_number={metadata.get('claim_number', '')}",
                                f"heading={metadata.get('heading', '')}",
                                f"pages={self._chunk_page_numbers(doc)}",
                                doc.page_content,
                            ]
                        )
                    )
                if chunks:
                    case_contexts.append(f"Case Evidence: {case_id}\n" + "\n\n".join(chunks))
            if case_contexts:
                context = "\n\n".join(case_contexts)
            else:
                context = "\n\n".join(doc.page_content for doc in supporting_docs)
        else:
            context_chunks = []
            for index, doc in enumerate(supporting_docs, start=1):
                metadata = doc.metadata or {}
                context_chunks.append(
                    "\n".join(
                        [
                            f"[Context {index}]",
                            f"doc_id={metadata.get('doc_id', '')}",
                            f"claim_number={metadata.get('claim_number', '')}",
                            f"heading={metadata.get('heading', '')}",
                            f"pages={self._chunk_page_numbers(doc)}",
                            doc.page_content,
                        ]
                    )
                )
            context = "\n\n".join(context_chunks)
        instruction = self._type_instruction(answer_type)
        return (
            "You answer legal challenge questions using only the provided context.\n"
            "If the context is insufficient, return null for deterministic types or the standard absence statement for free_text.\n"
            f"{instruction}\n\n"
            f"Question: {question_text}\n\n"
            f"Context:\n{context}\n\n"
            "Answer (only the answer, nothing else):"
        )

    def _build_free_text_prompt(
        self,
        question_text: str,
        supporting_docs: Sequence[Document],
    ) -> str:
        subtype = self._select_free_text_prompt_subtype(question_text, supporting_docs)
        return build_free_text_prompt(question_text, supporting_docs, subtype, self._default_absent_answer("free_text"))

    def _select_free_text_prompt_subtype(
        self,
        question_text: str,
        supporting_docs: Sequence[Document],
    ) -> str:
        return detect_free_text_subtype(question_text, supporting_docs)

    def _type_instruction(self, answer_type: str) -> str:
        normalized = answer_type.lower()
        strict = " Output only that value, nothing else (no period, explanation, or extra words)."
        if normalized == "boolean":
            return "Return only the word true or the word false." + strict
        if normalized == "number":
            return "Return only a single number (integer or decimal, e.g. 1000 or 1000.0)." + strict
        if normalized == "name":
            return "Return only the exact name or entity from the context." + strict
        if normalized == "names":
            return "Return only a semicolon-separated list of exact names from the context (e.g. Name A; Name B)." + strict
        if normalized == "date":
            return "Return only the date in YYYY-MM-DD format (e.g. 2026-02-02)." + strict
        return "Return a concise answer grounded in the context, max 260 characters."

    def _invoke_llm(
        self,
        prompt: str,
        telemetry_timer: Optional[Any] = None,
        timing_out: Optional[Dict[str, float]] = None,
    ) -> str:
        import time as _time
        t_start = _time.perf_counter()
        if hasattr(self.llm, "stream"):
            response_chunks = []
            ttft_marked = False
            for chunk in self.llm.stream(prompt):
                if hasattr(chunk, "content"):
                    content = chunk.content
                    if isinstance(content, list):
                        text = "".join(
                            part.get("text", "") if isinstance(part, dict) else str(part)
                            for part in content
                        )
                    else:
                        text = str(content)
                else:
                    text = str(chunk)
                if not ttft_marked and text.strip():
                    if telemetry_timer:
                        telemetry_timer.mark_token()
                    if timing_out is not None:
                        timing_out["llm_ttft_ms"] = (_time.perf_counter() - t_start) * 1000
                    ttft_marked = True
                response_chunks.append(text)
            if not ttft_marked and (telemetry_timer or timing_out is not None):
                if telemetry_timer:
                    telemetry_timer.mark_token()
                if timing_out is not None:
                    timing_out["llm_ttft_ms"] = (_time.perf_counter() - t_start) * 1000
            if timing_out is not None:
                timing_out["llm_total_ms"] = (_time.perf_counter() - t_start) * 1000
            return "".join(response_chunks).strip()
        else:
            response = self.llm.invoke(prompt)
            total_ms = (_time.perf_counter() - t_start) * 1000
            if telemetry_timer:
                telemetry_timer.mark_token()
            if timing_out is not None:
                timing_out["llm_total_ms"] = total_ms
                timing_out["llm_ttft_ms"] = total_ms
            if hasattr(response, "content"):
                content = response.content
                if isinstance(content, list):
                    return "".join(part.get("text", "") if isinstance(part, dict) else str(part) for part in content).strip()
                return str(content).strip()
            return str(response).strip()

    def _extract_deterministic_candidate(self, raw: str) -> str:
        """Extract the answer substring for deterministic types: strip prefixes, take first line."""
        if not (raw or "").strip():
            return ""
        text = raw.strip()
        for prefix in (
            r"Answer:\s*",
            r"The answer is\s*",
            r"Answer is\s*",
            r"The value is\s*",
            r"Result:\s*",
        ):
            m = re.search(prefix, text, re.IGNORECASE)
            if m:
                text = text[m.end() :].strip()
                break
        text = re.sub(r"^[\[\]`\"]+|[\[\]`\"]+$", "", text).strip()
        first_line = text.split("\n")[0].strip()
        first_sentence = re.split(r"[.!?]\s+", first_line)[0].strip()
        out = (first_sentence or first_line or text).strip()
        return out.rstrip(".,;:") if out else out

    def _parse_answer_by_type(self, raw: str, answer_type: str) -> Any:
        text = (raw or "").strip()
        lowered = text.lower()
        absent_phrases = {
            "null", "none", "not found", "unknown", "insufficient information", "n/a",
            "cannot be determined", "not stated", "no information", "unable to determine",
            "does not appear", "is not stated", "not in the context", "not in the documents",
            "not found in the documents", "no information available", "information not found",
        }
        if lowered in absent_phrases:
            return None

        normalized = answer_type.lower()
        if normalized == "boolean":
            first_word = re.split(r"[\s.,;:]+", lowered)[0] if lowered else ""
            if first_word in {"true", "yes"}:
                return True
            if first_word in {"false", "no"}:
                return False
            if re.match(r"^true\b", lowered):
                return True
            if re.match(r"^false\b", lowered):
                return False
            return None
        if normalized == "number":
            cleaned = text.replace(",", "")
            numbers = re.findall(r"-?\d+(?:\.\d+)?", cleaned)
            if not numbers:
                return None
            number_text = numbers[0]
            return float(number_text) if "." in number_text else int(number_text)
        if normalized == "names":
            if not text:
                return None
            normalized_text = re.sub(r"\s+and\s+", ";", text, flags=re.IGNORECASE)
            names = [
                " ".join(item.strip().split()).strip(".,;:\"'")
                for item in re.split(r";|,|\n", normalized_text)
                if item.strip()
            ]
            return names or None
        if normalized == "date":
            iso_match = re.search(r"\b\d{4}-\d{2}-\d{2}\b", text)
            if iso_match:
                return iso_match.group(0)
            parsed = self._parse_non_iso_date(text)
            return parsed
        if normalized == "name":
            cleaned = " ".join(text.split()).strip(".,;:\"'")
            return cleaned or None
        cleaned = " ".join(text.split())
        if cleaned.lower().startswith("there is no information"):
            return self._default_absent_answer("free_text")
        return cleaned or self._default_absent_answer("free_text")

    _MONTH_NAMES_RE = (
        r"January|February|March|April|May|June|July|August"
        r"|September|October|November|December"
    )

    def _parse_non_iso_date(self, text: str) -> Optional[str]:
        from datetime import datetime

        cleaned = re.sub(r"(\d+)(?:st|nd|rd|th)\b", r"\1", text.strip())
        t = " ".join(cleaned.split()).title()
        for fmt in (
            "%B %d, %Y",
            "%b %d, %Y",
            "%d %B %Y",
            "%d %b %Y",
            "%B %d %Y",
            "%b %d %Y",
            "%d/%m/%Y",
            "%d-%m-%Y",
            "%d.%m.%Y",
            "%Y/%m/%d",
        ):
            try:
                return datetime.strptime(t, fmt).strftime("%Y-%m-%d")
            except ValueError:
                continue
        day_month_year = re.search(
            rf"(\d{{1,2}})\s+({self._MONTH_NAMES_RE})\s+(\d{{4}})", t, re.IGNORECASE,
        )
        if day_month_year:
            try:
                return datetime.strptime(
                    f"{day_month_year.group(1)} {day_month_year.group(2)} {day_month_year.group(3)}", "%d %B %Y",
                ).strftime("%Y-%m-%d")
            except ValueError:
                pass
        month_day_year = re.search(
            rf"({self._MONTH_NAMES_RE})\s+(\d{{1,2}}),?\s+(\d{{4}})", t, re.IGNORECASE,
        )
        if month_day_year:
            try:
                return datetime.strptime(
                    f"{month_day_year.group(2)} {month_day_year.group(1)} {month_day_year.group(3)}", "%d %B %Y",
                ).strftime("%Y-%m-%d")
            except ValueError:
                pass
        return None

    def _default_absent_answer(self, answer_type: str) -> Any:
        if answer_type.lower() == "free_text":
            return "There is no information on this question in the provided documents."
        return None

    def _clean_free_text_answer(self, answer: str) -> str:
        """Strip filler prefixes to improve clarity."""
        if not answer or not isinstance(answer, str):
            return answer
        text = " ".join(answer.split())
        for prefix in (
            "Based on the context, ",
            "Based on the provided context, ",
            "According to the document, ",
            "According to the context, ",
            "From the context, ",
            "The context states that ",
            "In the provided documents, ",
        ):
            if text.startswith(prefix):
                text = text[len(prefix) :].strip()
                break
        return text

    def _serialize_chunk(self, chunk: Document) -> Document:
        metadata = sanitize_metadata_for_vectorstore(dict(chunk.metadata or {}))
        return Document(page_content=chunk.page_content, metadata=metadata, id=chunk.id)

    def _decode_dense_document(self, doc: Document) -> Document:
        metadata = dict(doc.metadata or {})
        metadata["page_numbers"] = decode_page_numbers(metadata)
        if isinstance(metadata.get("section_path"), str):
            metadata["section_path"] = [item.strip() for item in str(metadata["section_path"]).split(",") if item.strip()]
        return Document(page_content=doc.page_content, metadata=metadata, id=doc.id)

    def _chunk_page_numbers(self, doc: Document) -> List[int]:
        return decode_page_numbers(doc.metadata or {})

    def _chunk_overlaps_entry(self, doc: Document, entry: Dict[str, Any]) -> bool:
        metadata = doc.metadata or {}
        chunk_start = int(metadata.get("block_start") or 0)
        chunk_end = int(metadata.get("block_end") or 0)
        entry_start = int(entry.get("block_start") or 0)
        entry_end = int(entry.get("block_end") or 0)
        if chunk_end and entry_end and max(chunk_start, entry_start) <= min(chunk_end, entry_end):
            return True
        chunk_pages = set(self._chunk_page_numbers(doc))
        entry_pages = {
            int(entry.get("page_start") or 0),
            int(entry.get("page_end") or 0),
        }
        entry_pages.discard(0)
        return bool(chunk_pages & entry_pages)

    def _normalize_article_ref(self, value: str) -> str:
        cleaned = re.sub(r"\barticle\b", "Article", value or "", flags=re.IGNORECASE)
        cleaned = re.sub(r"\s+", "", cleaned.replace("Article", ""))
        return f"article:{cleaned.lower()}" if cleaned else ""

    def _normalize_key(self, value: str) -> str:
        cleaned = re.sub(r"[^a-z0-9]+", " ", (value or "").strip().lower())
        return re.sub(r"\s+", " ", cleaned).strip()


HybridRAGPipeline = LegalHybridRAGPipeline
