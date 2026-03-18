"""Legal-challenge hybrid RAG pipeline built on ingest outputs."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import logging
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
        self.store = store or TurbopufferStore(namespace="legal-challenge", region="gcp-europe-west3")
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

        if self.skip_indexing:
            logger.info("Skipping Turbopuffer index (skip_indexing=True); using existing namespace.")
            return self.store

        def _embed(texts: List[str]) -> List[List[float]]:
            return self.embedding_model.embed_documents(texts)

        logger.info("Indexing %s legal chunks into Turbopuffer", len(self.raw_chunks))
        self.store.index_chunks(self.raw_chunks, _embed)
        return self.store

    def retrieve(
        self,
        question_text: str,
        answer_type: str = "free_text",
        route: Optional[RoutePlan] = None,
    ) -> Tuple[List[Document], RoutePlan]:
        if not self.raw_chunks:
            self.build_indexes()

        active_route = route or self.router.route(question_text, answer_type)
        resolution = self._resolve_route_context(active_route)
        sparse_pool = self._filter_chunks(self.raw_chunks, resolution.candidate_doc_ids)

        dense_docs = self._dense_search(question_text, resolution, active_route)
        sparse_docs = self._sparse_search(question_text, sparse_pool, active_route, resolution)
        fused_docs = self._fuse_results(dense_docs, sparse_docs, self.top_k_docs * 4)
        reranked_docs = self._rerank(question_text, fused_docs)
        return reranked_docs[: self.top_k_docs], active_route

    def answer_question(self, question_item: Dict[str, Any], telemetry_timer: Optional[Any] = None) -> AnswerResult:
        question_text = question_item["question"]
        answer_type = question_item.get("answer_type", "free_text")
        route = self.router.route(question_text, answer_type)

        resolution = self._resolve_route_context(route)
        bypass_result = self._try_bypass_llm(question_text, answer_type, route, resolution)
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
            supporting_docs = self._retrieve_for_comparison(question_text, answer_type, route)
        else:
            supporting_docs, route = self.retrieve(question_text, answer_type=answer_type, route=route)

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

        prompt = self._build_prompt(question_text, answer_type, supporting_docs, route)
        raw_response = self._invoke_llm(prompt, telemetry_timer=telemetry_timer)
        
        used_indices = []
        parsed_answer = raw_response
        
        if answer_type.lower() == "free_text":
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
                answer = answer[:280].strip()

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

        def _get_bypass_docs(d_ids: List[str], use_page_1_and_2: bool = False) -> List[Document]:
            docs = []
            for d_id in d_ids:
                meta = dict(self._doc_metadata.get(d_id, {}))
                meta["doc_id"] = d_id
                if use_page_1_and_2 and int(meta.get("page_count") or 0) >= 2:
                    meta["page_numbers"] = [1, 2]
                else:
                    meta["page_numbers"] = [1]
                docs.append(Document(page_content="[Metadata Bypass]", metadata=meta))
            return docs

        if answer_type == "boolean":
            if route.common_entity_mode and len(route.case_ids) == 2 and len(doc_ids) == 2:
                if "judge" not in lowered:
                    parties_sets = []
                    for d_id in doc_ids:
                        meta = self._doc_metadata.get(d_id, {})
                        parties = set()
                        if meta.get("claimant"): parties.add(str(meta["claimant"]).lower())
                        for d in meta.get("defendants", []): parties.add(str(d).lower())
                        parties_sets.append(parties)
                    
                    if len(parties_sets) == 2 and (parties_sets[0] or parties_sets[1]):
                        common = parties_sets[0].intersection(parties_sets[1])
                        return (bool(common), _get_bypass_docs(doc_ids))

        if answer_type == "date":
            if ("date of issue" in lowered or "issue date" in lowered) and len(doc_ids) == 1:
                meta = self._doc_metadata.get(doc_ids[0], {})
                d = meta.get("judgment_date") or meta.get("judgment_release_date")
                if d:
                    ans = self._parse_answer_by_type(d, "date")
                    if ans:
                        return (ans, _get_bypass_docs(doc_ids, use_page_1_and_2=True))
                        
        if answer_type == "number":
            if ("official difc law number" in lowered or "law number" in lowered) and len(doc_ids) == 1:
                meta = self._doc_metadata.get(doc_ids[0], {})
                num = meta.get("law_number")
                if num:
                    ans = self._parse_answer_by_type(str(num), "number")
                    if ans is not None:
                        return (ans, _get_bypass_docs(doc_ids))

        if answer_type == "name":
            if ("earlier issue date" in lowered or "earlier date" in lowered) and len(route.case_ids) == 2 and len(doc_ids) == 2:
                dates_and_cases = []
                for d_id in doc_ids:
                    meta = self._doc_metadata.get(d_id, {})
                    d = meta.get("judgment_date") or meta.get("judgment_release_date")
                    claim_num = meta.get("claim_number")
                    if d and claim_num:
                        dates_and_cases.append((d, claim_num))
                if len(dates_and_cases) == 2:
                    dates_and_cases.sort()
                    return (dates_and_cases[0][1], _get_bypass_docs(doc_ids, use_page_1_and_2=True))
                    
        return None

    def _build_metadata_indexes(self, source_documents: Sequence[Document]) -> None:
        self._all_doc_ids = set()
        self._claim_number_index = {}
        self._neutral_citation_index = {}
        self._law_number_index = {}
        self._law_alias_index = {}
        self._article_index = {}
        self._doc_metadata = {}

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
        raw = self.store.vector_search(query_vec, top_k=base_k)
        filtered = [
            doc for doc in raw
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
        
        # We can still use text_search on turbopuffer but it searches globally, then we filter
        sparse_query = self._build_sparse_query(question_text, route)
        top_k_sparse = self.sparse_candidate_k * 2
        if route.prefer_title_page or route.prefer_last_page:
            top_k_sparse = min(top_k_sparse * 2, 60)
            
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
                common_entity_mode=route.common_entity_mode,
                page_specific_mode=route.page_specific_mode,
                preferred_chunk_kinds=list(route.preferred_chunk_kinds),
            )
            docs, _ = self.retrieve(question_text, answer_type=answer_type, route=subroute)
            combined.extend(docs[: max(2, self.top_k_docs // max(1, len(route.case_ids)))])

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
            "Answer:"
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
        if normalized == "boolean":
            return "Return only true or false."
        if normalized == "number":
            return "Return only a numeric value."
        if normalized == "name":
            return "Return only the exact name or entity from the context."
        if normalized == "names":
            return "Return only a semicolon-separated list of exact names from the context."
        if normalized == "date":
            return "Return only the date in YYYY-MM-DD format."
        return "Return a concise answer grounded in the context, max 280 characters."

    def _invoke_llm(self, prompt: str, telemetry_timer: Optional[Any] = None) -> str:
        if hasattr(self.llm, "stream"):
            response_chunks = []
            for chunk in self.llm.stream(prompt):
                if not response_chunks and telemetry_timer:
                    telemetry_timer.mark_token()
                
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
                response_chunks.append(text)
            return "".join(response_chunks).strip()
        else:
            response = self.llm.invoke(prompt)
            if telemetry_timer:
                telemetry_timer.mark_token()
            if hasattr(response, "content"):
                content = response.content
                if isinstance(content, list):
                    return "".join(part.get("text", "") if isinstance(part, dict) else str(part) for part in content).strip()
                return str(content).strip()
            return str(response).strip()

    def _parse_answer_by_type(self, raw: str, answer_type: str) -> Any:
        text = (raw or "").strip()
        lowered = text.lower()
        if lowered in {"null", "none", "not found", "unknown", "insufficient information"}:
            return None

        normalized = answer_type.lower()
        if normalized == "boolean":
            if lowered in {"true", "yes"}:
                return True
            if lowered in {"false", "no"}:
                return False
            return None
        if normalized == "number":
            match = re.search(r"-?\d+(?:\.\d+)?", text.replace(",", ""))
            if not match:
                return None
            number_text = match.group(0)
            return float(number_text) if "." in number_text else int(number_text)
        if normalized == "names":
            if not text:
                return None
            names = [item.strip() for item in re.split(r";|,", text) if item.strip()]
            return names or None
        if normalized == "date":
            iso_match = re.search(r"\b\d{4}-\d{2}-\d{2}\b", text)
            if iso_match:
                return iso_match.group(0)
            parsed = self._parse_non_iso_date(text)
            return parsed
        if normalized == "name":
            return text or None
        cleaned = " ".join(text.split())
        if cleaned.lower().startswith("there is no information"):
            return self._default_absent_answer("free_text")
        return cleaned or self._default_absent_answer("free_text")

    def _parse_non_iso_date(self, text: str) -> Optional[str]:
        from datetime import datetime

        for fmt in ["%B %d, %Y", "%b %d, %Y", "%d %B %Y", "%d %b %Y"]:
            try:
                return datetime.strptime(text.strip().title(), fmt).strftime("%Y-%m-%d")
            except ValueError:
                continue
        return None

    def _default_absent_answer(self, answer_type: str) -> Any:
        if answer_type.lower() == "free_text":
            return "There is no information on this question in the provided documents."
        return None

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
