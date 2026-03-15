"""Legal-challenge hybrid RAG pipeline built on ingest outputs."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from langchain_core.documents import Document

from arlc import RetrievalRef, normalize_retrieved_pages
from retrieval.chunkers import LegalChunkerConfig, LegalIngestChunker
from retrieval.chunkers.legal_chunk_types import decode_page_numbers, sanitize_metadata_for_vectorstore
from retrieval.legal_question_router import LegalQuestionRouter, RoutePlan
from retrieval.loaders import IngestedCorpusLoader
from retrieval.retrievers.base import BaseRAGRetriever
from retrieval.retrievers.sparse_retriever import BM25SparseRetriever
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


class LegalHybridRAGPipeline:
    """Challenge-specific hybrid RAG pipeline backed by ingestion output."""

    def __init__(
        self,
        llm: Any = None,
        embedding_model: Any = None,
        ingest_root: str = "ingestion/docs_corpus_ingest_result",
        docs_root: str = "docs_corpus",
        chroma_persist_dir: str = "tmp/legal_hybrid_chroma",
        collection_name: str = "legal_challenge",
        top_k_docs: int = 6,
        dense_candidate_k: int = 12,
        sparse_candidate_k: int = 20,
        dense_weight: float = 0.4,
        sparse_weight: float = 0.6,
        use_persistent_db: bool = False,
        enable_reranking: bool = True,
        reranker: Optional[BaseReranker] = None,
        loader: Optional[IngestedCorpusLoader] = None,
        chunker: Optional[LegalIngestChunker] = None,
        router: Optional[LegalQuestionRouter] = None,
    ) -> None:
        self.llm = llm
        self.embedding_model = embedding_model
        self.ingest_root = ingest_root
        self.docs_root = docs_root
        self.chroma_persist_dir = chroma_persist_dir
        self.collection_name = collection_name
        self.top_k_docs = top_k_docs
        self.dense_candidate_k = dense_candidate_k
        self.sparse_candidate_k = sparse_candidate_k
        self.dense_weight = dense_weight
        self.sparse_weight = sparse_weight
        self.use_persistent_db = use_persistent_db

        self.loader = loader or IngestedCorpusLoader(ingest_root=ingest_root, docs_root=docs_root)
        self.chunker = chunker or LegalIngestChunker(LegalChunkerConfig())
        self.router = router or LegalQuestionRouter()
        self.reranker = reranker or (MiniLMReranker() if enable_reranking else None)

        self.source_documents: List[Document] = []
        self.raw_chunks: List[Document] = []
        self.doc_search: Optional[Any] = None
        self._all_doc_ids: set[str] = set()
        self._claim_number_index: Dict[str, set[str]] = {}
        self._law_lookup: Dict[str, set[str]] = {}
        self._doc_metadata: Dict[str, Dict[str, Any]] = {}

    def load_corpus(self) -> List[Document]:
        self.source_documents = self.loader.load_corpus(self.ingest_root, self.docs_root)
        self._build_metadata_indexes(self.source_documents)
        return self.source_documents

    def _chroma_persist_dir_exists(self) -> bool:
        """Return True if Chroma has already persisted a DB in chroma_persist_dir."""
        p = Path(self.chroma_persist_dir)
        if not p.is_dir():
            return False
        # Chroma persists chroma.sqlite3 in the given directory
        return (p / "chroma.sqlite3").is_file()

    def build_indexes(self) -> Any:
        if self.embedding_model is None:
            raise ValueError("embedding_model must be provided")

        if not self.source_documents:
            self.load_corpus()

        chunk_result = self.chunker.chunk(self.source_documents)
        self.raw_chunks = [chunk for chunk in chunk_result.chunks if chunk.page_content.strip()]

        from langchain_community.vectorstores import Chroma

        os.makedirs(self.chroma_persist_dir, exist_ok=True)

        # If persistent DB is enabled and a Chroma DB already exists, load it instead of re-embedding.
        if self.use_persistent_db and self._chroma_persist_dir_exists():
            try:
                self.doc_search = Chroma(
                    collection_name=self.collection_name,
                    embedding_function=self.embedding_model,
                    persist_directory=self.chroma_persist_dir,
                )
                # Verify the collection has documents (e.g. not empty or wrong collection)
                count = self.doc_search._collection.count()
                if count > 0:
                    logger.info(
                        "Loaded existing Chroma index from %s (%s chunks); skipped re-embedding",
                        self.chroma_persist_dir,
                        count,
                    )
                    return self.doc_search
            except Exception as e:
                logger.warning("Failed to load existing Chroma DB, rebuilding: %s", e)

        serializable_chunks = [self._serialize_chunk(chunk) for chunk in self.raw_chunks]
        ids = [str(chunk.metadata["chunk_id"]) for chunk in self.raw_chunks]

        logger.info("Building Chroma index with %s legal chunks", len(serializable_chunks))
        self.doc_search = Chroma.from_documents(
            documents=serializable_chunks,
            embedding=self.embedding_model,
            collection_name=self.collection_name,
            persist_directory=self.chroma_persist_dir,
            ids=ids,
        )
        if self.use_persistent_db:
            self.doc_search.persist()
        return self.doc_search

    def retrieve(
        self,
        question_text: str,
        answer_type: str = "free_text",
        route: Optional[RoutePlan] = None,
    ) -> Tuple[List[Document], RoutePlan]:
        if self.doc_search is None:
            self.build_indexes()

        active_route = route or self.router.route(question_text, answer_type)
        candidate_doc_ids = self._resolve_candidate_doc_ids(active_route)
        sparse_pool = self._filter_chunks(self.raw_chunks, candidate_doc_ids)

        dense_docs = self._dense_search(question_text, candidate_doc_ids, active_route)
        sparse_docs = self._sparse_search(question_text, sparse_pool, active_route)
        fused_docs = self._fuse_results(dense_docs, sparse_docs, self.top_k_docs * 3)
        reranked_docs = self._rerank(question_text, fused_docs)
        return reranked_docs[: self.top_k_docs], active_route

    def answer_question(self, question_item: Dict[str, Any]) -> AnswerResult:
        question_text = question_item["question"]
        answer_type = question_item.get("answer_type", "free_text")
        route = self.router.route(question_text, answer_type)

        if route.comparison_mode and len(route.case_ids) > 1:
            supporting_docs = self._retrieve_for_comparison(question_text, answer_type, route)
        else:
            supporting_docs, route = self.retrieve(question_text, answer_type=answer_type, route=route)

        if not supporting_docs:
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
        raw_response = self._invoke_llm(prompt)
        answer = self._parse_answer_by_type(raw_response, answer_type)

        if answer is None and answer_type == "free_text":
            answer = self._default_absent_answer(answer_type)
            retrieval_refs = []
        else:
            retrieval_refs = normalize_retrieved_pages(
                RetrievalRef(doc_id=str(doc.metadata["doc_id"]), page_numbers=self._chunk_page_numbers(doc))
                for doc in supporting_docs
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

    def _build_metadata_indexes(self, source_documents: Sequence[Document]) -> None:
        self._all_doc_ids = set()
        self._claim_number_index = {}
        self._law_lookup = {}
        self._doc_metadata = {}

        for document in source_documents:
            metadata = dict(document.metadata or {})
            doc_id = str(metadata.get("doc_id") or "")
            if not doc_id:
                continue
            self._all_doc_ids.add(doc_id)
            self._doc_metadata[doc_id] = metadata

            claim_number = self._normalize_key(str(metadata.get("claim_number") or ""))
            if claim_number:
                self._claim_number_index.setdefault(claim_number, set()).add(doc_id)

            for key in self._candidate_law_terms(metadata, document.page_content):
                self._law_lookup.setdefault(self._normalize_key(key), set()).add(doc_id)

    def _candidate_law_terms(self, metadata: Dict[str, Any], page_content: str) -> List[str]:
        terms = []
        for value in [
            metadata.get("case_name") or "",
            metadata.get("title") or "",
            metadata.get("neutral_citation") or "",
        ]:
            if isinstance(value, str) and value.strip():
                terms.append(value.strip())

        first_lines = "\n".join(page_content.splitlines()[:20])
        law_matches = re.findall(r"([A-Z][A-Za-z&,\- ]+? Law(?:\s+\d{4})?)", first_lines)
        terms.extend(match.strip() for match in law_matches if match.strip())
        return terms

    def _resolve_candidate_doc_ids(self, route: RoutePlan) -> set[str]:
        candidate_doc_ids: set[str] = set()

        for case_id in route.case_ids:
            candidate_doc_ids.update(self._claim_number_index.get(self._normalize_key(case_id), set()))

        if not candidate_doc_ids and (route.law_names or route.law_numbers):
            for term in route.law_names + route.law_numbers:
                normalized = self._normalize_key(term)
                for key, doc_ids in self._law_lookup.items():
                    if normalized in key or key in normalized:
                        candidate_doc_ids.update(doc_ids)

        return candidate_doc_ids or set(self._all_doc_ids)

    def _filter_chunks(self, chunks: Sequence[Document], candidate_doc_ids: set[str]) -> List[Document]:
        return [
            chunk
            for chunk in chunks
            if str(chunk.metadata.get("doc_id") or "") in candidate_doc_ids
        ]

    def _dense_search(
        self,
        question_text: str,
        candidate_doc_ids: set[str],
        route: RoutePlan,
    ) -> List[Document]:
        if self.doc_search is None:
            return []

        search_kwargs: Dict[str, Any] = {"query": question_text, "k": self.dense_candidate_k}
        if candidate_doc_ids and candidate_doc_ids != self._all_doc_ids:
            search_kwargs["filter"] = {"doc_id": {"$in": sorted(candidate_doc_ids)}}

        try:
            results_with_scores = self.doc_search.similarity_search_with_relevance_scores(**search_kwargs)
            results = [self._decode_dense_document(doc) for doc, _ in results_with_scores]
        except Exception as exc:
            logger.warning("Dense retrieval with filter failed, falling back to unfiltered search: %s", exc)
            results = [self._decode_dense_document(doc) for doc in self.doc_search.similarity_search(question_text, k=self.dense_candidate_k * 3)]

        filtered = [doc for doc in results if str(doc.metadata.get("doc_id") or "") in candidate_doc_ids]
        dense_docs = filtered or results[: self.dense_candidate_k]
        return self._strictly_filter_route(self._apply_route_bias(dense_docs, route), route)

    def _sparse_search(
        self,
        question_text: str,
        sparse_pool: Sequence[Document],
        route: RoutePlan,
    ) -> List[Document]:
        if not sparse_pool:
            return []

        sparse_query = self._build_sparse_query(question_text, route)
        retriever = BM25SparseRetriever(documents=list(sparse_pool), default_k=self.sparse_candidate_k)
        result = retriever.retrieve(sparse_query, k=self.sparse_candidate_k)
        return self._strictly_filter_route(self._apply_route_bias(result.documents, route), route)

    def _build_sparse_query(self, question_text: str, route: RoutePlan) -> str:
        tokens = [question_text]
        tokens.extend(route.case_ids)
        tokens.extend(route.article_refs)
        tokens.extend(route.law_names)
        tokens.extend(route.law_numbers)
        return " ".join(token for token in tokens if token).strip()
    
    def _strictly_filter_route(self, docs: Sequence[Document], route: RoutePlan) -> List[Document]:
        def strictly_filter(doc: Document) -> bool:
            metadata = doc.metadata or {}
            chunk_kind = str(metadata.get("chunk_kind") or "")
            page_numbers = self._chunk_page_numbers(doc)
            if route.target_pages and (not any(page in page_numbers for page in route.target_pages)):
                return False
            if route.prefer_title_page and chunk_kind != "title_page":
                return False
            if route.prefer_last_page:
                doc_pages = int(self._doc_metadata.get(str(metadata.get("doc_id") or ""), {}).get("page_count") or 0)
                if not (doc_pages and page_numbers and max(page_numbers) == doc_pages):
                    return False
            return True
        return list(filter(strictly_filter, docs))

    def _apply_route_bias(self, docs: Sequence[Document], route: RoutePlan) -> List[Document]:
        def score(doc: Document) -> Tuple[int, int, int]:
            metadata = doc.metadata or {}
            chunk_kind = str(metadata.get("chunk_kind") or "")
            page_numbers = self._chunk_page_numbers(doc)
            preferred_kind = 1 if chunk_kind in route.preferred_chunk_kinds else 0
            target_page_hit = 0
            if route.target_pages and any(page in page_numbers for page in route.target_pages):
                target_page_hit = 1
            if route.prefer_title_page and chunk_kind == "title_page":
                target_page_hit = 2
            if route.prefer_last_page:
                doc_pages = int(self._doc_metadata.get(str(metadata.get("doc_id") or ""), {}).get("page_count") or 0)
                if doc_pages and page_numbers and max(page_numbers) == doc_pages:
                    target_page_hit = 2
            return (preferred_kind, target_page_hit, -int(metadata.get("page_start") or 0))

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
                scores[key] = scores.get(key, 0.0) + (weight / float(rank + 1))

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
                article_refs=list(route.article_refs),
                law_names=list(route.law_names),
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

    def _invoke_llm(self, prompt: str) -> str:
        response = self.llm.invoke(prompt)
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

    def _normalize_key(self, value: str) -> str:
        return re.sub(r"\s+", " ", value.strip().lower())


HybridRAGPipeline = LegalHybridRAGPipeline
