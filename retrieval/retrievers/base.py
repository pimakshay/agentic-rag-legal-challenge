"""
Abstract base class for RAG retrievers.

Provides a consistent interface for document retrieval,
enabling easy swapping of different retrieval strategies.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import hashlib
from typing import Dict, List, Any, Optional

from langchain_core.documents import Document


@dataclass
class RetrievalResult:
    """
    Result container for retrieval operations.

    Attributes:
        documents: List of retrieved Document objects
        scores: Optional list of relevance scores
        metadata: Additional retrieval metadata (timings, metrics)
    """

    documents: List[Document]
    scores: Optional[List[float]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseRAGRetriever(ABC):
    """
    Abstract base class for RAG retrievers.

    Retrievers are responsible for:
    - Finding relevant documents given a query
    - Returning ranked results with optional scores
    - Providing retrieval metadata for debugging

    Implementations can use different strategies:
    - Dense retrieval (vector similarity)
    - Sparse retrieval (BM25, TF-IDF)
    - Hybrid retrieval (combination of dense and sparse)

    Example:
        retriever = BM25SparseRetriever(documents=docs, default_k=5)
        result = retriever.retrieve("What is the dosage?")
        for doc, score in zip(result.documents, result.scores):
            print(f"Score {score}: {doc.page_content[:100]}")
    """

    @abstractmethod
    def retrieve(self, query: str, k: int = 5) -> RetrievalResult:
        """
        Retrieve relevant documents for the given query.

        Args:
            query: The search query
            k: Number of documents to retrieve

        Returns:
            RetrievalResult containing documents and scores
        """
        pass

    def __call__(self, query: str, k: int = 5) -> RetrievalResult:
        """Allow using retriever as a callable."""
        return self.retrieve(query, k)

    @staticmethod
    def fusion_key(doc: Document) -> str:
        """
        Build a stable key for deduplicating documents.

        Uses chunk_id if available, otherwise falls back to
        source/page/content hash.
        """
        metadata = doc.metadata or {}
        if "chunk_id" in metadata:
            return str(metadata["chunk_id"])
        source = metadata.get("doc_id") or metadata.get("source") or metadata.get("file_path") or ""
        page = metadata.get("page", "")
        digest = hashlib.sha256(doc.page_content.encode("utf-8")).hexdigest()[:16]
        return f"{source}::{page}::{digest}"
