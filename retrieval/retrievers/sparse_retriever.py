"""
Sparse (BM25) retriever for keyword-based document retrieval.

Uses BM25 algorithm for term-frequency based retrieval.
"""

import logging
from typing import Any, List, Optional

from langchain_core.documents import Document

from retrieval.retrievers.base import BaseRAGRetriever, RetrievalResult

logger = logging.getLogger(__name__)


class BM25SparseRetriever(BaseRAGRetriever):
    """
    Sparse retriever using BM25 algorithm.

    Uses term frequency and inverse document frequency
    for keyword-based document retrieval.

    Args:
        documents: List of Document objects to index
        default_k: Default number of documents to retrieve
    """

    def __init__(
        self,
        documents: Optional[List[Document]] = None,
        default_k: int = 5,
    ):
        self.default_k = default_k
        self._documents = documents or []
        self._retriever: Optional[Any] = None

        if documents:
            self._build_index(documents)

    def _build_index(self, documents: List[Document]) -> None:
        """
        Build the BM25 index from documents.

        Args:
            documents: List of Document objects to index
        """
        if not documents:
            logger.warning("No documents provided for BM25 index")
            self._retriever = None
            return

        from langchain_community.retrievers import BM25Retriever

        self._documents = documents
        self._retriever = BM25Retriever.from_documents(documents=documents)
        self._retriever.k = self.default_k
        logger.info(f"Built BM25 index with {len(documents)} documents")

    def set_documents(self, documents: List[Document]) -> None:
        """
        Set or update the documents for the BM25 index.

        Args:
            documents: List of Document objects to index
        """
        self._build_index(documents)

    def retrieve(self, query: str, k: Optional[int] = None) -> RetrievalResult:
        """
        Retrieve documents using BM25 search.

        Args:
            query: The search query (keywords)
            k: Number of documents to retrieve (uses default if not specified)

        Returns:
            RetrievalResult with documents (no scores from BM25Retriever)
        """
        k = k or self.default_k

        if not self._retriever:
            logger.warning("BM25 retriever not initialized, returning empty results")
            return RetrievalResult(
                documents=[],
                metadata={"retriever_type": "sparse", "error": "not_initialized"},
            )

        # Temporarily set k for this retrieval
        original_k = self._retriever.k
        self._retriever.k = k

        try:
            # Use invoke() (LangChain LCEL); get_relevant_documents is deprecated
            documents = self._retriever.invoke(query)
        finally:
            self._retriever.k = original_k

        return RetrievalResult(
            documents=documents,
            scores=None,  # BM25Retriever doesn't expose scores
            metadata={
                "retriever_type": "sparse",
                "k": k,
                "query": query,
            },
        )

    def get_langchain_retriever(self) -> Optional[Any]:
        """
        Get the underlying LangChain BM25Retriever.

        Returns:
            BM25Retriever instance or None if not initialized
        """
        return self._retriever

    @property
    def is_initialized(self) -> bool:
        """Check if the BM25 index has been built."""
        return self._retriever is not None
