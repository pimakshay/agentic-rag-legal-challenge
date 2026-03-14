"""
Retrievers for RAG pipeline.

This module provides abstract interfaces and concrete implementations
for document retrieval (sparse/BM25). Dense and hybrid retrieval are
handled inside LegalHybridRAGPipeline.
"""

from retrieval.retrievers.base import BaseRAGRetriever, RetrievalResult
from retrieval.retrievers.sparse_retriever import BM25SparseRetriever

__all__ = [
    "BaseRAGRetriever",
    "RetrievalResult",
    "BM25SparseRetriever",
]
