"""Repo-local retrieval stack for the legal challenge."""

from retrieval.chunkers import (
    ChunkerConfig,
    ChunkingResult,
    LegalChunkerConfig,
    LegalIngestChunker,
    RecursiveMarkdownChunker,
    TextChunker,
)
from retrieval.hybrid_rag_pipeline import AnswerResult, HybridRAGPipeline, LegalHybridRAGPipeline, NormalizedQuery
from retrieval.legal_question_router import LegalQuestionRouter, RoutePlan
from retrieval.loaders import IngestedCorpusLoader
from retrieval.retrievers import BaseRAGRetriever, BM25SparseRetriever, RetrievalResult
from retrieval.utils import BGEReranker, BaseReranker, MiniLMReranker, NoOpReranker, RerankResult

__all__ = [
    "AnswerResult",
    "HybridRAGPipeline",
    "LegalHybridRAGPipeline",
    "NormalizedQuery",
    "IngestedCorpusLoader",
    "TextChunker",
    "ChunkingResult",
    "RecursiveMarkdownChunker",
    "ChunkerConfig",
    "LegalIngestChunker",
    "LegalChunkerConfig",
    "LegalQuestionRouter",
    "RoutePlan",
    "BaseRAGRetriever",
    "RetrievalResult",
    "BM25SparseRetriever",
    "BaseReranker",
    "RerankResult",
    "NoOpReranker",
    "MiniLMReranker",
    "BGEReranker",
]
