"""
Rerankers for document retrieval.

This module provides cross-encoder rerankers that rescore retrieved documents
for improved relevance ranking before LLM generation.

Supported models:
1. ms-marco-MiniLM-L-6-v2 (baseline, fast)
2. bge-reranker-v2-m3 (higher quality, future)
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple

from langchain_core.documents import Document

logger = logging.getLogger(__name__)


@dataclass
class RerankResult:
    """
    Result container for reranking operations.
    
    Attributes:
        documents: List of reranked Document objects (sorted by relevance)
        scores: Relevance scores for each document
        metadata: Additional reranking metadata (model, timings, etc.)
    """
    documents: List[Document]
    scores: List[float]
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseReranker(ABC):
    """
    Abstract base class for document rerankers.
    
    Rerankers use cross-encoder models to score query-document pairs
    for more accurate relevance ranking than bi-encoder retrieval alone.
    
    Typical flow:
        1. Retrieve 2*k candidates using hybrid retrieval
        2. Rerank candidates using cross-encoder
        3. Return top k documents for LLM generation
    
    Example:
        reranker = MiniLMReranker()
        result = reranker.rerank(query, candidates, top_k=5)
        for doc, score in zip(result.documents, result.scores):
            print(f"Score {score:.3f}: {doc.page_content[:100]}")
    """
    
    @abstractmethod
    def rerank(
        self,
        query: str,
        documents: List[Document],
        top_k: int = 5,
    ) -> RerankResult:
        """
        Rerank documents by relevance to the query.
        
        Args:
            query: The search query
            documents: Candidate documents to rerank
            top_k: Number of top documents to return
            
        Returns:
            RerankResult with sorted documents and scores
        """
        pass
    
    def __call__(
        self,
        query: str,
        documents: List[Document],
        top_k: int = 5,
    ) -> RerankResult:
        """Allow using reranker as a callable."""
        return self.rerank(query, documents, top_k)


class NoOpReranker(BaseReranker):
    """
    A reranker that does nothing - returns documents unchanged.
    
    Useful as a default when reranking is disabled.
    """
    
    def rerank(
        self,
        query: str,
        documents: List[Document],
        top_k: int = 5,
    ) -> RerankResult:
        """Return top_k documents without reranking."""
        result_docs = documents[:top_k]
        return RerankResult(
            documents=result_docs,
            scores=[1.0 - (i * 0.01) for i in range(len(result_docs))],  # Fake descending scores
            metadata={
                "reranker": "noop",
                "input_docs": len(documents),
                "output_docs": len(result_docs),
            },
        )


class MiniLMReranker(BaseReranker):
    """
    Cross-encoder reranker using ms-marco-MiniLM-L-6-v2.
    
    This model is trained on MS MARCO passage ranking and provides
    a good balance of speed and accuracy for document reranking.
    
    Model: cross-encoder/ms-marco-MiniLM-L-6-v2
    - 22M parameters
    - ~100 docs/sec on CPU
    - Good for real-time reranking
    
    Args:
        model_name: HuggingFace model name (default: ms-marco-MiniLM-L-6-v2)
        device: Device to run model on ('cpu', 'cuda', or None for auto)
        batch_size: Batch size for scoring (higher = faster but more memory)
    """
    
    DEFAULT_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    
    def __init__(
        self,
        model_name: str = None,
        device: Optional[str] = None,
        batch_size: int = 32,
    ):
        self.model_name = model_name or self.DEFAULT_MODEL
        self.device = device
        self.batch_size = batch_size
        self._model = None
    
    def _ensure_model(self):
        """Lazily load the cross-encoder model."""
        if self._model is None:
            try:
                from sentence_transformers import CrossEncoder
                
                logger.info(f"Loading cross-encoder model: {self.model_name}")
                self._model = CrossEncoder(
                    self.model_name,
                    device=self.device,
                )
                logger.info(f"Cross-encoder model loaded successfully")
            except ImportError:
                raise ImportError(
                    "sentence-transformers is required for MiniLMReranker. "
                    "Install with: pip install sentence-transformers"
                )
    
    def rerank(
        self,
        query: str,
        documents: List[Document],
        top_k: int = 5,
    ) -> RerankResult:
        """
        Rerank documents using cross-encoder scoring.
        
        Args:
            query: The search query
            documents: Candidate documents to rerank
            top_k: Number of top documents to return
            
        Returns:
            RerankResult with documents sorted by relevance score
        """
        if not documents:
            return RerankResult(
                documents=[],
                scores=[],
                metadata={"reranker": self.model_name, "input_docs": 0, "output_docs": 0},
            )
        
        # Ensure model is loaded
        self._ensure_model()
        
        # Create query-document pairs for scoring
        pairs = [(query, doc.page_content) for doc in documents]
        
        # Score all pairs
        try:
            scores = self._model.predict(
                pairs,
                batch_size=self.batch_size,
                show_progress_bar=False,
            )
        except Exception as e:
            logger.error(f"Reranking failed: {e}")
            # Fallback: return original order
            return RerankResult(
                documents=documents[:top_k],
                scores=[0.0] * min(top_k, len(documents)),
                metadata={
                    "reranker": self.model_name,
                    "error": str(e),
                    "input_docs": len(documents),
                    "output_docs": min(top_k, len(documents)),
                },
            )
        
        # Pair documents with scores and sort by score descending
        doc_scores: List[Tuple[Document, float]] = list(zip(documents, scores))
        doc_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Take top_k
        top_docs = [doc for doc, _ in doc_scores[:top_k]]
        top_scores = [float(score) for _, score in doc_scores[:top_k]]
        
        return RerankResult(
            documents=top_docs,
            scores=top_scores,
            metadata={
                "reranker": self.model_name,
                "input_docs": len(documents),
                "output_docs": len(top_docs),
                "score_range": {
                    "min": float(min(scores)) if scores.size > 0 else 0,
                    "max": float(max(scores)) if scores.size > 0 else 0,
                },
            },
        )


class BGEReranker(BaseReranker):
    """
    Cross-encoder reranker using BGE-reranker-v2-m3.
    
    Higher quality than MiniLM but slower. Good for offline batch processing
    or when accuracy is more important than latency.
    
    Model: BAAI/bge-reranker-v2-m3
    - Multilingual support
    - Higher accuracy on academic benchmarks
    
    Args:
        model_name: HuggingFace model name
        device: Device to run model on ('cpu', 'cuda', or None for auto)
        batch_size: Batch size for scoring
    """
    
    DEFAULT_MODEL = "BAAI/bge-reranker-v2-m3"
    
    def __init__(
        self,
        model_name: str = None,
        device: Optional[str] = None,
        batch_size: int = 16,
    ):
        self.model_name = model_name or self.DEFAULT_MODEL
        self.device = device
        self.batch_size = batch_size
        self._model = None
    
    def _ensure_model(self):
        """Lazily load the cross-encoder model."""
        if self._model is None:
            try:
                from sentence_transformers import CrossEncoder
                
                logger.info(f"Loading cross-encoder model: {self.model_name}")
                self._model = CrossEncoder(
                    self.model_name,
                    device=self.device,
                )
                logger.info(f"Cross-encoder model loaded successfully")
            except ImportError:
                raise ImportError(
                    "sentence-transformers is required for BGEReranker. "
                    "Install with: pip install sentence-transformers"
                )
    
    def rerank(
        self,
        query: str,
        documents: List[Document],
        top_k: int = 5,
    ) -> RerankResult:
        """
        Rerank documents using BGE cross-encoder.
        
        Args:
            query: The search query
            documents: Candidate documents to rerank
            top_k: Number of top documents to return
            
        Returns:
            RerankResult with documents sorted by relevance score
        """
        if not documents:
            return RerankResult(
                documents=[],
                scores=[],
                metadata={"reranker": self.model_name, "input_docs": 0, "output_docs": 0},
            )
        
        self._ensure_model()
        
        pairs = [(query, doc.page_content) for doc in documents]
        
        try:
            scores = self._model.predict(
                pairs,
                batch_size=self.batch_size,
                show_progress_bar=False,
            )
        except Exception as e:
            logger.error(f"Reranking failed: {e}")
            return RerankResult(
                documents=documents[:top_k],
                scores=[0.0] * min(top_k, len(documents)),
                metadata={
                    "reranker": self.model_name,
                    "error": str(e),
                    "input_docs": len(documents),
                    "output_docs": min(top_k, len(documents)),
                },
            )
        
        doc_scores: List[Tuple[Document, float]] = list(zip(documents, scores))
        doc_scores.sort(key=lambda x: x[1], reverse=True)
        
        top_docs = [doc for doc, _ in doc_scores[:top_k]]
        top_scores = [float(score) for _, score in doc_scores[:top_k]]
        
        return RerankResult(
            documents=top_docs,
            scores=top_scores,
            metadata={
                "reranker": self.model_name,
                "input_docs": len(documents),
                "output_docs": len(top_docs),
                "score_range": {
                    "min": float(min(scores)) if scores.size > 0 else 0,
                    "max": float(max(scores)) if scores.size > 0 else 0,
                },
            },
        )
