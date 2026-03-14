"""
Text chunkers for RAG pipeline.

This module provides abstract interfaces and concrete implementations
for splitting documents into chunks suitable for embedding and retrieval.
"""

from retrieval.chunkers.base import ChunkingResult, TextChunker
from retrieval.chunkers.legal_ingest_chunker import LegalChunkerConfig, LegalIngestChunker
from retrieval.chunkers.recursive_chunker import ChunkerConfig, RecursiveMarkdownChunker

__all__ = [
    "TextChunker",
    "ChunkingResult",
    "RecursiveMarkdownChunker",
    "ChunkerConfig",
    "LegalIngestChunker",
    "LegalChunkerConfig",
]
