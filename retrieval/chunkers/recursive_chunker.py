"""
Recursive text splitter for Markdown documents.

Uses LangChain's RecursiveCharacterTextSplitter with Markdown-aware
splitting to preserve document structure.
"""

import logging
from dataclasses import dataclass
from typing import List
from uuid import uuid4

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter, Language

from retrieval.chunkers.base import ChunkingResult, TextChunker

logger = logging.getLogger(__name__)


@dataclass
class ChunkerConfig:
    """Configuration for recursive chunking."""

    chunk_size: int = 1000
    chunk_overlap: int = 0
    min_chunk_size: int = 100
    language: Language = Language.MARKDOWN


class RecursiveMarkdownChunker(TextChunker):
    """
    Text chunker using RecursiveCharacterTextSplitter with Markdown support.

    Features:
    - Respects Markdown structure (headers, lists, code blocks)
    - Configurable chunk size and overlap
    - Filters out very short chunks
    - Assigns unique chunk IDs

    Args:
        config: ChunkerConfig with size and overlap settings
    """

    def __init__(self, config: ChunkerConfig = None):
        self.config = config or ChunkerConfig()
        self._splitter = RecursiveCharacterTextSplitter.from_language(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
            language=self.config.language,
        )

    def chunk(self, documents: List[Document]) -> ChunkingResult:
        """
        Split documents into chunks using recursive splitting.

        Args:
            documents: List of Document objects to chunk

        Returns:
            ChunkingResult with chunks and statistics
        """
        if not documents:
            return ChunkingResult(
                chunks=[],
                stats={
                    "total_docs": 0,
                    "total_chunks": 0,
                    "chunks_before_filter": 0,
                    "chunks_after_filter": 0,
                },
            )

        # Split all documents
        split_docs = self._splitter.split_documents(documents)

        # Filter and add chunk IDs
        filtered_chunks: List[Document] = []
        for chunk in split_docs:
            if len(chunk.page_content.strip()) < self.config.min_chunk_size:
                continue

            # Create unique chunk ID
            chunk_id = str(uuid4())

            # Add chunk metadata
            new_metadata = dict(chunk.metadata)
            new_metadata["chunk_id"] = chunk_id
            chunk.metadata = new_metadata

            filtered_chunks.append(chunk)

        stats = {
            "total_docs": len(documents),
            "total_chunks": len(filtered_chunks),
            "chunks_before_filter": len(split_docs),
            "chunks_after_filter": len(filtered_chunks),
            "chunk_size": self.config.chunk_size,
            "chunk_overlap": self.config.chunk_overlap,
            "min_chunk_size": self.config.min_chunk_size,
        }

        logger.info(
            f"Chunking: {len(split_docs)} -> {len(filtered_chunks)} chunks (filtered {len(split_docs) - len(filtered_chunks)} small chunks)"
        )

        return ChunkingResult(chunks=filtered_chunks, stats=stats)

    def get_chunk_ids(self, chunks: List[Document]) -> List[str]:
        """
        Extract chunk IDs from a list of chunked documents.

        Args:
            chunks: List of chunked Document objects

        Returns:
            List of chunk ID strings
        """
        return [doc.metadata.get("chunk_id", "") for doc in chunks]
