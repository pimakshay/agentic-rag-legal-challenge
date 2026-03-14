"""
Abstract base class for text chunkers.

Provides a consistent interface for splitting documents into chunks,
enabling easy swapping of different chunking strategies.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Any

from langchain_core.documents import Document


@dataclass
class ChunkingResult:
    """
    Result container for text chunking operations.
    
    Attributes:
        chunks: List of chunked Document objects
        stats: Statistics about the chunking operation
    """
    chunks: List[Document]
    stats: Dict[str, Any] = field(default_factory=dict)


class TextChunker(ABC):
    """
    Abstract base class for text chunkers.
    
    Text chunkers are responsible for:
    - Splitting documents into smaller chunks
    - Preserving semantic boundaries (sentences, paragraphs, sections)
    - Adding chunk-level metadata (chunk_id, position)
    - Filtering out very short chunks
    
    Implementations can use different strategies:
    - RecursiveCharacterTextSplitter (hierarchical splitting)
    - SentenceTransformers (semantic splitting)
    - Custom strategies for structured documents
    
    Example:
        chunker = LegalIngestChunker(LegalChunkerConfig())
        result = chunker.chunk(documents)
        for chunk in result.chunks:
            print(f"Chunk {chunk.metadata['chunk_id']}: {len(chunk.page_content)} chars")
    """
    
    @abstractmethod
    def chunk(self, documents: List[Document]) -> ChunkingResult:
        """
        Split documents into chunks.
        
        Args:
            documents: List of Document objects to chunk
            
        Returns:
            ChunkingResult containing chunks and statistics
        """
        pass
    
    def __call__(self, documents: List[Document]) -> ChunkingResult:
        """Allow using chunker as a callable."""
        return self.chunk(documents)

