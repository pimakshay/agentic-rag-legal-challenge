"""Legal-specific chunker for ingest-backed documents."""

from __future__ import annotations

from dataclasses import dataclass
import logging
import re
from typing import Dict, Iterable, List

from langchain_core.documents import Document

from retrieval.chunkers.base import ChunkingResult, TextChunker
from retrieval.chunkers.legal_chunk_types import (
    SectionWindow,
    approximate_token_count,
    build_chunk_document,
    overlap_tail_blocks,
)

logger = logging.getLogger(__name__)


@dataclass
class LegalChunkerConfig:
    """Configuration for legal ingest chunking."""

    target_chunk_tokens: int = 350
    chunk_overlap_tokens: int = 50
    min_chunk_tokens: int = 80


class LegalIngestChunker(TextChunker):
    """Structure-first chunker for legal ingest output."""

    def __init__(self, config: LegalChunkerConfig | None = None):
        self.config = config or LegalChunkerConfig()

    def chunk(self, documents: List[Document]) -> ChunkingResult:
        chunks: List[Document] = []
        for document in documents:
            chunks.extend(self._chunk_document(document))

        stats = {
            "total_docs": len(documents),
            "total_chunks": len(chunks),
            "chunker": "legal_ingest",
            "target_chunk_tokens": self.config.target_chunk_tokens,
            "chunk_overlap_tokens": self.config.chunk_overlap_tokens,
            "min_chunk_tokens": self.config.min_chunk_tokens,
        }
        return ChunkingResult(chunks=chunks, stats=stats)

    def _chunk_document(self, document: Document) -> List[Document]:
        metadata = dict(document.metadata or {})
        blocks = list(metadata.get("blocks") or [])
        if not blocks:
            return []

        chunks: List[Document] = []

        title_page_blocks = [block for block in blocks if int(block["page_number"]) == 1]
        if title_page_blocks:
            chunks.append(
                self._window_to_chunk(
                    metadata,
                    SectionWindow(
                        chunk_kind="title_page",
                        heading="Title Page",
                        section_path=["Title Page"],
                        level=1,
                        blocks=title_page_blocks,
                    ),
                )
            )

        section_windows = self._build_section_windows(blocks, bool(metadata.get("structure_available")))
        for window in section_windows:
            chunks.extend(self._split_window(metadata, window))

        chunks.extend(self._build_page_anchor_chunks(metadata, blocks))
        return chunks

    def _build_section_windows(
        self,
        blocks: List[Dict[str, object]],
        structure_available: bool,
    ) -> List[SectionWindow]:
        if not structure_available:
            return []

        windows: List[SectionWindow] = []
        current_heading = "Document Body"
        current_level = 2
        current_path: List[str] = [current_heading]
        current_blocks: List[Dict[str, object]] = []
        heading_stack: List[tuple[int, str]] = []

        for block in blocks:
            text = str(block["text"]).strip()
            level = int(block.get("level") or 4)
            if self._is_heading_block(text, level):
                if current_blocks:
                    windows.append(
                        SectionWindow(
                            chunk_kind="section",
                            heading=current_heading,
                            section_path=list(current_path),
                            level=current_level,
                            blocks=list(current_blocks),
                        )
                    )
                    current_blocks = []

                while heading_stack and heading_stack[-1][0] >= level:
                    heading_stack.pop()
                heading_stack.append((level, text))
                current_heading = text
                current_level = level
                current_path = [entry[1] for entry in heading_stack]
                current_blocks.append(block)
            else:
                current_blocks.append(block)

        if current_blocks:
            windows.append(
                SectionWindow(
                    chunk_kind="section",
                    heading=current_heading,
                    section_path=list(current_path),
                    level=current_level,
                    blocks=list(current_blocks),
                )
            )

        return windows

    def _build_page_anchor_chunks(
        self,
        metadata: Dict[str, object],
        blocks: List[Dict[str, object]],
    ) -> List[Document]:
        by_page: Dict[int, List[Dict[str, object]]] = {}
        for block in blocks:
            by_page.setdefault(int(block["page_number"]), []).append(block)

        page_chunks: List[Document] = []
        for page_number in sorted(by_page):
            window = SectionWindow(
                chunk_kind="page_anchor",
                heading=f"Page {page_number}",
                section_path=[f"Page {page_number}"],
                level=5,
                blocks=by_page[page_number],
            )
            page_chunks.append(self._window_to_chunk(metadata, window))
        return page_chunks

    def _split_window(self, metadata: Dict[str, object], window: SectionWindow) -> List[Document]:
        if approximate_token_count(window.text) <= self.config.target_chunk_tokens:
            return [self._window_to_chunk(metadata, window)]

        chunks: List[Document] = []
        current_blocks: List[Dict[str, object]] = []
        running_tokens = 0

        for block in window.blocks:
            block_tokens = approximate_token_count(str(block["text"]))
            if current_blocks and running_tokens + block_tokens > self.config.target_chunk_tokens:
                chunks.append(
                    self._window_to_chunk(
                        metadata,
                        SectionWindow(
                            chunk_kind=window.chunk_kind,
                            heading=window.heading,
                            section_path=list(window.section_path),
                            level=window.level,
                            blocks=list(current_blocks),
                        ),
                    )
                )
                overlap_blocks = overlap_tail_blocks(current_blocks, self.config.chunk_overlap_tokens)
                current_blocks = list(overlap_blocks)
                running_tokens = sum(approximate_token_count(str(item["text"])) for item in current_blocks)

            current_blocks.append(block)
            running_tokens += block_tokens

        if current_blocks:
            chunks.append(
                self._window_to_chunk(
                    metadata,
                    SectionWindow(
                        chunk_kind=window.chunk_kind,
                        heading=window.heading,
                        section_path=list(window.section_path),
                        level=window.level,
                        blocks=list(current_blocks),
                    ),
                )
            )

        return chunks

    def _window_to_chunk(self, metadata: Dict[str, object], window: SectionWindow) -> Document:
        chunk = build_chunk_document(
            metadata,
            window.chunk_kind,
            window.section_path,
            window.heading,
            window.level,
            window.page_numbers,
            window.block_start,
            window.block_end,
            window.text,
        )

        if (
            window.chunk_kind == "section"
            and approximate_token_count(window.text) < self.config.min_chunk_tokens
            and not self._is_short_legal_section(window.heading, window.text)
        ):
            chunk.metadata["undersized"] = True
        return chunk

    def _is_heading_block(self, text: str, level: int) -> bool:
        normalized = text.strip()
        if not normalized:
            return False
        if len(normalized) <= 140 and normalized.upper() == normalized:
            return True
        if re.match(r"^(Article|Part|Chapter|Section)\b", normalized, re.IGNORECASE):
            return True
        if re.match(r"^\d+(?:\.\d+)*\s+[A-Z]", normalized):
            return True
        if re.match(r"^(UPON|AND UPON|IT IS HEREBY ORDERED THAT)", normalized):
            return True
        return level <= 2 and len(normalized) <= 180

    def _is_short_legal_section(self, heading: str, text: str) -> bool:
        normalized = f"{heading}\n{text}".lower()
        return any(
            marker in normalized
            for marker in [
                "ordered",
                "order",
                "disposition",
                "declaration",
                "relief sought",
            ]
        )
