"""Helpers and data models for legal ingest chunking."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import re
from typing import Dict, List, Sequence

from langchain_core.documents import Document


def approximate_token_count(text: str) -> int:
    """Approximate token count without adding a hard tokenizer dependency."""
    return max(1, int(len(re.findall(r"\S+", text)) * 1.3))


def stable_chunk_id(
    doc_id: str,
    chunk_kind: str,
    page_numbers: Sequence[int],
    block_start: int,
    block_end: int,
    text: str,
) -> str:
    normalized = re.sub(r"\s+", " ", text).strip()
    digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:20]
    page_key = "-".join(str(page) for page in page_numbers)
    return f"{doc_id}:{chunk_kind}:{page_key}:{block_start}:{block_end}:{digest}"


def build_breadcrumbs(
    metadata: Dict[str, object], section_path: Sequence[str], chunk_kind: str
) -> str:
    parts: List[str] = []
    claim_number = str(metadata.get("claim_number") or "").strip()
    if claim_number:
        parts.append(claim_number)
    case_name = str(metadata.get("case_name") or "").strip()
    if case_name:
        parts.append(case_name)
    if section_path:
        parts.append(" > ".join(section_path))
    parts.append(f"chunk={chunk_kind}")
    return " | ".join(part for part in parts if part)


def build_chunk_document(
    base_metadata: Dict[str, object],
    chunk_kind: str,
    section_path: Sequence[str],
    heading: str,
    level: int,
    page_numbers: Sequence[int],
    block_start: int,
    block_end: int,
    text: str,
) -> Document:
    breadcrumbs = build_breadcrumbs(base_metadata, section_path, chunk_kind)
    chunk_text = text.strip()
    if breadcrumbs:
        chunk_text = f"{breadcrumbs}\n\n{chunk_text}"

    page_numbers_list = sorted({int(page) for page in page_numbers})
    chunk_metadata = dict(base_metadata)
    chunk_metadata.update(
        {
            "chunk_id": stable_chunk_id(
                str(base_metadata.get("doc_id") or ""),
                chunk_kind,
                page_numbers_list,
                block_start,
                block_end,
                chunk_text,
            ),
            "chunk_kind": chunk_kind,
            "heading": heading,
            "section_path": list(section_path),
            "level": int(level),
            "page_numbers": page_numbers_list,
            "page_start": min(page_numbers_list) if page_numbers_list else 0,
            "page_end": max(page_numbers_list) if page_numbers_list else 0,
            "block_start": int(block_start),
            "block_end": int(block_end),
        }
    )
    return Document(page_content=chunk_text, metadata=chunk_metadata)


@dataclass
class SectionWindow:
    """Intermediate representation of a legal chunk candidate."""

    chunk_kind: str
    heading: str
    section_path: List[str]
    level: int
    blocks: List[Dict[str, object]]

    @property
    def text(self) -> str:
        return "\n".join(str(block["text"]) for block in self.blocks if block.get("text")).strip()

    @property
    def page_numbers(self) -> List[int]:
        return sorted({int(block["page_number"]) for block in self.blocks})

    @property
    def block_start(self) -> int:
        return int(self.blocks[0]["block_index"])

    @property
    def block_end(self) -> int:
        return int(self.blocks[-1]["block_index"])


def sanitize_metadata_for_vectorstore(metadata: Dict[str, object]) -> Dict[str, object]:
    """Convert list-rich metadata into Chroma-friendly primitive values."""
    serialized: Dict[str, object] = {}
    for key, value in metadata.items():
        if isinstance(value, list):
            serialized[key] = ", ".join(str(item) for item in value)
        elif isinstance(value, dict):
            serialized[key] = str(value)
        else:
            serialized[key] = value
    return serialized


def decode_page_numbers(metadata: Dict[str, object]) -> List[int]:
    value = metadata.get("page_numbers", [])
    if isinstance(value, list):
        return sorted({int(item) for item in value if str(item).isdigit()})
    if isinstance(value, int):
        return [value] if value > 0 else []
    if not isinstance(value, str):
        return []
    tokens = re.split(r"[^0-9]+", value)
    return sorted({int(token) for token in tokens if token})


def overlap_tail_blocks(
    blocks: Sequence[Dict[str, object]], target_tokens: int
) -> List[Dict[str, object]]:
    """Return trailing blocks approximating the requested overlap size."""
    if target_tokens <= 0:
        return []
    collected: List[Dict[str, object]] = []
    running_tokens = 0
    for block in reversed(blocks):
        collected.insert(0, dict(block))
        running_tokens += approximate_token_count(str(block.get("text") or ""))
        if running_tokens >= target_tokens:
            break
    return collected
