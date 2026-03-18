"""Ingest-backed loader for the legal challenge corpus."""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from langchain_core.documents import Document

logger = logging.getLogger(__name__)


class IngestedCorpusLoader:
    """Load ingested legal documents from metadata/content/structure JSON files."""

    CASE_NUMBER_PATTERN = re.compile(r"\b[A-Z]{2,4}\s+\d{3}/\d{4}\b")

    def __init__(
        self,
        ingest_root: str | Path | None = None,
        docs_root: str | Path | None = None,
    ) -> None:
        self.ingest_root = Path(ingest_root) if ingest_root else None
        self.docs_root = Path(docs_root) if docs_root else None

    def load_corpus(
        self,
        ingest_root: str | Path | None = None,
        docs_root: str | Path | None = None,
    ) -> List[Document]:
        root = Path(ingest_root) if ingest_root else self.ingest_root
        if root is None:
            raise ValueError("ingest_root must be provided")

        pdf_root = Path(docs_root) if docs_root else self.docs_root
        documents: List[Document] = []
        for doc_dir in sorted(path for path in root.iterdir() if path.is_dir()):
            loaded = self._load_single_document(doc_dir, pdf_root)
            if loaded is not None:
                documents.append(loaded)

        logger.info("Loaded %s ingested source documents from %s", len(documents), root)
        return documents

    def _load_single_document(self, doc_dir: Path, docs_root: Optional[Path]) -> Optional[Document]:
        doc_id = doc_dir.name
        txt_dir = doc_dir / "txt"
        content_path = txt_dir / f"{doc_id}_content_list.json"
        metadata_path = txt_dir / f"{doc_id}_metadata.json"
        structure_path = txt_dir / f"{doc_id}_structure.json"

        if not content_path.exists():
            logger.warning("Skipping %s because content list is missing", doc_id)
            return None

        content_items = self._read_json(content_path, default=[])
        metadata = self._read_json(metadata_path, default={})
        structured_items = self._read_structure(structure_path)
        structure_available = bool(structured_items)

        blocks = self._build_blocks(content_items, structured_items)
        if not blocks:
            logger.warning("Skipping %s because no text blocks were found", doc_id)
            return None

        source_path = str((docs_root / f"{doc_id}.pdf").resolve()) if docs_root else ""
        doc_metadata = dict(metadata or {})
        doc_metadata.update(
            {
                "doc_id": doc_id,
                "source": source_path or f"{doc_id}.pdf",
                "source_path": source_path,
                "doc_type": self._coerce_doc_type(metadata),
                "structure_available": structure_available,
                "block_count": len(blocks),
                "page_count": int(metadata.get("page_count") or max(block["page_number"] for block in blocks)),
                "title_page_number": int(metadata.get("title_page_number") or 1),
                "last_page_number": int(metadata.get("last_page_number") or max(block["page_number"] for block in blocks)),
                "title": self._derive_title(metadata, doc_id),
                "blocks": blocks,
            }
        )
        doc_metadata.setdefault("alias_keys", [])
        doc_metadata.setdefault("article_index", {})
        doc_metadata.setdefault("section_heading_index", {})
        doc_metadata.setdefault("defendants", [])
        doc_metadata.setdefault("claimant_counsel", [])
        doc_metadata.setdefault("defendant_counsel", [])
        doc_metadata.setdefault("amending_laws", [])

        page_content = "\n".join(block["text"] for block in blocks if block["text"])
        return Document(page_content=page_content, metadata=doc_metadata, id=doc_id)

    def _read_json(self, path: Path, default: Any) -> Any:
        if not path.exists():
            return default
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    def _read_structure(self, path: Path) -> Dict[int, Dict[str, Any]]:
        if not path.exists():
            return {}
        raw = self._read_json(path, default={})
        normalized: Dict[int, Dict[str, Any]] = {}
        for key, value in raw.items():
            try:
                normalized[int(key)] = value
            except (TypeError, ValueError):
                continue
        return normalized

    def _build_blocks(
        self,
        content_items: List[Dict[str, Any]],
        structured_items: Dict[int, Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        blocks: List[Dict[str, Any]] = []
        for index, item in enumerate(content_items):
            text = (item.get("text") or item.get("table_body") or "").strip()
            if not text:
                continue
            structure = structured_items.get(index, {})
            page_idx = structure.get("page_idx", item.get("page_idx", 0))
            blocks.append(
                {
                    "block_index": index,
                    "page_idx": int(page_idx),
                    "page_number": int(page_idx) + 1,
                    "text": text,
                    "type": structure.get("type", item.get("type", "text")),
                    "level": int(structure.get("level") or self._synthetic_level(item, text)),
                    "bbox": item.get("bbox") or [],
                    "synthetic_structure": index not in structured_items,
                }
            )
        return blocks

    def _synthetic_level(self, item: Dict[str, Any], text: str) -> int:
        hinted = item.get("text_level")
        if isinstance(hinted, int) and 1 <= hinted <= 5:
            return hinted
        normalized = text.strip()
        if len(normalized) <= 80 and normalized.upper() == normalized:
            return 1
        if re.match(r"^(Article|Part|Chapter|Section)\b", normalized, re.IGNORECASE):
            return 2
        if re.match(r"^\(?[0-9]+\)?[.)]?\s+", normalized):
            return 3
        return 4

    def _coerce_doc_type(self, metadata: Dict[str, Any]) -> str:
        value = str(metadata.get("doc_type") or "").strip().lower()
        if value in {"case", "law"}:
            return value
        if metadata.get("claim_number") or metadata.get("case_name"):
            return "case"
        if metadata.get("official_title") or metadata.get("alias_keys"):
            return "law"
        return "unknown"

    def _derive_title(self, metadata: Dict[str, Any], doc_id: str) -> str:
        for key in ("official_title", "short_title", "case_name", "claim_number"):
            value = str(metadata.get(key) or "").strip()
            if value:
                return value
        return doc_id
