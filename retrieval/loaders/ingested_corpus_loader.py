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
        """Load the full ingest corpus as one LangChain document per source file."""
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
        first_page_text = "\n".join(block["text"] for block in blocks if block["page_number"] == 1)
        claim_number = (
            metadata.get("claim_number") or self._extract_claim_number(first_page_text)
        ).strip()
        case_name = (metadata.get("case_name") or "").strip()

        doc_metadata = {
            "doc_id": doc_id,
            "source": source_path or f"{doc_id}.pdf",
            "source_path": source_path,
            "claim_number": claim_number,
            "case_name": case_name,
            "court": (metadata.get("court") or "").strip(),
            "court_division": (metadata.get("court_division") or "").strip(),
            "judgment_date": (metadata.get("judgment_date") or "").strip(),
            "judgment_release_date": (metadata.get("judgment_release_date") or "").strip(),
            "hearing_date": (metadata.get("hearing_date") or "").strip(),
            "claimant": (metadata.get("claimant") or "").strip(),
            "defendants": metadata.get("defendants") or [],
            "neutral_citation": (metadata.get("neutral_citation") or "").strip(),
            "claimant_counsel": metadata.get("claimant_counsel") or [],
            "defendant_counsel": metadata.get("defendant_counsel") or [],
            "claimant_law_firm": (metadata.get("claimant_law_firm") or "").strip(),
            "defendant_law_firm": (metadata.get("defendant_law_firm") or "").strip(),
            "doc_type": self._detect_doc_type(metadata, first_page_text),
            "structure_available": structure_available,
            "block_count": len(blocks),
            "page_count": max(block["page_number"] for block in blocks),
            "title": case_name or claim_number or doc_id,
            "blocks": blocks,
        }

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
            block = {
                "block_index": index,
                "page_idx": int(page_idx),
                "page_number": int(page_idx) + 1,
                "text": text,
                "type": structure.get("type", item.get("type", "text")),
                "level": int(structure.get("level") or self._synthetic_level(item, text)),
                "bbox": item.get("bbox") or [],
                "synthetic_structure": index not in structured_items,
            }
            blocks.append(block)
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

    def _extract_claim_number(self, text: str) -> str:
        match = self.CASE_NUMBER_PATTERN.search(text)
        return match.group(0) if match else ""

    def _detect_doc_type(self, metadata: Dict[str, Any], first_page_text: str) -> str:
        claim_number = (metadata.get("claim_number") or "").strip()
        if claim_number:
            return "case_judgment"

        haystack = " ".join(
            value
            for value in [
                metadata.get("case_name") or "",
                metadata.get("neutral_citation") or "",
                first_page_text,
            ]
            if isinstance(value, str)
        ).lower()

        if " law " in f" {haystack} " or "difc law" in haystack:
            return "law"
        return "unknown"
