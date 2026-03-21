from __future__ import annotations

import os
from typing import Any, Iterable, List, Dict

from langchain_core.documents import Document


class TurbopufferStore:
    """Hybrid (vector + BM25) store backed by Turbopuffer."""

    def __init__(self, namespace: str, region: str = "gcp-europe-west3") -> None:
        try:
            import turbopuffer  # type: ignore
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "Optional dependency `turbopuffer` is required to use TurbopufferStore. "
                "Install it or disable Turbopuffer by setting `use_turbopuffer=False` / "
                "leaving `TURBOPUFFER_API_KEY` unset."
            ) from exc

        api_key = os.getenv("TURBOPUFFER_API_KEY")
        if not api_key:
            raise ValueError("TURBOPUFFER_API_KEY not set")
        client = turbopuffer.Turbopuffer(api_key=api_key, region=region)
        # Namespace is created automatically on first write
        self.ns = client.namespace(namespace)

    def index_chunks(self, chunks: List[Document], embed_fn: Any) -> None:
        """Upsert chunk embeddings and metadata into Turbopuffer."""
        texts = [doc.page_content for doc in chunks]
        vectors = embed_fn(texts)

        upsert_rows: List[Dict[str, Any]] = []
        for i, (doc, vec) in enumerate(zip(chunks, vectors)):
            meta = doc.metadata or {}
            page_numbers = meta.get("page_numbers") or []
            upsert_rows.append(
                {
                    "id": i,
                    "vector": vec,
                    "text": doc.page_content,
                    "chunk_id": str(meta.get("chunk_id", "")),
                    "doc_id": meta.get("doc_id", ""),
                    "page_numbers": ",".join(str(p) for p in page_numbers),
                    "chunk_kind": meta.get("chunk_kind", ""),
                    "doc_type": meta.get("doc_type", ""),
                }
            )

        self.ns.write(
            upsert_rows=upsert_rows,
            distance_metric="cosine_distance",
            schema={
                "text": {
                    "type": "string",
                    "full_text_search": True,
                }
            },
        )

    def _row_attrs(self, row: Any) -> Dict[str, Any]:
        """Extract a string-keyed dict from a Turbopuffer Row.

        The client returns Row (Pydantic) with id/vector/$dist and requested
        attributes as extra fields (model_extra), not under .attributes.
        """
        attrs: Dict[str, Any] = {}
        extra = getattr(row, "model_extra", None) or getattr(row, "__pydantic_extra__", None)
        if extra:
            attrs.update(extra)
        attrs["id"] = getattr(row, "id", None)
        for key in ("text", "chunk_id", "doc_id", "page_numbers", "chunk_kind", "doc_type"):
            if key not in attrs:
                attrs[key] = getattr(row, key, None)
        return attrs

    def _rows_to_docs(self, rows: Iterable[Any]) -> List[Document]:
        """Convert Turbopuffer row objects into LangChain Documents."""
        docs: List[Document] = []
        for row in rows:
            attrs = self._row_attrs(row)
            row_id = attrs.get("id")

            pn_raw = attrs.get("page_numbers") or ""
            page_numbers: List[int] = []
            if pn_raw:
                for token in str(pn_raw).split(","):
                    token = token.strip()
                    if token.isdigit():
                        page_numbers.append(int(token))

            text = attrs.get("text") or ""
            if hasattr(text, "strip"):
                text = str(text).strip()
            else:
                text = str(text)

            docs.append(
                Document(
                    page_content=text,
                    metadata={
                        "chunk_id": attrs.get("chunk_id") or row_id,
                        "doc_id": attrs.get("doc_id") or "",
                        "page_numbers": page_numbers,
                        "chunk_kind": attrs.get("chunk_kind") or "",
                        "doc_type": attrs.get("doc_type") or "",
                    },
                )
            )
        return docs

    def vector_search(self, query_vector: List[float], top_k: int) -> List[Document]:
        result = self.ns.query(
            rank_by=("vector", "ANN", query_vector),
            top_k=top_k,
            include_attributes=[
                "text",
                "chunk_id",
                "doc_id",
                "page_numbers",
                "chunk_kind",
                "doc_type",
            ],
        )
        return self._rows_to_docs(result.rows)

    def text_search(self, query_text: str, top_k: int) -> List[Document]:
        result = self.ns.query(
            rank_by=("text", "BM25", query_text),
            top_k=top_k,
            include_attributes=[
                "text",
                "chunk_id",
                "doc_id",
                "page_numbers",
                "chunk_kind",
                "doc_type",
            ],
        )
        return self._rows_to_docs(result.rows)

