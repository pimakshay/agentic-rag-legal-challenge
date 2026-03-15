"""
Sparse (BM25) retriever using Pyserini (Lucene backend).
"""

import json
import logging
import os
import subprocess
from typing import List, Optional

from langchain_core.documents import Document
from pyserini.search.lucene import LuceneSearcher

from retrieval.retrievers.base import BaseRAGRetriever, RetrievalResult

logger = logging.getLogger(__name__)

os.environ["JAVA_TOOL_OPTIONS"] = "-Dorg.apache.lucene.store.MMapDirectory.enableMemorySegments=false"

class PyseriniBM25Retriever(BaseRAGRetriever):
    """
    BM25 retriever backed by Pyserini (Lucene).

    Args:
        corpus_dir: Folder containing json documents for indexing
        index_dir: Folder to store Lucene index
        default_k: Default number of documents to retrieve
        threads: Indexing threads
    """

    def __init__(
        self,
        documents: Optional[List[Document]] = None,
        corpus_dir: Optional[str] = None,
        index_dir: Optional[str] = None,
        default_k: int = 5,
        threads: int = 8,
    ):
        self.corpus_dir = corpus_dir
        self.index_dir = index_dir
        self.default_k = default_k
        self.threads = threads
        self._documents = documents or []
        self._searcher: Optional[LuceneSearcher] = None

        os.makedirs(corpus_dir, exist_ok=True)
        os.makedirs(index_dir, exist_ok=True)
        if documents:
            self._build_index(documents)

    def _init_searcher(self):
        """Initialize LuceneSearcher."""
        self._searcher = LuceneSearcher(self.index_dir)
        logger.info(f"Loaded Pyserini index from {self.index_dir}")

    def _build_index(self, documents: List[Document]):
        """Build Lucene BM25 index using Pyserini."""
        if not self.corpus_dir or not self.index_dir:
            raise ValueError("corpus_dir and index_dir must be set")
        with open(os.path.join(self.corpus_dir, "corpus.jsonl"), 'w') as f:
            for i, document in enumerate(documents):
                metadata = dict(document.metadata or {})
                doc_id = str(metadata.get("doc_id") or "")
                metadata["page_content"] = document.page_content
                metadata["id"] = document.id
                content = self._preprocess(document.page_content)
                f.write(json.dumps({"id": i, "contents": content, **metadata}, ensure_ascii=False)+ "\n")
        cmd = [
            "python",
            "-m",
            "pyserini.index.lucene",
            "--collection",
            "JsonCollection",
            "--input",
            self.corpus_dir,
            "--index",
            self.index_dir,
            "--generator",
            "DefaultLuceneDocumentGenerator",
            "--threads",
            str(self.threads),
            "--storePositions",
            "--storeRaw",
        ]

        logger.info("Building Pyserini index...")
        subprocess.run(cmd, check=True)

        self._init_searcher()

    def _preprocess(self, text: str) -> str:
        """Simple query preprocessing."""
        text = text.lower()
        text = text.replace("\n", " . ")
        text = "".join(c if c.isalnum() else " " for c in text)
        text = " ".join(text.split())
        return text

    def retrieve(self, query: str, k: Optional[int] = None, candidate_doc_ids: List = []) -> RetrievalResult:
        """
        Retrieve documents using BM25.

        Returns:
            RetrievalResult with documents and scores.
        """
        if not self._searcher:
            raise RuntimeError("Searcher not initialized")

        k = k or self.default_k
        query = self._preprocess(query)

        hits = self._searcher.search(query, k=k)

        documents: List[Document] = []
        scores: List[float] = []
        for hit in hits:
            doc = self._searcher.doc(hit.docid)
            metadata = json.loads(doc.raw())
            if candidate_doc_ids and (str(metadata.get("doc_id") or "") not in candidate_doc_ids):
                continue
            page_content = metadata["page_content"]
            del metadata["page_content"]
            id = metadata["id"]
            del metadata["id"]
            del metadata["contents"]

            document = Document(page_content=page_content, metadata=metadata, id=id)
            documents.append(
                document
            )

            scores.append(hit.score)
        return RetrievalResult(
            documents=documents,
            scores=scores,
            metadata={
                "retriever_type": "sparse",
                "k": k,
                "query": query,
            },
        )

    @property
    def is_initialized(self) -> bool:
        return self._searcher is not None