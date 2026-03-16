"""
End-to-end check: run full pipeline (retrieve + LLM answer) on N questions without reranker.
Prints per-question summary and aggregate stats (refs, docs, empty answers).
No RAGAS dependency. Set LEGAL_HYBRID_ENABLE_RERANK=0 and LEGAL_HYBRID_SKIP_INDEXING=1.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))

os.environ.setdefault("LEGAL_HYBRID_ENABLE_RERANK", "0")
os.environ.setdefault("LEGAL_HYBRID_SKIP_INDEXING", "1")

from examples.legal_hybrid_rag import (  # type: ignore[import-untyped]
    CONFIG,
    ROOT_DIR as RUNNER_ROOT,
    build_pipeline,
    validate_ingest_coverage,
)


def run_e2e_check(limit: int = 15) -> None:
    questions_path = RUNNER_ROOT / "public_dataset" / "questions.json"
    questions: List[Dict[str, Any]] = json.loads(questions_path.read_text(encoding="utf-8"))
    subset = questions[:limit]
    ingest_root = RUNNER_ROOT / "ingestion" / "docs_corpus_ingest_result"
    validate_ingest_coverage(CONFIG.docs_dir, ingest_root)

    print("Building pipeline (no reranker, skip index)...")
    pipeline = build_pipeline()
    pipeline.build_indexes()
    print(f"Running e2e on {len(subset)} questions...\n")

    rows: List[Dict[str, Any]] = []
    for idx, q in enumerate(subset, start=1):
        qid = q.get("id", "")[:12]
        result = pipeline.answer_question(q)
        n_refs = len(getattr(result, "retrieval_refs", []) or [])
        n_docs = len(getattr(result, "supporting_docs", []) or [])
        answer = result.answer
        answer_str = str(answer) if answer is not None else ""
        empty = answer is None or (isinstance(answer, str) and not answer_str.strip())
        rows.append({
            "id": qid,
            "answer_type": q.get("answer_type", ""),
            "answer": answer_str[:80] if answer_str else "(null)",
            "n_refs": n_refs,
            "n_docs": n_docs,
            "empty": empty,
        })
        print(f"  [{idx:2}/{limit}] {qid}  type={q.get('answer_type', '')}  refs={n_refs}  docs={n_docs}  empty={empty}  answer={answer_str[:50] or '(null)'}...")

    n = len(rows)
    mean_refs = sum(r["n_refs"] for r in rows) / n if n else 0
    mean_docs = sum(r["n_docs"] for r in rows) / n if n else 0
    zero_refs = sum(1 for r in rows if r["n_refs"] == 0)
    empty_answers = sum(1 for r in rows if r["empty"])

    print("\n=== E2E summary (no reranker) ===")
    print(f"  Questions: {n}")
    print(f"  Mean retrieval_refs per question: {mean_refs:.2f}")
    print(f"  Mean supporting_docs per question: {mean_docs:.2f}")
    print(f"  Questions with 0 refs: {zero_refs}")
    print(f"  Questions with null/empty answer: {empty_answers}")


def main() -> None:
    limit = 15
    if len(sys.argv) > 1:
        try:
            limit = int(sys.argv[1])
        except ValueError:
            pass
    run_e2e_check(limit=limit)


if __name__ == "__main__":
    main()
