"""Local evaluation of the legal hybrid RAG pipeline using RAGAS (no submission)."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List
import warnings

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))

from arlc import get_config  # noqa: E402
from examples.legal_hybrid_rag import (  # noqa: E402
    ROOT_DIR as RUNNER_ROOT,
    build_pipeline,
    validate_ingest_coverage,
)


def run_local_ragas_eval(limit: int = 20) -> None:
    """Run the hybrid pipeline on a subset of questions and score with RAGAS.

    This uses a single retrieval-focused metric (ContextRelevance) to keep runtime reasonable.
    """
    from datasets import Dataset  # type: ignore[import-untyped]
    from langchain_openai import ChatOpenAI  # type: ignore[import-untyped]
    from ragas import evaluate  # type: ignore[import-untyped]
    from ragas.metrics import (  # type: ignore[import-untyped]
        ContextRelevance,
    )

    # Suppress the ragas ContextRelevance deprecation warning for this script.
    warnings.filterwarnings(
        "ignore",
        category=DeprecationWarning,
        message="Importing ContextRelevance from 'ragas.metrics' is deprecated*",
    )

    cfg = get_config()
    eval_llm = ChatOpenAI(
        model=cfg.llm_model,
        temperature=0.0,
        openai_api_key=cfg.get_llm_api_key(),
        openai_api_base=cfg.llm_api_base,
    )

    questions_path = ROOT_DIR / "public_dataset" / "questions.json"
    if not questions_path.exists():
        raise FileNotFoundError(f"questions.json not found at {questions_path}")
    questions: List[Dict[str, Any]] = json.loads(questions_path.read_text(encoding="utf-8"))
    if not questions:
        raise ValueError("questions.json is empty")

    limit = min(limit, len(questions))
    subset = questions[:limit]
    print(f"Loaded {len(questions)} questions; evaluating first {limit}")

    ingest_root = RUNNER_ROOT / "ingestion" / "docs_corpus_ingest_result"
    validate_ingest_coverage(cfg.docs_dir, ingest_root)

    print("Building pipeline (will reuse existing index if available)...")
    pipeline = build_pipeline()
    t0 = time.perf_counter()
    pipeline.build_indexes()
    t_index = time.perf_counter()
    print(f"Index build/load took {int((t_index - t0) * 1000)} ms")

    eval_rows: List[Dict[str, Any]] = []
    answer_latencies_ms: List[int] = []

    for idx, q in enumerate(subset, start=1):
        q_id = q["id"]
        q_text = q["question"]
        print(f"[{idx}/{limit}] {q_id}")
        t_q_start = time.perf_counter()
        result = pipeline.answer_question(q)
        t_q_end = time.perf_counter()
        latency_ms = int((t_q_end - t_q_start) * 1000)
        answer_latencies_ms.append(latency_ms)
        print(f"  answer latency: {latency_ms} ms, refs: {len(result.retrieval_refs)}")

        contexts = [
            str(getattr(doc, "page_content", "") or "").strip()
            for doc in getattr(result, "supporting_docs", [])[:4]
        ]
        contexts = [c for c in contexts if c]

        eval_rows.append(
            {
                "question": q_text,
                "answer": str(result.answer),
                "contexts": contexts,
            }
        )

    ds = Dataset.from_dict(
        {
            "question": [row["question"] for row in eval_rows],
            "answer": [row["answer"] for row in eval_rows],
            "contexts": [row["contexts"] for row in eval_rows],
        }
    )

    print("\nRunning RAGAS evaluation (ContextRelevance only)...")
    result = evaluate(ds, metrics=[ContextRelevance()], llm=eval_llm)
    print("\n=== RAGAS scores (mean) ===")
    if hasattr(result, "_repr_dict") and result._repr_dict:
        for key, value in result._repr_dict.items():
            try:
                v = float(value)
            except Exception:
                v = value
            print(f"  {key}: {v}")
    elif hasattr(result, "to_pandas"):
        df = result.to_pandas()
        print(df.describe() or str(result))
    else:
        print(result)

    if answer_latencies_ms:
        mean_lat = sum(answer_latencies_ms) / len(answer_latencies_ms)
        print(f"\nMean answer latency: {mean_lat:.1f} ms over {len(answer_latencies_ms)} questions")


def main() -> None:
    limit_env = sys.argv[1] if len(sys.argv) > 1 else ""
    try:
        limit = int(limit_env) if limit_env else 20
    except ValueError:
        limit = 20
    run_local_ragas_eval(limit=limit)


if __name__ == "__main__":
    main()

