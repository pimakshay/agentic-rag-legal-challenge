"""Local evaluation of the legal hybrid RAG pipeline using RAGAS (no submission)."""

from __future__ import annotations

import asyncio
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

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

    Uses ragas.metrics.collections: ContextRelevance (retrieval), Faithfulness
    (answer grounded in context), AnswerRelevancy (answer addresses the question).
    No ground truth required. Uses llm_factory + OpenAI client from config.
    """
    from openai import AsyncOpenAI  # type: ignore[import-untyped]

    from ragas.embeddings import OpenAIEmbeddings  # type: ignore[import-untyped]
    from ragas.llms import llm_factory  # type: ignore[import-untyped]
    from ragas.metrics.collections import (  # type: ignore[import-untyped]
        ContextRelevance,
        Faithfulness,
        AnswerRelevancy,
    )

    cfg = get_config()
    client = AsyncOpenAI(
        api_key=cfg.get_llm_api_key(),
        base_url=cfg.llm_api_base,
    )
    llm = llm_factory(cfg.llm_model, client=client, max_tokens=8192)
    embeddings = OpenAIEmbeddings(client=client, model=cfg.embedding_model)

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
        answer_str = str(result.answer).strip() if result.answer is not None else ""
        if not answer_str:
            answer_str = "[No response]"

        eval_rows.append(
            {
                "user_input": q_text,
                "response": answer_str,
                "retrieved_contexts": contexts,
            }
        )

    async def run_metrics() -> Dict[str, float]:
        context_relevance = ContextRelevance(llm=llm)
        faithfulness = Faithfulness(llm=llm)
        answer_relevancy = AnswerRelevancy(llm=llm, embeddings=embeddings)

        inputs_cr = [
            {"user_input": r["user_input"], "retrieved_contexts": r["retrieved_contexts"]}
            for r in eval_rows
        ]
        inputs_f = [
            {
                "user_input": r["user_input"],
                "response": r["response"],
                "retrieved_contexts": r["retrieved_contexts"],
            }
            for r in eval_rows
        ]
        inputs_ar = [
            {"user_input": r["user_input"], "response": r["response"]}
            for r in eval_rows
        ]

        results_cr, results_f, results_ar = await asyncio.gather(
            context_relevance.abatch_score(inputs_cr),
            faithfulness.abatch_score(inputs_f),
            answer_relevancy.abatch_score(inputs_ar),
        )

        def mean_score(results: List[Any]) -> float:
            values = [r.value for r in results if hasattr(r, "value") and r.value is not None]
            if not values:
                return float("nan")
            return sum(float(v) for v in values) / len(values)

        return {
            "context_relevance": mean_score(results_cr),
            "faithfulness": mean_score(results_f),
            "answer_relevancy": mean_score(results_ar),
        }

    print("\nRunning RAGAS evaluation (collections: ContextRelevance, Faithfulness, AnswerRelevancy)...")
    scores = asyncio.run(run_metrics())

    print("\n=== RAGAS scores (mean) ===")
    for key, value in scores.items():
        try:
            v = float(value)
        except (TypeError, ValueError):
            v = value
        print(f"  {key}: {v}")

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
