"""
Generate a full submission.json locally (no upload to competition API).

Runs the legal hybrid pipeline against the local question set and writes the
submission payload compatible with the evaluation runner.

Example:
  uv run python examples/generate_submission_json_local.py --out submission.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))

from arlc import (  # noqa: E402
    SubmissionAnswer,
    SubmissionBuilder,
    Telemetry,
    TelemetryTimer,
    TimingMetrics,
    UsageMetrics,
    get_config,
)
from examples.legal_hybrid_rag import build_pipeline, validate_ingest_coverage  # noqa: E402


def count_tokens(tokenizer, text: str) -> int:
    # Use the same behavior as the starter kit runner.
    return len(tokenizer.encode(text)) if tokenizer else max(1, len(text.split()))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="submission.json")
    ap.add_argument("--questions-path", default=str(ROOT_DIR / "public_dataset" / "questions.json"))
    ap.add_argument("--limit", type=int, default=0, help="If > 0, only run first N questions")
    args = ap.parse_args()

    cfg = get_config()

    questions_path = Path(args.questions_path)
    if not questions_path.exists():
        raise FileNotFoundError(f"Questions file not found: {questions_path}")

    questions = json.loads(questions_path.read_text(encoding="utf-8"))
    if not isinstance(questions, list):
        raise ValueError("Expected questions.json to be a list")
    if args.limit and args.limit > 0:
        questions = questions[: args.limit]

    ingest_root = ROOT_DIR / "ingestion" / "docs_corpus_ingest_result"
    validate_ingest_coverage(cfg.docs_dir, ingest_root)

    print("Building pipeline...")
    pipeline = build_pipeline()
    pipeline.build_indexes()

    if hasattr(pipeline.store, "hint_cache_warm"):
        try:
            pipeline.store.hint_cache_warm()
            print("Turbopuffer cache warm hint sent.")
        except Exception:
            pass

    # Tokenizer mapping can fail for some model names (e.g. "gpt-5.4").
    # Fall back to a simple whitespace token approximation.
    tokenizer = None
    try:
        import tiktoken

        tokenizer = tiktoken.encoding_for_model(cfg.llm_model)
    except Exception:
        tokenizer = None

    builder = SubmissionBuilder(
        architecture_summary=(
            "Legal hybrid RAG over ingest outputs with structure-aware chunking, "
            "typed routing, hybrid dense/BM25 retrieval, and starter-kit telemetry."
        )
    )

    print(f"Answering {len(questions)} questions...")
    for i, q in enumerate(questions, start=1):
        question_id = q["id"]
        print(f"[{i}/{len(questions)}] {question_id}")

        telemetry_timer = TelemetryTimer()
        result = pipeline.answer_question(q, telemetry_timer=telemetry_timer)
        timing = telemetry_timer.finish()

        ttft_ms = max(0, min(timing.ttft_ms, timing.total_time_ms))
        total_time_ms = max(timing.total_time_ms, ttft_ms)

        prompt = result.debug_metadata.get("prompt", "") if isinstance(result.debug_metadata, dict) else ""
        output_text = result.raw_response or str(result.answer)
        usage = UsageMetrics(
            input_tokens=count_tokens(tokenizer, prompt),
            output_tokens=count_tokens(tokenizer, output_text),
        )

        telemetry = Telemetry(
            timing=TimingMetrics(
                ttft_ms=ttft_ms,
                tpot_ms=timing.tpot_ms,
                total_time_ms=total_time_ms,
            ),
            retrieval=result.retrieval_refs,
            usage=usage,
            model_name=cfg.llm_model,
        )

        builder.add_answer(
            SubmissionAnswer(
                question_id=question_id,
                answer=result.answer,
                telemetry=telemetry,
            )
        )

    out_path = Path(args.out)
    out_path.write_text(json.dumps(builder.build(), ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved: {out_path.resolve()}")


if __name__ == "__main__":
    main()

