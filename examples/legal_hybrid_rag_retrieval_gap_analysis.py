"""
Run retrieval on all questions and build a gap matrix: where we get 0 docs vs 1-3 vs 4-6,
broken down by answer_type, title-page routing, comparison mode, and case/law routing.
Outputs a matrix + list of zero-doc questions for data/retrieval gap analysis.

Eval mode (default): compares current run to retrieval_gap_baseline.json and exits 1 if
retrieval regresses (more zero-doc questions or fewer 4-6-doc retrievals). Use to catch
regressions in CI or before merge.
"""

from __future__ import annotations

import json
import logging
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
# Suppress httpx INFO so we only see our summary
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("retrieval").setLevel(logging.WARNING)

from examples.legal_hybrid_rag import (  # type: ignore[import-untyped]
    CONFIG,
    ROOT_DIR as RUNNER_ROOT,
    build_pipeline,
    validate_ingest_coverage,
)


def bucket_docs(n: int) -> str:
    if n == 0:
        return "0_docs"
    if n <= 3:
        return "1-3_docs"
    return "4-6_docs"


def run_retrieval_on_all_questions(limit: int | None = None) -> List[Dict[str, Any]]:
    questions_path = RUNNER_ROOT / "public_dataset" / "questions.json"
    questions: List[Dict[str, Any]] = json.loads(questions_path.read_text(encoding="utf-8"))
    if limit is not None:
        questions = questions[:limit]
    ingest_root = RUNNER_ROOT / "ingestion" / "docs_corpus_ingest_result"
    validate_ingest_coverage(CONFIG.docs_dir, ingest_root)

    os.environ.setdefault("LEGAL_HYBRID_SKIP_INDEXING", "1")
    os.environ.setdefault("LEGAL_HYBRID_ENABLE_RERANK", "0")
    pipeline = build_pipeline()
    pipeline.build_indexes()

    results: List[Dict[str, Any]] = []
    for i, q in enumerate(questions):
        qid = q.get("id", "")
        question_text = q.get("question", "")
        answer_type = q.get("answer_type", "free_text")
        route = pipeline.router.route(question_text, answer_type)
        docs, _ = pipeline.retrieve(question_text, answer_type=answer_type, route=route)
        n_docs = len(docs)
        doc_types = list({str((d.metadata or {}).get("doc_type", "")) for d in docs})
        chunk_kinds = list({str((d.metadata or {}).get("chunk_kind", "")) for d in docs})
        results.append({
            "id": qid,
            "question": question_text[:120],
            "answer_type": answer_type,
            "n_docs": n_docs,
            "bucket": bucket_docs(n_docs),
            "prefer_title_page": route.prefer_title_page,
            "comparison_mode": route.comparison_mode,
            "has_case_ids": len(route.case_ids) > 0,
            "has_law_names_or_numbers": bool(route.law_names or route.law_numbers),
            "preferred_chunk_kinds": list(route.preferred_chunk_kinds),
            "case_ids": list(route.case_ids),
            "doc_types": doc_types,
            "chunk_kinds": chunk_kinds,
        })
    return results


def build_matrix(results: List[Dict[str, Any]]) -> Tuple[str, List[Dict[str, Any]]]:
    buckets = ["0_docs", "1-3_docs", "4-6_docs"]
    total = len(results)

    def row(name: str, subset: List[Dict[str, Any]]) -> Dict[str, Any]:
        n = len(subset)
        counts = defaultdict(int)
        for r in subset:
            counts[r["bucket"]] += 1
        return {
            "segment": name,
            "n": n,
            "pct_of_total": round(100 * n / total, 1) if total else 0,
            "0_docs": counts["0_docs"],
            "1-3_docs": counts["1-3_docs"],
            "4-6_docs": counts["4-6_docs"],
            "pct_zero": round(100 * counts["0_docs"] / n, 1) if n else 0,
        }

    rows: List[Dict[str, Any]] = []

    # Overall
    rows.append(row("OVERALL", results))

    # By answer_type
    by_answer = defaultdict(list)
    for r in results:
        by_answer[r["answer_type"]].append(r)
    for at in sorted(by_answer.keys()):
        rows.append(row(f"  answer_type={at}", by_answer[at]))

    # By prefer_title_page
    title_page = [r for r in results if r["prefer_title_page"]]
    if title_page:
        rows.append(row("  prefer_title_page=True", title_page))
    not_title = [r for r in results if not r["prefer_title_page"]]
    if not_title:
        rows.append(row("  prefer_title_page=False", not_title))

    # By comparison_mode
    comp = [r for r in results if r["comparison_mode"]]
    if comp:
        rows.append(row("  comparison_mode=True", comp))
    single = [r for r in results if not r["comparison_mode"]]
    if single:
        rows.append(row("  comparison_mode=False", single))

    # By has_case_ids
    with_case = [r for r in results if r["has_case_ids"]]
    if with_case:
        rows.append(row("  has_case_ids=True", with_case))
    no_case = [r for r in results if not r["has_case_ids"]]
    if no_case:
        rows.append(row("  has_case_ids=False", no_case))

    # Format as text matrix
    lines = [
        "",
        "RETRIEVAL GAP MATRIX (0_docs = retrieval gap; 4-6_docs = healthy)",
        "=" * 90,
    ]
    header = f"{'segment':<36} {'n':>5} {'%tot':>6} {'0_docs':>8} {'1-3':>6} {'4-6':>6} {'%zero':>7}"
    lines.append(header)
    lines.append("-" * 90)
    for r in rows:
        lines.append(
            f"{r['segment']:<36} {r['n']:>5} {r['pct_of_total']:>5.1f}% "
            f"{r['0_docs']:>8} {r['1-3_docs']:>6} {r['4-6_docs']:>6} {r['pct_zero']:>6.1f}%"
        )
    lines.append("")
    zero_doc_list = [r for r in results if r["n_docs"] == 0]
    soft_gap_list = [r for r in results if 1 <= r["n_docs"] <= 3]
    lines.append(f"ZERO-DOC QUESTIONS (retrieval/data gaps): {len(zero_doc_list)}")
    lines.append("-" * 90)
    for r in zero_doc_list:
        lines.append(f"  id: {r['id']}")
        lines.append(f"    answer_type={r['answer_type']} prefer_title_page={r['prefer_title_page']} comparison={r['comparison_mode']}")
        lines.append(f"    preferred_kinds={r['preferred_chunk_kinds']} case_ids={r['case_ids'][:3]}")
        lines.append(f"    Q: {r['question']}...")
        lines.append("")
    lines.append(f"SOFT GAPS (1-3 docs): {len(soft_gap_list)}")
    lines.append("-" * 90)
    for r in soft_gap_list:
        lines.append(f"  id: {r['id']} n_docs={r['n_docs']}")
        lines.append(f"    answer_type={r['answer_type']} prefer_title_page={r['prefer_title_page']} comparison={r['comparison_mode']}")
        lines.append(f"    preferred_kinds={r['preferred_chunk_kinds']} case_ids={r['case_ids'][:3]}")
        lines.append(f"    Q: {r['question']}...")
        lines.append("")
    return "\n".join(lines), zero_doc_list


def compute_overall_counts(results: List[Dict[str, Any]]) -> Dict[str, int]:
    n = len(results)
    counts = defaultdict(int)
    for r in results:
        counts[r["bucket"]] += 1
    return {
        "n_questions": n,
        "0_docs": counts["0_docs"],
        "1-3_docs": counts["1-3_docs"],
        "4-6_docs": counts["4-6_docs"],
    }


def run_eval(baseline_path: Path, results: List[Dict[str, Any]]) -> int:
    """Compare current results to baseline. Return 0 if pass, 1 if regressed."""
    with open(baseline_path, encoding="utf-8") as f:
        baseline = json.load(f)
    current = compute_overall_counts(results)
    n_zero = current["0_docs"]
    n_4_6 = current["4-6_docs"]
    max_zero = baseline.get("max_zero_docs", 999)
    min_4_6 = baseline.get("min_4_6_docs", 0)
    regressions: List[str] = []
    if n_zero > max_zero:
        regressions.append(f"zero_docs {n_zero} > baseline max {max_zero}")
    if n_4_6 < min_4_6:
        regressions.append(f"4-6_docs {n_4_6} < baseline min {min_4_6}")
    if regressions:
        print("EVAL FAIL: retrieval regressed:", "; ".join(regressions), flush=True)
        return 1
    print("EVAL PASS: retrieval within baseline (zero_docs<={}, 4-6_docs>={})".format(max_zero, min_4_6), flush=True)
    return 0


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Retrieval gap analysis on legal hybrid pipeline")
    parser.add_argument("--limit", type=int, default=None, help="Max questions (default: all)")
    parser.add_argument("--out", type=str, default=None, help="Write full results JSON here")
    parser.add_argument(
        "--save-baseline",
        action="store_true",
        help="Write current overall metrics to retrieval_gap_baseline.json (update lock after intentional improvements)",
    )
    parser.add_argument(
        "--no-eval",
        action="store_true",
        help="Skip baseline comparison; just print matrix and optional --out",
    )
    args = parser.parse_args()

    print("Loading questions and running retrieval (query-only)...", flush=True)
    results = run_retrieval_on_all_questions(limit=args.limit)
    print(f"Done. {len(results)} questions.", flush=True)

    matrix_text, zero_doc_list = build_matrix(results)
    print(matrix_text, flush=True)

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump({"results": results, "n_zero_docs": len(zero_doc_list)}, f, indent=2)
        print(f"Full results written to {out_path}", flush=True)

    baseline_path = Path(__file__).resolve().parent / "retrieval_gap_baseline.json"
    if args.save_baseline:
        counts = compute_overall_counts(results)
        baseline = {
            "description": "Locked retrieval gap metrics; eval fails if current run is worse.",
            "n_questions": counts["n_questions"],
            "max_zero_docs": counts["0_docs"],
            "min_4_6_docs": counts["4-6_docs"],
        }
        with open(baseline_path, "w", encoding="utf-8") as f:
            json.dump(baseline, f, indent=2)
        print(f"Baseline updated: {baseline_path}", flush=True)
        raise SystemExit(0)

    if not args.no_eval and baseline_path.exists():
        raise SystemExit(run_eval(baseline_path, results))
    elif not args.no_eval:
        print("No baseline found at {}; run with --save-baseline once to lock.".format(baseline_path), flush=True)
        raise SystemExit(0)


if __name__ == "__main__":
    main()
