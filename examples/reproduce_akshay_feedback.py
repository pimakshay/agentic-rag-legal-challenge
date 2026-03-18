"""
Reproduce Akshay's 7 reported failures with the current pipeline.

Run from repo root:
  uv run python examples/reproduce_akshay_feedback.py

Use --from-results to compare against existing offline_eval_results.json
without calling the pipeline (faster).
"""
import argparse
import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

# Akshay's test cases: question_id -> (expected_answer, short_why)
AKSHAY_CASES = {
    "a8813f8f56099bd156b5fe797cc44b061d2a54fb69ca2e748ffb7cc462eaa85e": (
        "CFI 010/2024",
        "CFI 010/2024=2026-01-23, CFI 016/2025=2026-02-03",
    ),
    "acd7d2aea67ff920768aae75300837412f603e0e29f048ea33171ba52c1c0cd8": (
        "ENF 269/2023",
        "ENF 269/2023=2025-07-01, SCT 514/2025=2026-01-07",
    ),
    "6b20dd63f6b2072c51f7063e32f5accf7218fd36fdfcc3fb627867139b41359e": (
        "SCT 295/2025",
        "SCT 169/2025=2025-12-24, SCT 295/2025=2025-12-10",
    ),
    "30040cc854022341de6bb9ee7dc3e932d540666a1addda4750bf974ce9d9292f": (
        "ENF 269/2023",
        "CFI 016/2025=2026-02-03, ENF 269/2023=2025-07-01",
    ),
    "dd736c7955429054fc55ea9eb55b96d3feb13e410f27a25651eefe5ecb536575": (
        "ENF 269/2023",
        "ENF 269/2023=2025-07-01, SCT 169/2025=2025-12-24",
    ),
    "de67a8574c9e4352ba32d662761e8b2e59ce1d2ae36c33cf5c563dfc5d879920": (
        True,
        "Judge overlap [WAYNE MARTIN]",
    ),
    "b13500141f04c86f328b49c56ba01c7bc13bce5afee9646860912718291a5c1a": (
        False,
        "Article 7(3)(j) requires Board approval for delegation",
    ),
}


def _normalize(s):
    if s is None:
        return None
    if isinstance(s, bool):
        return s
    return " ".join((str(s) or "").lower().split()).strip()


def _match(actual, expected):
    if actual is None and expected is None:
        return True
    if actual is None or expected is None:
        return False
    if isinstance(expected, bool):
        return bool(actual) == bool(expected)
    return _normalize(str(actual)) == _normalize(str(expected))


def run_from_pipeline():
    from examples.legal_hybrid_rag import build_pipeline, validate_ingest_coverage

    questions_path = ROOT_DIR / "public_dataset" / "questions.json"
    questions_all = json.loads(questions_path.read_text(encoding="utf-8"))
    questions = [q for q in questions_all if q["id"] in AKSHAY_CASES]
    if len(questions) != 7:
        print(f"Expected 7 questions, found {len(questions)} in questions.json")
        return None

    ingest_root = ROOT_DIR / "ingestion" / "docs_corpus_ingest_result"
    from arlc import get_config

    validate_ingest_coverage(get_config().docs_dir, ingest_root)
    print("Building pipeline...")
    pipeline = build_pipeline()
    pipeline.build_indexes()
    if hasattr(pipeline.store, "hint_cache_warm"):
        try:
            pipeline.store.hint_cache_warm()
        except Exception:
            pass

    results = []
    for i, q in enumerate(questions, 1):
        print(f"[{i}/7] {q['id'][:16]}...")
        ans_res = pipeline.answer_question(q)
        results.append({
            "id": q["id"],
            "question": q["question"],
            "answer_type": q.get("answer_type", "free_text"),
            "predicted_answer": ans_res.answer,
        })
    return results


def run_from_results():
    path = ROOT_DIR / "offline_eval_results.json"
    if not path.exists():
        print(f"File not found: {path}")
        return None
    results = json.loads(path.read_text(encoding="utf-8"))
    return [r for r in results if r["id"] in AKSHAY_CASES]


def main():
    ap = argparse.ArgumentParser(description="Reproduce Akshay feedback on 7 questions")
    ap.add_argument(
        "--from-results",
        action="store_true",
        help="Use offline_eval_results.json instead of running pipeline",
    )
    args = ap.parse_args()

    if args.from_results:
        print("Using existing offline_eval_results.json")
        results = run_from_results()
    else:
        results = run_from_pipeline()

    if not results or len(results) != 7:
        print("Could not get 7 results. Aborting.")
        sys.exit(1)

    print()
    print("Akshay feedback reproduction")
    print("=" * 72)
    wrong = 0
    for r in results:
        qid = r["id"]
        expected, why = AKSHAY_CASES[qid]
        actual = r.get("predicted_answer")
        match = _match(actual, expected)
        status = "OK" if match else "WRONG"
        if not match:
            wrong += 1
        q_short = (r.get("question") or "")[:65] + "..." if len(r.get("question") or "") > 65 else (r.get("question") or "")
        print(f"{status} {qid[:16]}...")
        print(f"  Q: {q_short}")
        print(f"  ACTUAL:   {actual!r}")
        print(f"  EXPECTED: {expected!r}")
        print(f"  WHY: {why}")
        print()

    print("=" * 72)
    print(f"Summary: {7 - wrong}/7 correct, {wrong}/7 WRONG (Akshay's report: all 7 WRONG)")
    if wrong == 7:
        print("Issue reproduced: pipeline matches Akshay's reported failures.")
    sys.exit(0 if wrong == 0 else 1)


if __name__ == "__main__":
    main()
