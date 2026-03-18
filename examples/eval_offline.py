import asyncio
import json
import sys
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))

from arlc import get_config
from examples.legal_hybrid_rag import build_pipeline, validate_ingest_coverage

def process_question(pipeline, q, i, total):
    t0 = time.perf_counter()
    ans_res = pipeline.answer_question(q)
    t1 = time.perf_counter()
    latency = (t1 - t0) * 1000
    
    result = {
        "id": q["id"],
        "question": q["question"],
        "answer_type": q.get("answer_type", "free_text"),
        "predicted_answer": ans_res.answer,
        "retrieval_refs": [
            {"doc_id": r.doc_id, "page_numbers": r.page_numbers} 
            for r in ans_res.retrieval_refs
        ],
        "latency_ms": latency,
        "raw_response": ans_res.raw_response
    }
    
    print(f"[{i}/{total}] ID: {q['id']} | Type: {q.get('answer_type', 'free_text')} | Latency: {latency:.1f}ms")
    return result

def main():
    cfg = get_config()
    questions_path = ROOT_DIR / "public_dataset" / "questions.json"
    if not questions_path.exists():
        print(f"File not found: {questions_path}")
        return
    
    questions = json.loads(questions_path.read_text(encoding="utf-8"))[:100]
    print(f"Loaded {len(questions)} questions")
    
    ingest_root = ROOT_DIR / "ingestion" / "docs_corpus_ingest_result"
    validate_ingest_coverage(cfg.docs_dir, ingest_root)

    print("Building pipeline...")
    pipeline = build_pipeline()
    pipeline.build_indexes()
    
    results = []
    
    print("Starting concurrent evaluation...")
    t_start_eval = time.perf_counter()
    
    with ThreadPoolExecutor(max_workers=10) as executor:
        future_to_q = {
            executor.submit(process_question, pipeline, q, i+1, len(questions)): q 
            for i, q in enumerate(questions)
        }
        for future in as_completed(future_to_q):
            try:
                res = future.result()
                results.append(res)
            except Exception as exc:
                print(f"Question generated an exception: {exc}")
                
    t_end_eval = time.perf_counter()
    print(f"Total concurrent evaluation time: {(t_end_eval - t_start_eval):.2f}s")
        
    out_path = ROOT_DIR / "offline_eval_results.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"Done! Results saved to {out_path}")

if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    main()
