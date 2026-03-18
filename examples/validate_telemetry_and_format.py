import json
import re

def validate_results(file_path):
    with open(file_path, "r") as f:
        data = json.load(f)
        
    print(f"Validating {len(data)} results...\n")
    
    malformed_telemetry = 0
    type_errors = 0
    fluff_errors = 0
    missing_answers = 0
    
    for q in data:
        ans = q["predicted_answer"]
        refs = q["retrieval_refs"]
        t = q["answer_type"]
        latency = q["latency_ms"]
        
        # In actual submission we have ttft_ms <= total_time_ms.
        # Here we just check latency > 0.
        if latency < 0:
            print(f"[{q['id']}] Negative latency!")
            malformed_telemetry += 1
            
        # Retrieval format check
        if not isinstance(refs, list):
            print(f"[{q['id']}] Refs is not a list!")
            malformed_telemetry += 1
            
        for ref in refs:
            if "doc_id" not in ref or "page_numbers" not in ref:
                print(f"[{q['id']}] Ref missing doc_id or page_numbers: {ref}")
                malformed_telemetry += 1
                
        # Type validation (S_det)
        if ans is None or ans == "There is no information on this question in the provided documents.":
            missing_answers += 1
            if len(refs) > 0:
                print(f"[{q['id']}] Absent answer but has refs: {refs}")
                malformed_telemetry += 1
        else:
            if t == "boolean":
                if not isinstance(ans, bool):
                    print(f"[{q['id']}] Boolean type mismatch: {ans} (type: {type(ans)})")
                    type_errors += 1
            elif t == "number":
                if not isinstance(ans, (int, float)):
                    print(f"[{q['id']}] Number type mismatch: {ans} (type: {type(ans)})")
                    type_errors += 1
            elif t == "name":
                if not isinstance(ans, str):
                    print(f"[{q['id']}] Name type mismatch: {ans} (type: {type(ans)})")
                    type_errors += 1
            elif t == "names":
                if not isinstance(ans, list) or not all(isinstance(x, str) for x in ans):
                    print(f"[{q['id']}] Names type mismatch: {ans} (type: {type(ans)})")
                    type_errors += 1
            elif t == "date":
                if not isinstance(ans, str) or not re.match(r"^\d{4}-\d{2}-\d{2}$", ans):
                    print(f"[{q['id']}] Date type/format mismatch: {ans}")
                    type_errors += 1
            elif t == "free_text":
                if not isinstance(ans, str):
                    print(f"[{q['id']}] Free_text type mismatch: {ans} (type: {type(ans)})")
                    type_errors += 1
                elif len(ans) > 280:
                    print(f"[{q['id']}] Free_text too long (>280 chars): {len(ans)}")
                    fluff_errors += 1

    print("=== Validation Results ===")
    print(f"Malformed Telemetry (T score risk): {malformed_telemetry}")
    print(f"Deterministic Type Errors (S_det risk): {type_errors}")
    print(f"Free Text Length Errors (S_asst fluff risk): {fluff_errors}")
    print(f"Safely Identified Unanswerable Questions: {missing_answers}")
    
    if malformed_telemetry == 0 and type_errors == 0 and fluff_errors == 0:
        print("\n✅ All validation checks passed perfectly!")
    else:
        print("\n❌ Validation found issues.")

if __name__ == "__main__":
    validate_results("offline_eval_results.json")
