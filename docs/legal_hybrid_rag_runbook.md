# Legal Hybrid RAG Runbook

## 1. Generate ingest output
Run the ingestion pipeline before the legal runner:

```bash
cd ingestion
python ingest.py
```

Expected output root:

```text
ingestion/docs_corpus_ingest_result/<doc_id>/txt/
```

Minimum required files per document:
- `*_content_list.json`
- `*_metadata.json`

Optional but preferred:
- `*_structure.json`

## 2. Install runner dependencies
Use the dedicated example requirements:

```bash
pip install -r examples/requirements_legal_hybrid_rag.txt
```

## 3. Configure environment
Set the same variables used by the starter-kit examples:

```bash
export EVAL_API_KEY="..."
export EVAL_BASE_URL="https://platform.agentic-challenge.ai/api/v1"
export OPENROUTER_API_KEY="..."
export OPENAI_API_KEY="..."
```

## 4. Run the legal hybrid runner

```bash
python examples/legal_hybrid_rag.py
```

The runner will:
1. Download questions and PDFs through `arlc`
2. Validate that ingest output exists for each PDF stem
3. Build the legal hybrid index from ingest output
4. Answer all questions
5. Save `submission.json`
6. Create `code_archive.zip`
7. Submit both artifacts to the evaluation API

## 5. Inspect artifacts
- `submission.json`: final answer payload
- `code_archive.zip`: code archive for challenge submission
- `tmp/legal_hybrid_chroma/`: Chroma persistence directory if persistence is enabled

## 6. Fallbacks and failure modes
- Missing ingest output: runner aborts with a clear error asking you to run `ingestion/ingest.py`
- Missing `structure.json`: loader falls back to synthetic structure
- Missing reranker dependencies: retrieval falls back to fused dense/BM25 ranking
- Missing answer evidence: deterministic answers return `null`; `free_text` answers return the standard absence statement
- Chroma metadata limitations: list metadata is serialized before indexing and decoded after retrieval
