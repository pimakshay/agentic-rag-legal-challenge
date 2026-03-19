# Legal Hybrid RAG Runbook

## 1. What this runbook covers

This runbook describes how to operate the ingest-backed legal hybrid RAG pipeline end to end:
- generate ingest output from the PDF corpus
- understand which ingest artifacts are required at runtime
- configure embeddings, reranking, and optional `Turbopuffer`
- run the legal hybrid submission pipeline

The main entrypoints are:
- `ingestion/ingest.py` for offline ingestion
- `examples/legal_hybrid_rag.py` for retrieval, answer generation, and submission packaging

## 2. Generate ingest output

Run the ingestion pipeline before the legal runner:

```bash
cd ingestion
python ingest.py
```

The ingestion pipeline performs three stages:
1. parse PDFs into normalized text and table blocks
2. infer heading structure for legal-aware chunking
3. extract legal metadata for cases and laws

The checked-in script currently points at `../public_dataset/docs_corpus` by default. For an evaluation run, make sure ingestion is executed against the same corpus that the runner will later validate through `DOCS_DIR`.

Expected output root:

```text
ingestion/docs_corpus_ingest_result/<doc_id>/txt/
```

Required files per document:
- `*_content_list.json`
- `*_metadata.json`

Preferred file:
- `*_structure.json`

Runtime use of each file:
- `*_content_list.json`: source text and table blocks used to rebuild documents
- `*_metadata.json`: case and law metadata used for routing, filtering, and deterministic bypasses
- `*_structure.json`: heading levels used to build `section` chunks; when missing, the loader synthesizes structure heuristically

## 3. Install runner dependencies

Use the dedicated example requirements:

```bash
pip install -r examples/requirements_legal_hybrid_rag.txt
```

## 4. Configure environment

Set the same variables used by the starter-kit examples:

```bash
export EVAL_API_KEY="..."
export EVAL_BASE_URL="https://platform.agentic-challenge.ai/api/v1"
export OPENROUTER_API_KEY="..."
export OPENAI_API_KEY="..."
```

Common runtime toggles:

```bash
export USE_COHERE_EMBEDDINGS=1
export USE_OPENAI_EMBEDDINGS=0
export LEGAL_HYBRID_ENABLE_RERANK=1
export LEGAL_HYBRID_SKIP_INDEXING=0
```

Optional services:
- set `COHERE_API_KEY` to use Cohere embeddings or Cohere reranking
- set `VOYAGE_API_KEY` to use Voyage reranking
- set `TURBOPUFFER_API_KEY` to enable the `Turbopuffer` backend when Cohere embeddings are active

`Turbopuffer` is intentionally disabled when OpenAI embeddings are used, even if `TURBOPUFFER_API_KEY` is set. This prevents vector dimension mismatches between the stored index and the active embedding model.

## 5. How the runtime uses ingest output

At startup, the legal runner:
1. validates that ingest output exists for each local PDF in `DOCS_DIR`
2. loads each document from `content`, `metadata`, and `structure`
3. builds legal chunks:
   - `title_page`
   - `section`
   - `page_anchor`
4. builds retrieval indexes:
   - optional `Turbopuffer` hybrid index, or
   - local dense vector index plus local BM25 retriever
5. answers questions with routing, retrieval, fusion, reranking, and answer synthesis

Why metadata matters at runtime:
- case IDs can prefilter by claim number
- law titles and law numbers can prefilter by alias and citation metadata
- article references can narrow retrieval to matching spans
- some deterministic answers can be returned directly from indexed metadata without an LLM call

## 6. Reranker selection order

When `LEGAL_HYBRID_ENABLE_RERANK=1`, the default runner chooses rerankers in this order:
1. `VoyageReranker`
2. `CohereReranker`
3. local `MiniLMReranker`

Operational notes:
- API rerankers avoid local model load time and usually improve time-to-first-token
- the local MiniLM fallback keeps reranking available without external reranker APIs
- if reranking fails during execution, the pipeline falls back to fused dense plus sparse order

## 7. Run the legal hybrid runner

```bash
python examples/legal_hybrid_rag.py
```

The runner will:
1. Download questions through `arlc`
2. Use the local PDF corpus already present under `DOCS_DIR`
3. Validate that ingest output exists for each PDF stem
4. Build the legal hybrid index from ingest output
5. Answer all questions
6. Save `submission.json`
7. Create `code_archive.zip`
8. Submit both artifacts to the evaluation API

## 8. Inspect artifacts

- `submission.json`: final answer payload
- `code_archive.zip`: code archive for challenge submission
- `questions.json`: downloaded evaluation questions
- `ingestion/docs_corpus_ingest_result/`: offline ingest artifacts reused by the runner

## 9. Fallbacks and failure modes

- Missing ingest output: runner aborts with a clear error asking you to run `ingestion/ingest.py`
- Missing `structure.json`: loader falls back to synthetic structure
- Missing reranker dependencies or reranker runtime failure: retrieval falls back to fused dense/BM25 ranking
- Missing answer evidence: deterministic answers return `null`; `free_text` answers return the standard absence statement
- `Turbopuffer` unavailable or disabled: pipeline falls back to local dense plus BM25 indexes
- `Turbopuffer` metadata limitations: list metadata is serialized before indexing and decoded after retrieval

## 10. Related docs

- `README.md`: project overview and architecture summary
- `docs/legal_hybrid_rag_design.md`: implementation-level retrieval design
- `EVALUATION.md`: challenge scoring, especially grounding and telemetry
