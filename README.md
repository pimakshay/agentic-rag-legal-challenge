# Legal Hybrid RAG for the Agentic RAG Legal Challenge

This repository contains a legal-domain hybrid RAG system built for the Agentic RAG Legal Challenge. The pipeline is optimized for grounded answers over heterogeneous PDF corpora, where page-level evidence matters as much as the answer itself.

The implementation is centered on ingest-backed retrieval rather than reparsing PDFs at query time. It turns parsed documents into structured metadata and legal-aware chunks, runs hybrid dense and sparse retrieval, optionally reranks the candidate set, and returns typed answers with telemetry-ready page references.

## Architecture at a glance

```text
PDFs
  -> ingest output
  -> metadata + structure
  -> legal-aware chunking
  -> hybrid retrieval (dense + BM25)
  -> reciprocal-rank fusion
  -> reranking
  -> typed answer synthesis
  -> telemetry with grounded page refs
```

## Core pipeline

### 1. Ingestion

The offline ingestion pipeline lives in `ingestion/ingest.py`. It processes the PDF corpus once and writes normalized JSON artifacts under:

```text
ingestion/docs_corpus_ingest_result/<doc_id>/txt/
```

Each document directory contains:
- `*_content_list.json`: parsed text and table blocks from the PDF
- `*_structure.json`: heading levels for the parsed blocks
- `*_metadata.json`: legal metadata used later during routing and retrieval

The pipeline runs three stages:
- parse the PDF into text and table blocks
- infer document structure for section-aware retrieval
- extract legal metadata for cases and laws

This ingest output is the source of truth for retrieval. The runner does not reparse PDFs during answer generation.

### 2. Metadata extraction

Metadata is not treated as a side artifact. It is part of the retrieval system.

For case judgments, the ingest step extracts fields such as:
- claim number
- neutral citation
- case name
- claimant and defendants
- issue and judgment dates
- judge signals from early-page headers

For laws, it extracts fields such as:
- official title and short title
- law number and year
- official citation
- alias keys
- article index
- section heading index

These fields power fast routing and prefiltering before dense or sparse search runs.

### 3. Chunking

`LegalIngestChunker` converts each ingested source document into multiple chunk types:
- `title_page`: first-page chunks for parties, issue dates, and law identifiers
- `section`: structure-aware chunks built from heading windows
- `page_anchor`: per-page chunks used for page-specific retrieval and fallbacks

The chunker keeps page numbers, block ranges, headings, section paths, and chunk kinds in metadata. This is important for both grounded answer synthesis and challenge telemetry.

### 4. Retrieval

`LegalQuestionRouter` inspects the question before retrieval. It detects:
- case IDs such as `CA 005/2025`
- neutral citations
- article references such as `Article 28(1)`
- law names and law numbers
- title-page, cover-page, explicit page, and last-page intent
- comparison questions across multiple cited cases

`LegalHybridRAGPipeline` then resolves this route against metadata indexes, narrows the candidate document set, and runs retrieval in parallel:
- dense retrieval over embeddings
- sparse retrieval over BM25-style lexical search

The two result sets are fused with reciprocal-rank fusion and then filtered and biased according to the route plan.

### 5. Reranking

Reranking is optional and happens after dense plus sparse fusion.

The current selection order in `examples/legal_hybrid_rag.py` is:
1. `VoyageReranker` when `VOYAGE_API_KEY` is available
2. `CohereReranker` when `COHERE_API_KEY` is available
3. local `MiniLMReranker` fallback otherwise

If reranking is disabled or fails at runtime, the pipeline falls back to the fused retrieval order.

## Indexing and storage

The pipeline supports two retrieval backends:
- optional `Turbopuffer` for hybrid vector plus BM25 retrieval
- local fallback indexes when `Turbopuffer` is disabled

The local fallback path builds:
- normalized dense vectors in memory for cosine similarity
- an in-process BM25 retriever over the legal chunks

`Turbopuffer` is only enabled when Cohere embeddings are active and `TURBOPUFFER_API_KEY` is set. This avoids vector dimension mismatches when switching embedding providers.

## Answering behavior

The runner supports both deterministic and `free_text` answers.

Important behaviors:
- some deterministic questions can be answered directly from metadata without calling the LLM
- comparison questions can retrieve evidence per cited case and then synthesize across the combined evidence
- unanswerable deterministic questions return `null`
- unanswerable `free_text` questions return the standard absence sentence with empty retrieval refs

All grounded references use canonical `doc_id` values and physical 1-based PDF page numbers derived from ingest output.

## Why this design fits the challenge

The evaluation emphasizes grounded retrieval, not just fluent generation. In practice that means:
- exact legal references matter
- title pages and page-specific chunks matter
- metadata-aware routing matters
- offline ingest and indexing are worth the effort because they are outside the scored answer path

This repository reflects that priority order.

## Quick start

1. Configure environment variables in `.env` or your shell:

   ```bash
   export EVAL_API_KEY="..."
   export EVAL_BASE_URL="https://platform.agentic-challenge.ai/api/v1"
   export OPENROUTER_API_KEY="..."
   export OPENAI_API_KEY="..."  # optional for embeddings or direct OpenAI usage
   export COHERE_API_KEY="..."  # optional for embeddings or reranking
   export VOYAGE_API_KEY="..."  # optional for reranking
   export USE_COHERE_EMBEDDINGS=1
   ```

2. Run ingestion:

   ```bash
   cd ingestion
   python ingest.py
   ```

   Ensure the ingest step is run against the same PDF corpus that the legal runner will validate through `DOCS_DIR`.

3. Run the legal hybrid pipeline:

   ```bash
   cd ..
   python examples/legal_hybrid_rag.py
   ```

The runner will:
- download questions
- use the local PDF corpus already present under `DOCS_DIR`
- validate ingest coverage for the PDF corpus
- build indexes from ingest output
- answer questions with telemetry
- write `submission.json`
- create `code_archive.zip`

## Key configuration

General runner configuration is loaded through `arlc.get_config()`.

Important environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `EVAL_API_KEY` | — | Evaluation API key |
| `EVAL_BASE_URL` | platform URL | Evaluation API base URL |
| `OPENROUTER_API_KEY` | — | LLM access through OpenRouter |
| `OPENAI_API_KEY` | — | OpenAI access for LLMs or embeddings |
| `COHERE_API_KEY` | — | Cohere access for embeddings or reranking |
| `VOYAGE_API_KEY` | — | Voyage access for reranking |
| `LLM_MODEL` | `gpt-5-mini` | Chat model used for answer synthesis |
| `EMBEDDING_MODEL` | `text-embedding-3-small` | OpenAI embedding model name |
| `USE_COHERE_EMBEDDINGS` | `0` | Enable Cohere embeddings |
| `USE_OPENAI_EMBEDDINGS` | `0` | Enable OpenAI embeddings |
| `TURBOPUFFER_API_KEY` | — | Enables the Turbopuffer retrieval backend when Cohere embeddings are active |
| `LEGAL_HYBRID_SKIP_INDEXING` | `0` | Reuse an existing Turbopuffer namespace without re-indexing |
| `LEGAL_HYBRID_ENABLE_RERANK` | `1` | Enable or disable reranking |
| `DOCS_DIR` | `docs_corpus` | Local PDF corpus directory |
| `SUBMISSION_PATH` | `submission.json` | Output submission path |
| `CODE_ARCHIVE_PATH` | `code_archive.zip` | Output code archive path |

## Repository map

- `examples/legal_hybrid_rag.py`: main legal hybrid runner
- `retrieval/legal_hybrid_rag_pipeline.py`: routing, retrieval, fusion, reranking, and answer logic
- `retrieval/chunkers/legal_ingest_chunker.py`: structure-aware legal chunking
- `retrieval/loaders/ingested_corpus_loader.py`: ingest-backed document loader
- `ingestion/ingest.py`: offline parsing, structure analysis, and metadata extraction
- `docs/legal_hybrid_rag_design.md`: architecture notes for the legal retrieval stack
- `docs/legal_hybrid_rag_runbook.md`: operational guide for running the pipeline

## Challenge context

This repo still includes the challenge client and submission helpers:
- `arlc/` handles downloads, submission building, and telemetry structures
- `API.md` documents the platform API
- `EVALUATION.md` documents the grounding-first scoring rules

The project can be understood as a concrete legal RAG system built on top of that submission framework.
