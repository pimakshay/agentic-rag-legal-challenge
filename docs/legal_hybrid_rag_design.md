# Legal Hybrid RAG Design

## Overview
This retrieval stack is tailored to the Agentic RAG Legal Challenge and uses the ingest pipeline output as the single source of retrieval text. It avoids reparsing PDFs during answer generation and keeps challenge `doc_id` values aligned with PDF stems and ingest directory names.

## Architecture
- `IngestedCorpusLoader` loads one LangChain `Document` per source file from `metadata.json`, `content_list.json`, and `structure.json`.
- `LegalIngestChunker` converts each loaded source document into:
  - `title_page` chunks for parties, issue dates, and law identifiers
  - `section` chunks for structure-aware retrieval
  - `page_anchor` chunks for page-specific questions and structure fallbacks
- `LegalQuestionRouter` detects case IDs, article references, law names/numbers, and page-specific cues before retrieval.
- `LegalHybridRAGPipeline` builds Chroma and BM25 over the legal chunks, applies route-aware filtering, fuses dense and sparse retrieval, optionally reranks, and returns typed answers with exact telemetry refs.

## Ingest Loader Contract
`IngestedCorpusLoader.load_corpus(ingest_root, docs_root) -> list[Document]`

Each returned source document contains:
- `doc_id`: canonical PDF stem / challenge document ID
- `source`, `source_path`
- legal metadata from `metadata.json`
- `doc_type`: `case_judgment`, `law`, or `unknown`
- `structure_available`
- `blocks`: ordered ingest blocks with `block_index`, `page_number`, `text`, `type`, `level`

If `structure.json` is missing, the loader synthesizes structure levels from `text_level`, uppercase/title heuristics, and numbering patterns.

## Chunk Metadata
Legal chunks carry:
- `chunk_id`
- `doc_id`
- `page_numbers`
- `page_start`, `page_end`
- `block_start`, `block_end`
- `chunk_kind`
- `heading`
- `section_path`
- `level`

When sent to Chroma, list-rich metadata is serialized into primitive values for compatibility. The pipeline decodes `page_numbers` back into lists before answer synthesis and telemetry generation.

## Routing Strategy
The router is regex-first and optimized for the challenge corpus:
- case IDs: `CA 005/2025`, `CFI 057/2025`, `SCT 295/2025`
- article references: `Article 28(1)`
- law identifiers: names ending in `Law` and explicit `DIFC Law No. X of YYYY`
- page hints: `title page`, `cover page`, `page 2`, `last page`
- comparison/common-entity/common-judge signals

Routing effects:
- case IDs prefilter by `claim_number`
- law names/numbers prefilter by legal-title metadata
- page-specific questions bias to `title_page` or `page_anchor`
- comparison questions retrieve per cited case and then synthesize across the merged evidence

## Grounding and Telemetry
- Retrieval refs use canonical `doc_id` values from ingest directory names / PDF stems.
- Page numbers are physical 1-based PDF pages derived from ingest `page_idx + 1`.
- Telemetry includes only pages present in the final supporting chunks passed to answer synthesis.
- Unanswerable deterministic answers return `null` with empty refs.
- Unanswerable `free_text` answers return the standard absence sentence with empty refs.

## Challenge Alignment
- Exact/legal lexical retrieval is favored over semantic-only retrieval.
- Title-page and page-anchor chunks are first-class because many challenge questions target parties, dates, cover pages, and specific page numbers.
- Offline indexing stays outside the scored path; only per-answer telemetry is emitted by the runner.
