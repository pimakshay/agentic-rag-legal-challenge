# Legal Hybrid RAG Design

## Overview
This retrieval stack is tailored to the Agentic RAG Legal Challenge and uses the ingest pipeline output as the single source of retrieval text. It avoids reparsing PDFs during answer generation, keeps challenge `doc_id` values aligned with PDF stems and ingest directory names, and treats metadata as a first-class retrieval signal rather than a reporting artifact.

## Architecture
- `IngestedCorpusLoader` loads one LangChain `Document` per source file from `metadata.json`, `content_list.json`, and `structure.json`.
- `ingestion/ingest.py` prepares those artifacts offline by parsing PDFs, inferring structure, and extracting legal metadata.
- `LegalIngestChunker` converts each loaded source document into:
  - `title_page` chunks for parties, issue dates, and law identifiers
  - `section` chunks for structure-aware retrieval
  - `page_anchor` chunks for page-specific questions and structure fallbacks
- `LegalQuestionRouter` detects case IDs, article references, law names/numbers, and page-specific cues before retrieval.
- `LegalHybridRAGPipeline` builds either an optional `Turbopuffer` hybrid index or local fallback dense plus BM25 indexes, applies route-aware filtering, fuses dense and sparse retrieval, optionally reranks, and returns typed answers with exact telemetry refs.

## Ingest Loader Contract
`IngestedCorpusLoader.load_corpus(ingest_root, docs_root) -> list[Document]`

Each returned source document contains:
- `doc_id`: canonical PDF stem / challenge document ID
- `source`, `source_path`
- legal metadata from `metadata.json`
- `doc_type`: `case`, `law`, or `unknown`
- `structure_available`
- `blocks`: ordered ingest blocks with `block_index`, `page_number`, `text`, `type`, `level`

If `structure.json` is missing, the loader synthesizes structure levels from `text_level`, uppercase/title heuristics, and numbering patterns.

## Ingestion and metadata model

The offline ingest pipeline writes three retrieval-facing artifacts per document:
- `*_content_list.json`: normalized text and table blocks
- `*_structure.json`: block-level heading assignments for section-aware chunking
- `*_metadata.json`: legal metadata for routing, filtering, and deterministic answers

Metadata extraction is document-type aware:
- case documents contribute claim numbers, neutral citations, parties, dates, and judge evidence
- law documents contribute official titles, short titles, law numbers, citation aliases, article indexes, and section heading indexes

During load, this metadata is preserved on the source `Document` and later indexed into several lookup maps:
- claim number index
- neutral citation index
- law number index
- law alias index
- article reference index
- case fact index for issue dates, parties, and judges

These indexes allow the pipeline to narrow the search space before retrieval and, in some cases, bypass the LLM entirely for deterministic answers.

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

Chunk construction is intentionally asymmetric:
- `title_page` chunks make first-page legal facts easy to retrieve
- `section` chunks preserve structural boundaries for laws and judgments
- `page_anchor` chunks provide a reliable fallback for page-specific queries

When sent to `Turbopuffer`, list-rich metadata such as `page_numbers` is serialized into primitive values for storage compatibility. The pipeline decodes those values back into lists before answer synthesis and telemetry generation. The local fallback path keeps the LangChain `Document` metadata in memory.

## Routing Strategy
The router is regex-first and optimized for the challenge corpus:
- case IDs: `CA 005/2025`, `CFI 057/2025`, `SCT 295/2025`
- article references: `Article 28(1)`
- law identifiers: names ending in `Law` and explicit `DIFC Law No. X of YYYY`
- page hints: `title page`, `cover page`, `page 2`, `last page`
- comparison/common-entity/common-judge signals

Routing effects:
- case IDs prefilter by `claim_number`
- neutral citations prefilter by citation metadata
- law names/numbers prefilter by legal-title metadata
- article references restrict candidate evidence to matching article spans when possible
- page-specific questions bias to `title_page` or `page_anchor`
- comparison questions retrieve per cited case and then synthesize across the merged evidence

The route plan is resolved before dense and sparse retrieval runs. This keeps candidate sets small, improves lexical recall for exact legal references, and reduces unnecessary reranking work.

## Retrieval and fusion

Retrieval runs in parallel:
- dense retrieval uses the configured embedding model
- sparse retrieval uses a BM25-style lexical query enriched with routed entities such as case IDs, article references, and law names

Backend behavior:
- when `Turbopuffer` is enabled, dense and sparse search run against the remote hybrid store
- otherwise the pipeline uses local normalized embedding vectors for cosine similarity and an in-process BM25 retriever

After retrieval:
- route-aware bias promotes preferred chunk kinds and page hits
- strict filtering enforces page-specific and article-specific constraints when possible
- reciprocal-rank fusion merges dense and sparse rankings

This flow favors legal precision over semantic-only recall and keeps title-page and page-anchor evidence competitive for challenge-style questions.

## Reranking and answer control

Reranking is optional and runs after fusion. The default runner chooses:
1. `VoyageReranker` if `VOYAGE_API_KEY` is present
2. `CohereReranker` if `COHERE_API_KEY` is present
3. `MiniLMReranker` otherwise

If reranking is disabled or raises an exception, the pipeline falls back to fused order without failing the request.

Answer control also includes metadata-driven shortcuts:
- deterministic issue-date comparisons can be answered from indexed case facts
- party-overlap and judge-overlap comparisons can be answered from extracted metadata
- some law-number questions can be answered directly from law metadata

These bypasses reduce latency and preserve grounding when the answer is already available in trusted structured evidence.

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
- Metadata-aware routing and deterministic bypasses improve both speed and grounding for challenge questions that ask for directly stated legal facts.
