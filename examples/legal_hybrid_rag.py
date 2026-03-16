"""Legal hybrid RAG runner aligned with the starter-kit submission flow."""

from __future__ import annotations

import os
from pathlib import Path
import sys
from zipfile import ZIP_DEFLATED, ZipFile

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))

from arlc import (  # noqa: E402
    EvaluationClient,
    SubmissionAnswer,
    SubmissionBuilder,
    Telemetry,
    TelemetryTimer,
    TimingMetrics,
    UsageMetrics,
    get_config,
)
from retrieval import (  # noqa: E402
    CohereReranker,
    LegalHybridRAGPipeline,
    VoyageReranker,
)

CONFIG = get_config()

_EXCLUDE_DIRS = {
    "__pycache__",
    "ingestion",
    "docs_corpus",
    "storage",
    ".venv",
    "venv",
    "env",
    "tmp",
    "code_archive",
    ".git",
    "public_dataset",
    "notebooks",
}
_EXCLUDE_FILES = {
    ".env",
    "submission.json",
    "questions.json",
    "code_archive.zip",
    "*.out",
    "*.zip",
}


def ensure_code_archive(archive_path: Path) -> Path:
    """Archive the repo while excluding runtime artifacts."""
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    archive_resolved = archive_path.resolve()
    with ZipFile(archive_path, "w", compression=ZIP_DEFLATED) as zip_file:
        for file_path in ROOT_DIR.rglob("*"):
            if not file_path.is_file():
                continue
            if file_path.resolve() == archive_resolved:
                continue
            parts = set(file_path.relative_to(ROOT_DIR).parts)
            if parts & _EXCLUDE_DIRS:
                continue
            if file_path.name in _EXCLUDE_FILES:
                continue
            zip_file.write(file_path, file_path.relative_to(ROOT_DIR))
    return archive_path


def count_tokens(tokenizer, text: str) -> int:
    return len(tokenizer.encode(text)) if tokenizer else max(1, len(text.split()))


def build_pipeline():
    from langchain_cohere import CohereEmbeddings
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings

    llm = ChatOpenAI(
        model=CONFIG.llm_model,
        temperature=0.0,
        openai_api_key=CONFIG.get_llm_api_key(),
        openai_api_base=CONFIG.llm_api_base,
    )

    # Use Cohere for embeddings when COHERE_API_KEY is set (rate-limited to 90/60s)
    if CONFIG.cohere_api_key:
        from retrieval.utils.cohere_rate_limit import RateLimitedCohereEmbeddings, get_cohere_rate_limiter

        base_embeddings = CohereEmbeddings(
            model="embed-english-v3.0",
            cohere_api_key=CONFIG.cohere_api_key,
        )
        embeddings = RateLimitedCohereEmbeddings(base_embeddings, limiter=get_cohere_rate_limiter())
    else:
        embeddings = OpenAIEmbeddings(
            model=CONFIG.embedding_model,
            openai_api_key=CONFIG.get_embedding_api_key(),
            openai_api_base=CONFIG.llm_api_base,
        )

    # Optional: skip re-indexing when Turbopuffer namespace is already populated (query-only).
    skip_indexing_env = os.getenv("LEGAL_HYBRID_SKIP_INDEXING", "0").strip()
    skip_indexing = skip_indexing_env in {"1", "true", "True"}

    # Optional: disable reranking via feature flag (fusion-only retrieval).
    enable_reranking_env = os.getenv("LEGAL_HYBRID_ENABLE_RERANK", "1").strip()
    enable_reranking = enable_reranking_env not in {"0", "false", "False"}

    reranker = None
    if enable_reranking:
        # Prefer API reranker (no local compute, better TTFT); fall back to local MiniLM.
        if CONFIG.voyage_api_key:
            reranker = VoyageReranker(api_key=CONFIG.voyage_api_key)
        elif CONFIG.cohere_api_key:
            reranker = CohereReranker(api_key=CONFIG.cohere_api_key)
        else:
            from retrieval.utils.rerankers import MiniLMReranker

            reranker = MiniLMReranker()

    return LegalHybridRAGPipeline(
        llm=llm,
        embedding_model=embeddings,
        ingest_root=str(ROOT_DIR / "ingestion" / "docs_corpus_ingest_result"),
        docs_root=str(Path(CONFIG.docs_dir)),
        enable_reranking=enable_reranking,
        reranker=reranker,
        skip_indexing=skip_indexing,
    )


def validate_ingest_coverage(docs_dir: str | Path, ingest_root: str | Path) -> None:
    docs_root = Path(docs_dir)
    ingest = Path(ingest_root)
    missing = []
    for pdf_path in sorted(docs_root.glob("*.pdf")):
        doc_id = pdf_path.stem
        content_path = ingest / doc_id / "txt" / f"{doc_id}_content_list.json"
        metadata_path = ingest / doc_id / "txt" / f"{doc_id}_metadata.json"
        if not content_path.exists() or not metadata_path.exists():
            missing.append(doc_id)
    if missing:
        raise FileNotFoundError(
            "Missing ingest output for PDF stems: "
            + ", ".join(missing[:10])
            + ". Run ingestion/ingest.py before this runner."
        )


def main() -> None:
    import tiktoken

    client = EvaluationClient.from_env()
    print("Downloading questions...")
    questions = client.download_questions(target_path=CONFIG.questions_path)
    print("Downloading documents...")
    client.download_documents(CONFIG.docs_dir)
    print("Documents extracted")

    ingest_root = ROOT_DIR / "ingestion" / "docs_corpus_ingest_result"
    validate_ingest_coverage(CONFIG.docs_dir, ingest_root)

    print("Building legal hybrid pipeline...")
    pipeline = build_pipeline()
    pipeline.build_indexes()

    tokenizer = None
    try:
        tokenizer = tiktoken.encoding_for_model(CONFIG.llm_model)
    except Exception:
        tokenizer = None

    builder = SubmissionBuilder(
        architecture_summary=(
            "Legal hybrid RAG over ingest outputs with structure-aware chunking, "
            "typed routing, hybrid dense/BM25 retrieval, and starter-kit telemetry."
        )
    )

    print("Answering questions...")
    for index_number, question_item in enumerate(questions, start=1):
        question_id = question_item["id"]
        print(f"[{index_number}/{len(questions)}] {question_id}")

        telemetry_timer = TelemetryTimer()
        result = pipeline.answer_question(question_item)
        telemetry_timer.mark_token()
        timing = telemetry_timer.finish()

        prompt = result.debug_metadata.get("prompt", "")
        usage = UsageMetrics(
            input_tokens=count_tokens(tokenizer, prompt),
            output_tokens=count_tokens(tokenizer, result.raw_response or str(result.answer)),
        )
        telemetry = Telemetry(
            timing=TimingMetrics(
                ttft_ms=timing.ttft_ms,
                tpot_ms=timing.tpot_ms,
                total_time_ms=timing.total_time_ms,
            ),
            retrieval=result.retrieval_refs,
            usage=usage,
            model_name=CONFIG.llm_model,
        )
        builder.add_answer(
            SubmissionAnswer(
                question_id=question_id,
                answer=result.answer,
                telemetry=telemetry,
            )
        )

    submission_path = builder.save(str(CONFIG.submission_path))
    code_archive_path = ensure_code_archive(CONFIG.code_archive_path)
    print("Saved submission.json")
    print(f"Using code archive: {code_archive_path}")
    print("Submitting...")
    response = client.submit_submission(submission_path, code_archive_path)
    print(response)


if __name__ == "__main__":
    main()
