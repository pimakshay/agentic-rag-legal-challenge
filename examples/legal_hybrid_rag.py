"""Legal hybrid RAG runner aligned with the starter-kit submission flow."""

from __future__ import annotations

import logging
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

_LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").strip().upper()
logging.basicConfig(
    level=getattr(logging, _LOG_LEVEL, logging.INFO),
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

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
        model_kwargs={"reasoning_effort": "low"},
    )

    # Use Cohere for embeddings when COHERE_API_KEY is set (rate-limited to 90/60s)
    embedding_provider: str
    embedding_model_name: str
    if CONFIG.cohere_api_key and CONFIG.use_cohere_embeddings:
        from retrieval.utils.cohere_rate_limit import RateLimitedCohereEmbeddings, get_cohere_rate_limiter

        base_embeddings = CohereEmbeddings(
            model="embed-english-v3.0",
            cohere_api_key=CONFIG.cohere_api_key,
        )
        embeddings = RateLimitedCohereEmbeddings(base_embeddings, limiter=get_cohere_rate_limiter())
        embedding_provider = "cohere"
        embedding_model_name = "embed-english-v3.0 (rate-limited wrapper)"
    elif CONFIG.openai_api_key and CONFIG.use_openai_embeddings:
        embeddings = OpenAIEmbeddings(
            model=CONFIG.embedding_model,
            openai_api_key=CONFIG.get_embedding_api_key(),
            openai_api_base=CONFIG.llm_api_base,
        )
        embedding_provider = "openai"
        embedding_model_name = CONFIG.embedding_model
    else:
        raise ValueError(
            "No embedding provider enabled. Configure USE_COHERE_EMBEDDINGS=1 or USE_OPENAI_EMBEDDINGS=1, "
            "and ensure the corresponding API key is set."
        )

    # Optional: skip re-indexing when Turbopuffer namespace is already populated (query-only).
    skip_indexing_env = os.getenv("LEGAL_HYBRID_SKIP_INDEXING", "0").strip()
    skip_indexing = skip_indexing_env in {"1", "true", "True"}

    # Optional: disable reranking via feature flag (fusion-only retrieval).
    enable_reranking_env = os.getenv("LEGAL_HYBRID_ENABLE_RERANK", "1").strip()
    enable_reranking = enable_reranking_env not in {"0", "false", "False"}

    reranker = None
    reranker_provider = "none"
    reranker_model = "-"
    if enable_reranking:
        # Prefer API reranker (no local compute, better TTFT); fall back to local MiniLM.
        if CONFIG.voyage_api_key:
            reranker = VoyageReranker(api_key=CONFIG.voyage_api_key)
            reranker_provider = "voyage"
            reranker_model = getattr(reranker, "model", getattr(reranker, "model_name", "-"))
        elif CONFIG.cohere_api_key:
            reranker = CohereReranker(api_key=CONFIG.cohere_api_key)
            reranker_provider = "cohere"
            reranker_model = getattr(reranker, "model", getattr(reranker, "model_name", "-"))
        else:
            from retrieval.utils.rerankers import MiniLMReranker

            reranker = MiniLMReranker()
            reranker_provider = "minilm"
            reranker_model = getattr(reranker, "model_name", "-")

    # Turbopuffer is an optional acceleration layer. It must not be used with
    # non-Cohere embedding models because vector dimensionality can mismatch.
    use_turbopuffer = CONFIG.use_cohere_embeddings and bool(os.getenv("TURBOPUFFER_API_KEY"))

    # If your environment keeps TURBOPUFFER_API_KEY set but you switch to OpenAI
    # embeddings, we intentionally disable Turbopuffer to avoid vector dimension mismatch.
    logger.info("=== Legal Hybrid RAG run configuration ===")
    logger.info(f"LLM: ChatOpenAI model={CONFIG.llm_model} base={CONFIG.llm_api_base} (temp=0.0)")
    logger.info(
        "Embeddings: "
        f"provider={embedding_provider} model={embedding_model_name} "
        f"(USE_COHERE_EMBEDDINGS={int(CONFIG.use_cohere_embeddings)} USE_OPENAI_EMBEDDINGS={int(CONFIG.use_openai_embeddings)})"
    )
    logger.info(f"Reranking: enabled={int(enable_reranking)} provider={reranker_provider} model={reranker_model}")
    logger.info(
        "Turbopuffer: enabled="
        f"{int(bool(use_turbopuffer))} "
        f"(TURBOPUFFER_API_KEY={'set' if bool(os.getenv('TURBOPUFFER_API_KEY')) else 'unset'}) "
        f"skip_indexing={int(skip_indexing)}"
    )
    logger.info(f"Routing/docs: docs_dir={CONFIG.docs_dir} ingest_root={ROOT_DIR/'ingestion'/'docs_corpus_ingest_result'}")

    pipeline = LegalHybridRAGPipeline(
        llm=llm,
        embedding_model=embeddings,
        ingest_root=str(ROOT_DIR / "ingestion" / "docs_corpus_ingest_result"),
        docs_root=str(Path(CONFIG.docs_dir)),
        enable_reranking=enable_reranking,
        reranker=reranker,
        use_turbopuffer=use_turbopuffer,
        skip_indexing=skip_indexing,
    )

    # Retrieval defaults that affect runtime cost/recall.
    logger.info(
        "Retrieval params: top_k_docs=%s dense_candidate_k=%s sparse_candidate_k=%s "
        "dense_weight=%.2f sparse_weight=%.2f rrf_k=%s",
        pipeline.top_k_docs,
        pipeline.dense_candidate_k,
        pipeline.sparse_candidate_k,
        pipeline.dense_weight,
        pipeline.sparse_weight,
        pipeline.rrf_k,
    )
    return pipeline


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
    print("Skipping downloading documents...")
    # client.download_documents(CONFIG.docs_dir)
    # print("Documents extracted")

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
        result = pipeline.answer_question(question_item, telemetry_timer=telemetry_timer)
        timing = telemetry_timer.finish()
        ttft_ms = max(0, min(timing.ttft_ms, timing.total_time_ms))
        total_time_ms = max(timing.total_time_ms, ttft_ms)

        prompt = result.debug_metadata.get("prompt", "")
        usage = UsageMetrics(
            input_tokens=count_tokens(tokenizer, prompt),
            output_tokens=count_tokens(tokenizer, result.raw_response or str(result.answer)),
        )
        telemetry = Telemetry(
            timing=TimingMetrics(
                ttft_ms=ttft_ms,
                tpot_ms=timing.tpot_ms,
                total_time_ms=total_time_ms,
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
