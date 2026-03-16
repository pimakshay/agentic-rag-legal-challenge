"""Smoke test: run retrieval pipeline without submission or API download.

- Uses a single-doc loader to avoid API rate limits on trial keys.
- Set LEGAL_RAG_SMOKE_FULL=1 to build the full index (may hit rate limits).
- Set LEGAL_RAG_SMOKE_MOCK_LLM=1 to skip real LLM call (no OpenAI/OpenRouter key needed).
- Prints end-to-end timings (index build, retrieve+answer, total) and optional relevance.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from arlc import get_config
from retrieval import (
    CohereReranker,
    IngestedCorpusLoader,
    LegalHybridRAGPipeline,
    VoyageReranker,
)


def _smoke_loader(ingest_root: Path, docs_root: Path, max_docs: int = 1):
    """Loader that returns at most max_docs for smoke test (fewer embedding calls)."""
    base = IngestedCorpusLoader(ingest_root=ingest_root, docs_root=docs_root)
    full = base.load_corpus()
    limited = full[:max_docs]
    return limited


def build_pipeline(use_smoke_loader: bool = True, mock_llm: bool = False):
    from langchain_cohere import CohereEmbeddings
    from langchain_core.language_models.chat_models import BaseChatModel
    from langchain_core.messages import AIMessage
    from langchain_core.outputs import ChatGeneration, ChatResult
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings

    CONFIG = get_config()

    if mock_llm:
        # No API key needed; just returns a fixed response so retrieval path is exercised.
        class _MockLLM(BaseChatModel):
            def _generate(self, messages, stop=None, run_manager=None, **kwargs):
                return ChatResult(generations=[ChatGeneration(message=AIMessage(content="true"))])

            @property
            def _llm_type(self) -> str:
                return "mock"
        llm = _MockLLM()
    else:
        llm = ChatOpenAI(
            model=CONFIG.llm_model,
            temperature=0.0,
            openai_api_key=CONFIG.get_llm_api_key(),
            openai_api_base=CONFIG.llm_api_base,
        )

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

    if CONFIG.voyage_api_key:
        reranker = VoyageReranker(api_key=CONFIG.voyage_api_key)
    elif CONFIG.cohere_api_key:
        reranker = CohereReranker(api_key=CONFIG.cohere_api_key)
    else:
        from retrieval.utils.rerankers import MiniLMReranker

        reranker = MiniLMReranker()

    ingest_root = ROOT_DIR / "ingestion" / "docs_corpus_ingest_result"
    docs_root = ROOT_DIR / CONFIG.docs_dir

    pipeline = LegalHybridRAGPipeline(
        llm=llm,
        embedding_model=embeddings,
        ingest_root=str(ingest_root),
        docs_root=str(docs_root),
        enable_reranking=True,
        reranker=reranker,
    )

    if use_smoke_loader:
        # Inject a single-doc corpus so build_indexes only embeds one document
        pipeline.source_documents = _smoke_loader(ingest_root, docs_root, max_docs=1)
    return pipeline


def _eval_relevance(question: str, result) -> None:
    """Print relevance signal: RAGAS context relevance if available, else context preview."""
    contexts = [str(getattr(doc, "page_content", "") or "").strip() for doc in result.supporting_docs]
    contexts = [c for c in contexts if c]
    if not contexts:
        print("Relevance: no retrieved context")
        return

    try:
        from ragas import evaluate
        from ragas.metrics import context_relevance
        from datasets import Dataset

        data = {
            "question": [question],
            "contexts": [contexts],
        }
        ds = Dataset.from_dict(data)
        out = evaluate(ds, metrics=[context_relevance])
        score = out["context_relevance"]
        if isinstance(score, (list, tuple)):
            score = score[0] if score else 0.0
        print(f"Relevance (RAGAS context_relevance): {float(score):.3f}")
    except ImportError:
        # No ragas: show short context preview and a crude keyword overlap hint
        preview_len = 120
        for i, doc in enumerate(result.supporting_docs[:3], 1):
            text = str(getattr(doc, "page_content", "") or "").strip().replace("\n", " ")
            print(f"  Context[{i}] preview: {text[:preview_len]}...")
        q_words = set(question.lower().split()) if question else set()
        overlap = sum(1 for w in q_words if any(w in c.lower() for c in contexts))
        total = len(q_words) or 1
        print(f"  Relevance hint: {overlap}/{total} question words in context (pip install ragas for LLM-based score)")
    except Exception as e:
        print(f"Relevance eval skipped: {e}")


def main() -> None:
    use_full_index = os.environ.get("LEGAL_RAG_SMOKE_FULL", "").strip() == "1"
    use_smoke_loader = not use_full_index
    mock_llm = os.environ.get("LEGAL_RAG_SMOKE_MOCK_LLM", "").strip() == "1"

    questions_path = ROOT_DIR / "public_dataset" / "questions.json"
    if not questions_path.exists():
        print("Using hardcoded question (no public_dataset/questions.json)")
        question_item = {
            "id": "smoke-test-1",
            "question": "Under Article 8(1) of the Operating Law 2018, is a person permitted to operate or conduct business in or from the DIFC without being incorporated?",
            "answer_type": "boolean",
        }
    else:
        questions = json.loads(questions_path.read_text(encoding="utf-8"))
        question_item = questions[0]
        print(f"Using first question from {questions_path.name}: {question_item['id']}")

    if mock_llm:
        print("Using mock LLM (LEGAL_RAG_SMOKE_MOCK_LLM=1); no LLM API key required.")
    print("Building pipeline...")
    pipeline = build_pipeline(use_smoke_loader=use_smoke_loader, mock_llm=mock_llm)

    t0 = time.perf_counter()
    if use_smoke_loader:
        print("Building indexes (1 doc for smoke test; use LEGAL_RAG_SMOKE_FULL=1 for full)...")
    else:
        print("Building indexes (full corpus)...")
    pipeline.build_indexes()
    t_index = time.perf_counter()
    index_build_ms = int((t_index - t0) * 1000)

    print("Answering one question...")
    result = pipeline.answer_question(question_item)
    t_answer = time.perf_counter()
    answer_ms = int((t_answer - t_index) * 1000)
    total_ms = int((t_answer - t0) * 1000)

    print("Answer:", result.answer)
    print("Retrieval refs:", len(result.retrieval_refs), "doc(s)")

    print("--- Timings (ms) ---")
    print(f"  index_build_ms: {index_build_ms}")
    print(f"  retrieve_and_answer_ms: {answer_ms}")
    print(f"  total_smoke_ms: {total_ms}")

    print("--- Relevance ---")
    _eval_relevance(question_item.get("question", ""), result)

    print("Smoke test OK.")


if __name__ == "__main__":
    main()
