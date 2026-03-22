"""Free-text prompt builders for the legal challenge pipeline."""

from __future__ import annotations

import re
from typing import Sequence

from langchain_core.documents import Document


FREE_TEXT_ABSENCE_STATEMENT = "There is no information on this question in the provided documents."
FREE_TEXT_SUBTYPE_DIRECT = "direct"
FREE_TEXT_SUBTYPE_MULTI_PART = "multi_part"
FREE_TEXT_SUBTYPE_ABSENCE_OR_PARTIAL = "absence_or_partial"

_ADVERSARIAL_PATTERNS = (
    "jury",
    "miranda",
    "plea bargain",
    "parole hearing",
    "parole hearings",
    "fifth amendment",
    "grand jury",
    "district attorney",
    "bail bond",
    "probation officer",
    "criminal sentencing",
    "habeas corpus",
    "supreme court of the united states",
    "federal court",
    "solar system",
    "largest planet",
    "speed of light",
    "vacuum",
    "smallest country",
    "square root",
    "photosynthesis",
    "plants absorb",
    "first modern olympic games",
    "olympic games held",
    "average distance from the earth to the moon",
    "distance from the earth to the moon",
    "distance to the moon",
)


def detect_free_text_subtype(question_text: str, supporting_docs: Sequence[Document]) -> str:
    """Classify free-text questions for prompt specialization."""

    lowered = " ".join((question_text or "").lower().split())
    context = _context_text(supporting_docs)

    article_count = len(re.findall(r"\barticle\s+\d+(?:\([^)]+\))*", lowered))
    law_count = len(re.findall(r"\bdifc law no\.?\s+\d+\s+of\s+\d{4}\b", lowered))
    multi_part_markers = (
        article_count > 1
        or law_count > 1
        or ("which specific" in lowered and "laws" in lowered)
        or "what are the" in lowered
        or "retention periods" in lowered
        or "respectively" in lowered
        or " and what " in lowered
    )

    if multi_part_markers:
        return FREE_TEXT_SUBTYPE_MULTI_PART

    if any(phrase in lowered for phrase in (
        "maximum fine", "maximum penalty", "maximum sentence", "maximum punishment",
    )):
        return FREE_TEXT_SUBTYPE_ABSENCE_OR_PARTIAL

    if "any information about" in lowered or any(pattern in lowered for pattern in _ADVERSARIAL_PATTERNS):
        return FREE_TEXT_SUBTYPE_ABSENCE_OR_PARTIAL

    if _looks_context_limited(lowered, context):
        return FREE_TEXT_SUBTYPE_ABSENCE_OR_PARTIAL

    return FREE_TEXT_SUBTYPE_DIRECT


def build_free_text_prompt(
    question_text: str,
    supporting_docs: Sequence[Document],
    subtype: str,
    absence_statement: str = FREE_TEXT_ABSENCE_STATEMENT,
) -> str:
    """Build the full free-text prompt for the legal challenge pipeline."""

    context = _render_context(supporting_docs)
    return (
        "You answer legal challenge questions using only the provided context.\n"
        "Start with the direct answer, not setup or meta-commentary.\n"
        "Never exceed 280 characters. Prefer one compact sentence and only the minimum words needed.\n"
        "If the answer would be long, compress aggressively by removing background and restating only the dispositive fact.\n"
        "Optimize for correctness, completeness, confidence calibration, and clarity.\n"
        "Avoid phrases like 'provided excerpts', 'context lacks', or similar unless the answer is genuinely unsupported.\n"
        f"{_subtype_instruction(subtype, absence_statement)}\n\n"
        f"Question: {question_text}\n\n"
        f"Context:\n{context}\n\n"
        "Respond in the following format exactly:\n"
        "Answer: <your answer>\n"
        "Indices: <comma-separated list of [Context N] indices used, e.g. 1, 3. Empty if not found>"
    )


def build_free_text_retry_prompt(
    question_text: str,
    supporting_docs: Sequence[Document],
    previous_answer: str,
    absence_statement: str = FREE_TEXT_ABSENCE_STATEMENT,
) -> str:
    """Build a rewrite prompt for overlong free-text answers."""

    context = _render_context(supporting_docs)
    return (
        "You are rewriting a legal challenge answer using only the provided context.\n"
        "Keep the meaning and grounding, but rewrite it so the answer is 280 characters or fewer.\n"
        "Use one compact sentence when possible and preserve the direct answer.\n"
        f"If the context does not support any answer, return exactly: {absence_statement}\n\n"
        f"Question: {question_text}\n"
        f"Previous answer: {previous_answer}\n\n"
        f"Context:\n{context}\n\n"
        "Respond in the following format exactly:\n"
        "Answer: <rewritten answer>\n"
        "Indices: <comma-separated list of [Context N] indices used, e.g. 1, 3. Empty if not found>"
    )


def _subtype_instruction(subtype: str, absence_statement: str) -> str:
    normalized = (subtype or "").strip().lower()
    if normalized == FREE_TEXT_SUBTYPE_MULTI_PART:
        return (
            "Cover every requested part if supported. Use terse labels or separators when helpful. "
            "Compress by removing background, not by dropping requested facts. "
            "If one requested part is unsupported, answer the supported parts first and end with a brief unsupported marker."
        )
    if normalized == FREE_TEXT_SUBTYPE_ABSENCE_OR_PARTIAL:
        return (
            f"If the context does not support any answer, return exactly: {absence_statement} "
            "If the context supports only part of the answer, give the supported fact first and briefly mark the unsupported remainder. "
            "Do not speculate or invent missing details."
        )
    return (
        "Answer in one compact sentence where possible. Prefer the dispositive fact first, such as the ruling, result, entity, date, or fine amount. "
        "Avoid background clauses that are not needed to answer the question."
    )


def _render_context(supporting_docs: Sequence[Document]) -> str:
    chunks = []
    for index, doc in enumerate(supporting_docs, start=1):
        metadata = doc.metadata or {}
        pages = metadata.get("page_numbers", [])
        chunks.append(
            "\n".join(
                [
                    f"[Context {index}]",
                    f"doc_id={metadata.get('doc_id', '')}",
                    f"claim_number={metadata.get('claim_number', '')}",
                    f"heading={metadata.get('heading', '')}",
                    f"pages={pages}",
                    doc.page_content,
                ]
            )
        )
    return "\n\n".join(chunks)


def _context_text(supporting_docs: Sequence[Document]) -> str:
    return " ".join(doc.page_content.lower() for doc in supporting_docs if doc.page_content).strip()


_CONTEXT_CHECK_FILLER = frozenset({
    "the", "and", "for", "are", "but", "not", "you", "all", "can", "had",
    "was", "one", "our", "has", "how", "who", "what", "which", "when",
    "where", "why", "does", "did", "this", "that", "with", "from", "have",
    "they", "been", "will", "each", "make", "than", "them", "into",
    "case", "document", "page", "based", "according", "under", "about",
    "any", "information", "provide", "there", "also", "other", "between",
    "law", "legal", "difc", "upon",
})


def _looks_context_limited(question_text: str, context_text: str) -> bool:
    if not context_text:
        return True
    if len(context_text.strip()) < 100:
        return True
    if "retention periods" in question_text:
        return question_text.count("article") > context_text.count("article")

    question_words = {
        w for w in re.findall(r"[a-z]{3,}", question_text) if w not in _CONTEXT_CHECK_FILLER
    }
    if len(question_words) < 2:
        return False
    found = sum(1 for w in question_words if w in context_text)
    return found / len(question_words) < 0.25
