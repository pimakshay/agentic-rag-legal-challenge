"""Typed question routing for the legal challenge."""

from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import List


CASE_ID_PATTERN = re.compile(r"\b(?:CA|CFI|SCT|DEC|ENF|ARB|TCD)\s+\d{3}/\d{4}\b")
NEUTRAL_CITATION_PATTERN = re.compile(r"\[\d{4}\]\s+DIFC\s+(?:CA|CFI|SCT|DEC|ENF|ARB|TCD)\s+\d{3}\b", re.IGNORECASE)
ARTICLE_PATTERN = re.compile(r"\bArticle\s+\d+(?:\([^)]+\))*", re.IGNORECASE)
LAW_NUMBER_PATTERN = re.compile(r"\bDIFC Law No\.?\s+\d+\s+of\s+\d{4}\b", re.IGNORECASE)
CONTEXTUAL_LAW_TITLE_PATTERN = re.compile(
    r"(?:(?:under|according to|pursuant to|of|in|under the|title page of|cover page of)\s+(?:the\s+)?)"
    r"((?:DIFC\s+)?[A-Z][A-Za-z]+(?:\s+[A-Za-z][A-Za-z&,\-]+){0,8}\s+Law(?:\s+\d{4})?)",
    re.IGNORECASE,
)
LAW_ON_TITLE_PATTERN = re.compile(
    r"((?:DIFC\s+)?Law\s+on\s+the\s+[A-Za-z][A-Za-z\s]+?Laws(?:\s+in\s+the\s+DIFC)?(?:\s+\d{4})?)",
    re.IGNORECASE,
)
FALLBACK_LAW_TITLE_PATTERN = re.compile(
    r"\b((?:DIFC\s+)?[A-Z][A-Za-z]+(?:\s+(?:[a-z]{2,3}|[A-Z][A-Za-z&,\-]+)){0,8}\s+Law(?:\s+\d{4})?)\b"
)
EXPLICIT_PAGE_PATTERN = re.compile(r"\bpage\s+(\d+)\b", re.IGNORECASE)
PHRASE_PAGE_PATTERN = re.compile(r"\b(second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth)\s+page\b", re.IGNORECASE)
DICT_PHRASE_PAGE = {
    "second": 2,
    "third": 3,
    "fourth": 4,
    "fifth": 5,
    "sixth": 6,
    "seventh": 7,
    "eighth": 8,
    "ninth": 9,
    "tenth": 10
}

GENERIC_LAW_TITLE_KEYS = {
    "application of the law",
    "application of this law",
    "scope of the law",
    "administration of the law",
    "administration of this law",
    "purpose of this law",
    "purpose of the law",
    "amendment law",
    "difc law",
}


def extract_phrase_page_patern(text: str) -> List[int]:
    matched = PHRASE_PAGE_PATTERN.findall(text)
    return sorted([DICT_PHRASE_PAGE[m] for m in matched])

@dataclass
class RoutePlan:
    """Routing decision for a single question."""

    question_text: str
    answer_type: str
    case_ids: List[str] = field(default_factory=list)
    neutral_citations: List[str] = field(default_factory=list)
    article_refs: List[str] = field(default_factory=list)
    law_title_candidates: List[str] = field(default_factory=list)
    law_numbers: List[str] = field(default_factory=list)
    target_pages: List[int] = field(default_factory=list)
    prefer_last_page: bool = False
    prefer_title_page: bool = False
    comparison_mode: bool = False
    comparison_kind: str | None = None
    common_entity_mode: bool = False
    page_specific_mode: bool = False
    preferred_chunk_kinds: List[str] = field(default_factory=list)


class LegalQuestionRouter:
    """Regex-first question router for legal retrieval."""

    def route(self, question_text: str, answer_type: str) -> RoutePlan:
        lowered = question_text.lower()
        case_ids = CASE_ID_PATTERN.findall(question_text)
        neutral_citations = [match.strip() for match in NEUTRAL_CITATION_PATTERN.findall(question_text)]
        article_refs = ARTICLE_PATTERN.findall(question_text)
        law_numbers = [match.strip() for match in LAW_NUMBER_PATTERN.findall(question_text)]
        law_title_candidates = self._extract_law_title_candidates(question_text)
        target_pages = [int(match) for match in EXPLICIT_PAGE_PATTERN.findall(question_text)]
        target_pages = target_pages + [page_num for page_num in extract_phrase_page_patern(question_text) if page_num not in target_pages]

        comparison_kind = self._detect_comparison_kind(lowered, answer_type)
        prefer_title_page = any(phrase in lowered for phrase in ["title page", "cover page", "first page"])
        if comparison_kind == "issue_date":
            prefer_title_page = True        
        prefer_last_page = "last page" in lowered
        page_specific_mode = prefer_title_page or prefer_last_page or bool(target_pages)
        comparison_mode = len(case_ids) > 1 or any(token in lowered for token in ["between", "both cases", "both case", "common to both", "party to both"])
        common_entity_mode = any(
            token in lowered
            for token in ["same legal", "same party", "common", "shared", "involve any of the same", "judge involved in both", "party to both"]
        )

        preferred_chunk_kinds: List[str] = []
        if prefer_title_page:
            preferred_chunk_kinds.append("title_page")
        if page_specific_mode:
            preferred_chunk_kinds.append("page_anchor")
        if not preferred_chunk_kinds:
            preferred_chunk_kinds.append("section")

        return RoutePlan(
            question_text=question_text,
            answer_type=answer_type,
            case_ids=case_ids,
            neutral_citations=neutral_citations,
            article_refs=article_refs,
            law_title_candidates=law_title_candidates,
            law_numbers=law_numbers,
            target_pages=target_pages,
            prefer_last_page=prefer_last_page,
            prefer_title_page=prefer_title_page,
            comparison_mode=comparison_mode,
            comparison_kind=comparison_kind,
            common_entity_mode=common_entity_mode,
            page_specific_mode=page_specific_mode,
            preferred_chunk_kinds=preferred_chunk_kinds,
        )

    def _detect_comparison_kind(self, lowered_question: str, answer_type: str) -> str | None:
        normalized_answer_type = answer_type.lower()
        if normalized_answer_type == "date" and (
            "date of issue" in lowered_question or "issue date" in lowered_question
        ):
            return "issue_date"
        if normalized_answer_type == "boolean" and "judge" in lowered_question and any(
            phrase in lowered_question
            for phrase in [
                "both case",
                "both cases",
                "both ca",
                "in both",
                "participated in both",
                "presided over both",
                "judge involved in both",
                "judge who presided over both",
            ]
        ):
            return "judge_overlap"
        if normalized_answer_type == "boolean" and "judge" not in lowered_question and any(
            phrase in lowered_question
            for phrase in [
                "same party",
                "common to both",
                "party to both",
                "involve any of the same",
                "same legal",
            ]
        ):
            return "party_overlap"
        if normalized_answer_type == "name" and any(
            phrase in lowered_question
            for phrase in [
                "earlier issue date",
                "earlier date of issue",
                "issued earlier",
                "earlier issue",
            ]
        ):
            return "issue_date"
        return None

    def _extract_law_title_candidates(self, question_text: str) -> List[str]:
        candidates: List[str] = []
        for pattern in [LAW_ON_TITLE_PATTERN, CONTEXTUAL_LAW_TITLE_PATTERN, FALLBACK_LAW_TITLE_PATTERN]:
            for match in pattern.findall(question_text):
                normalized = self._normalize_title_candidate(match)
                if normalized and normalized not in candidates:
                    candidates.append(normalized)
        return candidates

    def _normalize_title_candidate(self, value: str) -> str:
        normalized = " ".join(value.split()).strip(" ,.-")
        if not normalized:
            return ""
        key = re.sub(r"[^a-z0-9]+", " ", normalized.lower()).strip()
        key = re.sub(r"^(?:under|according to|pursuant to|of|in|title page of|cover page of|the)\s+", "", key).strip()
        key = re.sub(r"^(?:the)\s+", "", key).strip()
        if key in GENERIC_LAW_TITLE_KEYS:
            return ""
        if not re.search(r"\bLaw\b", normalized, re.IGNORECASE):
            return ""
        if any(generic in key for generic in GENERIC_LAW_TITLE_KEYS):
            return ""
        return key
