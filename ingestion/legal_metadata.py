from __future__ import annotations

import ast
import re
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


CASE_NUMBER_RE = re.compile(r"\b(?:CA|CFI|SCT|DEC|ENF|ARB|TCD)\s+\d{3}/\d{4}\b", re.IGNORECASE)
NEUTRAL_CITATION_RE = re.compile(r"\[\d{4}\]\s+DIFC\s+(?:CA|CFI|SCT|DEC|ENF|ARB|TCD)\s+\d{3}\b", re.IGNORECASE)
LAW_NUMBER_RE = re.compile(r"DIFC\s+LAW\s+NO\.?\s*(\d+)\s+OF\s+(\d{4})", re.IGNORECASE)
MONTH_NAME_RE = (
    r"January|February|March|April|May|June|July|August"
    r"|September|October|November|December"
)
TEXTUAL_DATE_RE = re.compile(
    rf"\b(?:{MONTH_NAME_RE})\s+\d{{1,2}},\s+\d{{4}}\b|\b\d{{1,2}}\s+(?:{MONTH_NAME_RE})\s+\d{{4}}\b",
    re.IGNORECASE,
)
CONSOLIDATED_VERSION_RE = re.compile(
    r"Consolidated\s+Version(?:\s+No\.?\s*\d+)?\s*\(([^)]+)\)",
    re.IGNORECASE,
)
TITLE_FROM_TEXT_RE = re.compile(
    r"cited\s+as\s+the\s+[\"“']([^\"”']+?)[\"”']",
    re.IGNORECASE,
)
QUOTED_TITLE_RE = re.compile(r"[\"“']([^\"”']+Law(?:\s+\d{4})?)[\"”']")
SECTION_HEADING_RE = re.compile(r"^(?P<number>\d+[A-Z]?)\.\s+(?P<title>.+)$")
NUMERIC_SUBSECTION_RE = re.compile(r"^\((\d+[A-Z]?)\)\s*")
ALPHA_SUBSECTION_RE = re.compile(r"^\(([a-z])\)\s*", re.IGNORECASE)
ROMAN_SUBSECTION_RE = re.compile(r"^\(([ivxlcdm]+)\)\s*", re.IGNORECASE)

GENERIC_ALIAS_HEADINGS = {
    "application of the law",
    "application of this law",
    "scope of the law",
    "administration of the law",
    "administration of this law",
    "purpose of the law",
    "purpose of this law",
    "amendment law",
    "difc law",
    "title",
}
STOPWORDS = {"a", "an", "and", "as", "at", "by", "for", "in", "of", "on", "or", "the", "to", "with"}
UPPER_EXCEPTIONS = {"DIFC", "UAE", "DFSA", "DIFCA", "LLP", "PSC", "PJSC", "DMCC", "KC"}

CASE_FIELDS = (
    "doc_type",
    "case_name",
    "case_name_normalized",
    "neutral_citation",
    "neutral_citation_normalized",
    "claim_number",
    "claim_number_normalized",
    "court",
    "court_division",
    "hearing_date",
    "judgment_date",
    "judgment_release_date",
    "claimant",
    "defendants",
    "claimant_counsel",
    "defendant_counsel",
    "claimant_law_firm",
    "defendant_law_firm",
    "title_page_number",
    "last_page_number",
    "page_count",
)
LAW_FIELDS = (
    "doc_type",
    "official_title",
    "official_title_normalized",
    "short_title",
    "short_title_normalized",
    "law_number",
    "law_year",
    "official_citation",
    "official_citation_normalized",
    "citation_title_from_text",
    "consolidated_version_date",
    "amending_laws",
    "administering_authority",
    "jurisdiction",
    "title_page_number",
    "last_page_number",
    "page_count",
    "alias_keys",
    "article_index",
    "section_heading_index",
)

LAW_METADATA_PROMPT = """You extract metadata for a DIFC law document from curated parse output.

Rules:
- Only use the provided title-page lines, early structured headings, and title/body snippets.
- Return JSON only.
- Do not invent missing values.
- Preserve law names as written, but normalize obvious OCR spacing only if the intended title is explicit.

Return this schema:
{{
  "official_title": "",
  "short_title": "",
  "law_number": "",
  "law_year": "",
  "official_citation": "",
  "citation_title_from_text": "",
  "consolidated_version_date": "",
  "amending_laws": [],
  "administering_authority": "",
  "jurisdiction": ""
}}

Input:
{payload}
"""

CASE_METADATA_PROMPT = """You extract metadata for a DIFC court case document from curated title-page parse output.

Rules:
- Only use the provided title-page lines and early structured blocks.
- Return JSON only.
- Do not invent missing values.
- Preserve party names and court wording as written, but normalize obvious OCR spacing only if the intended text is explicit.
- `defendants`, `claimant_counsel`, and `defendant_counsel` must be arrays.

Return this schema:
{{
  "case_name": "",
  "neutral_citation": "",
  "court": "",
  "court_division": "",
  "claim_number": "",
  "hearing_date": "",
  "judgment_date": "",
  "judgment_release_date": "",
  "claimant": "",
  "defendants": [],
  "claimant_counsel": [],
  "defendant_counsel": [],
  "claimant_law_firm": "",
  "defendant_law_firm": ""
}}

Input:
{payload}
"""


def parse_llm_json(text: str) -> Dict[str, Any]:
    cleaned = (text or "").strip()
    if not cleaned:
        return {}
    cleaned = re.sub(r"<think>.*?</think>", "", cleaned, flags=re.DOTALL).strip()
    cleaned = cleaned.replace("```json", "").replace("```", "").strip()
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start < 0 or end < 0 or end <= start:
        return {}
    cleaned = cleaned[start : end + 1]
    cleaned = cleaned.replace("NOT_FOUND", '""')
    try:
        value = ast.literal_eval(cleaned)
    except Exception:
        return {}
    return value if isinstance(value, dict) else {}


def normalize_whitespace(value: str) -> str:
    return re.sub(r"\s+", " ", (value or "").strip())


def clean_ocr_spacing(value: str) -> str:
    text = normalize_whitespace(value)
    text = re.sub(r"(?<=[A-Za-z])(?=DIFC LAW NO\.?\s*\d)", " ", text)
    text = re.sub(r"(?<=[a-z])(?=[A-Z]{2,})", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip(" .")


def normalize_lookup_key(value: str) -> str:
    cleaned = clean_ocr_spacing(value)
    cleaned = cleaned.replace("’", "'").replace("“", '"').replace("”", '"')
    cleaned = re.sub(r"[^a-z0-9]+", " ", cleaned.lower())
    return re.sub(r"\s+", " ", cleaned).strip()


def to_display_title(value: str) -> str:
    text = clean_ocr_spacing(value)
    if not text:
        return ""
    if not any(char.islower() for char in text):
        words = []
        for index, word in enumerate(text.split()):
            upper = word.upper()
            if upper in UPPER_EXCEPTIONS:
                words.append(upper)
                continue
            lower = word.lower()
            if index and lower in STOPWORDS:
                words.append(lower)
            else:
                words.append(lower.capitalize())
        text = " ".join(words)
    for token in UPPER_EXCEPTIONS:
        text = re.sub(rf"\b{token.title()}\b", token, text)
        text = re.sub(rf"\b{token.lower()}\b", token, text)
    return normalize_whitespace(text)


def dedupe_preserve(values: Iterable[str]) -> List[str]:
    seen: set[str] = set()
    result: List[str] = []
    for value in values:
        cleaned = normalize_whitespace(value)
        if cleaned and cleaned not in seen:
            seen.add(cleaned)
            result.append(cleaned)
    return result


def coerce_text_list(value: Any) -> List[str]:
    if value in (None, "", []):
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, (list, tuple, set)):
        return [str(item) for item in value if str(item).strip()]
    return [str(value)]


def empty_case_metadata(page_count: int) -> Dict[str, Any]:
    return {
        "doc_type": "case",
        "case_name": "",
        "case_name_normalized": "",
        "neutral_citation": "",
        "neutral_citation_normalized": "",
        "claim_number": "",
        "claim_number_normalized": "",
        "court": "",
        "court_division": "",
        "hearing_date": "",
        "judgment_date": "",
        "judgment_release_date": "",
        "claimant": "",
        "defendants": [],
        "claimant_counsel": [],
        "defendant_counsel": [],
        "claimant_law_firm": "",
        "defendant_law_firm": "",
        "title_page_number": 1,
        "last_page_number": page_count,
        "page_count": page_count,
    }


def empty_law_metadata(page_count: int) -> Dict[str, Any]:
    return {
        "doc_type": "law",
        "official_title": "",
        "official_title_normalized": "",
        "short_title": "",
        "short_title_normalized": "",
        "law_number": "",
        "law_year": "",
        "official_citation": "",
        "official_citation_normalized": "",
        "citation_title_from_text": "",
        "consolidated_version_date": "",
        "amending_laws": [],
        "administering_authority": "",
        "jurisdiction": "",
        "title_page_number": 1,
        "last_page_number": page_count,
        "page_count": page_count,
        "alias_keys": [],
        "article_index": {},
        "section_heading_index": {},
    }


def build_blocks(
    content_items: Sequence[Dict[str, Any]],
    structured_items: Dict[int, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    blocks: List[Dict[str, Any]] = []
    for index, item in enumerate(content_items):
        structure = structured_items.get(index, {})
        text = normalize_whitespace(structure.get("text") or item.get("text") or item.get("table_body") or "")
        if not text:
            continue
        page_idx = int(structure.get("page_idx", item.get("page_idx", 0)))
        blocks.append(
            {
                "block_index": index,
                "page_idx": page_idx,
                "page_number": page_idx + 1,
                "text": text,
                "type": structure.get("type", item.get("type", "text")),
                "level": int(structure.get("level") or item.get("text_level") or 4),
            }
        )
    return blocks


def title_page_lines(blocks: Sequence[Dict[str, Any]]) -> List[str]:
    return [normalize_whitespace(block["text"]) for block in blocks if int(block["page_number"]) == 1 and normalize_whitespace(block["text"])]


def page_count_from_blocks(blocks: Sequence[Dict[str, Any]]) -> int:
    return max((int(block["page_number"]) for block in blocks), default=0)


def extract_claim_number(text: str) -> str:
    match = CASE_NUMBER_RE.search(normalize_whitespace(text))
    return normalize_whitespace(match.group(0).upper()) if match else ""


def extract_neutral_citation(text: str) -> str:
    match = NEUTRAL_CITATION_RE.search(normalize_whitespace(text))
    return normalize_whitespace(match.group(0).upper()) if match else ""


def extract_textual_dates(text: str) -> List[str]:
    return [normalize_whitespace(match.group(0)) for match in TEXTUAL_DATE_RE.finditer(normalize_whitespace(text))]


def extract_law_number_components(text: str) -> Tuple[str, str]:
    match = LAW_NUMBER_RE.search(clean_ocr_spacing(text))
    if not match:
        return "", ""
    return match.group(1), match.group(2)


def format_official_citation(number: str, year: str) -> str:
    if not number or not year:
        return ""
    return f"DIFC Law No. {int(number)} of {year}"


def extract_title_page_title(lines: Sequence[str]) -> str:
    candidates: List[str] = []
    fallback_candidates: List[str] = []
    for line in lines:
        cleaned = clean_ocr_spacing(line)
        if not cleaned:
            continue
        if cleaned.upper() == "DIFC":
            continue
        if cleaned.upper().startswith("CONTENTS"):
            break
        if cleaned.upper().startswith("AS AMENDED BY"):
            continue
        if "AMENDMENT LAW" in cleaned.upper() or CONSOLIDATED_VERSION_RE.search(cleaned):
            continue
        match = LAW_NUMBER_RE.search(cleaned)
        if match:
            prefix = cleaned[: match.start()].strip(" -:")
            if prefix:
                candidates.append(prefix)
            continue
        if re.search(r"\bLAW\b", cleaned, re.IGNORECASE):
            fallback_candidates.append(cleaned)
        candidates.append(cleaned)
    title = " ".join(candidates).strip()
    if not title:
        title = " ".join(fallback_candidates).strip()
    title = re.sub(r"^\s*DIFC\s+", "", title, flags=re.IGNORECASE)
    return to_display_title(title)


def extract_consolidated_version_date(lines: Sequence[str]) -> str:
    for line in lines:
        match = CONSOLIDATED_VERSION_RE.search(clean_ocr_spacing(line))
        if match:
            return normalize_whitespace(match.group(1))
    return ""


def extract_amending_laws(lines: Sequence[str]) -> List[str]:
    matches: List[str] = []
    for line in lines:
        cleaned = clean_ocr_spacing(line)
        upper = cleaned.upper()
        if "AMENDMENT LAW" in upper:
            matches.append(to_display_title(cleaned))
    return dedupe_preserve(matches)


def find_title_text_snippet(blocks: Sequence[Dict[str, Any]]) -> str:
    for index, block in enumerate(blocks):
        text = normalize_whitespace(block["text"])
        heading_match = SECTION_HEADING_RE.match(text)
        heading_title = heading_match.group("title").lower() if heading_match else text.lower()
        if heading_title.startswith("title"):
            snippets = [text]
            for next_block in blocks[index + 1 : index + 3]:
                next_text = normalize_whitespace(next_block["text"])
                if SECTION_HEADING_RE.match(next_text):
                    break
                snippets.append(next_text)
            return " ".join(snippets)
    return ""


def extract_citation_title_from_text(blocks: Sequence[Dict[str, Any]]) -> str:
    snippet = find_title_text_snippet(blocks)
    if not snippet:
        return ""
    for pattern in (TITLE_FROM_TEXT_RE, QUOTED_TITLE_RE):
        match = pattern.search(snippet)
        if match:
            return to_display_title(match.group(1))
    return ""


def strip_trailing_year(title: str) -> str:
    return normalize_whitespace(re.sub(r"\s+\d{4}$", "", title or "").strip())


def infer_short_title(official_title: str, citation_title_from_text: str, law_year: str) -> str:
    if citation_title_from_text:
        return citation_title_from_text
    if official_title and law_year and not re.search(rf"\b{re.escape(law_year)}\b", official_title):
        return f"{official_title} {law_year}"
    return official_title


def infer_administering_authority(blocks: Sequence[Dict[str, Any]]) -> str:
    combined = " ".join(normalize_whitespace(block["text"]) for block in blocks[:200])
    rules = [
        (r"\bRelevant Authority\b", "Relevant Authority"),
        (r"\bRegistrar\b", "Registrar"),
        (r"\bDIFCA\b", "DIFCA"),
    ]
    for pattern, label in rules:
        if re.search(pattern, combined, re.IGNORECASE):
            return label
    return ""


def infer_jurisdiction(lines: Sequence[str], blocks: Sequence[Dict[str, Any]]) -> str:
    joined = " ".join(lines) + " " + " ".join(normalize_whitespace(block["text"]) for block in blocks[:50])
    if re.search(r"Dubai International Financial Centre", joined, re.IGNORECASE) or re.search(r"\bDIFC\b", joined):
        return "Dubai International Financial Centre"
    return ""


def _strip_case_line_prefixes(text: str) -> str:
    cleaned = normalize_whitespace(clean_ocr_spacing(text))
    cleaned = CASE_NUMBER_RE.sub("", cleaned, count=1)
    cleaned = NEUTRAL_CITATION_RE.sub("", cleaned)
    cleaned = re.sub(
        r"\[\d{4}\]\s+DIFC\s+(?:CA|CFI|SCT|DEC|ENF|ARB|TCD)\s+\d{3}\b",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(rf"\b(?:{MONTH_NAME_RE})\s+\d{{1,2}},\s+\d{{4}}\b.*$", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(rf"\b\d{{1,2}}\s+(?:{MONTH_NAME_RE})\s+\d{{4}}\b.*$", "", cleaned, flags=re.IGNORECASE)
    return normalize_whitespace(cleaned.strip(" -,:"))


def extract_case_name(lines: Sequence[str], claimant: str = "", defendants: Optional[Sequence[str]] = None) -> str:
    for line in lines[:6]:
        cleaned = normalize_whitespace(line)
        if re.search(r"\bv\b", cleaned, re.IGNORECASE):
            title = _strip_case_line_prefixes(cleaned)
            if title:
                return title
    if claimant and defendants:
        return normalize_whitespace(f"{claimant} v {' '.join(defendants)}")
    return ""


def extract_case_issue_date(lines: Sequence[str]) -> str:
    combined = "\n".join(lines)
    for pattern in (
        r"\bJudgment\s*:\s*([^<\n]+)",
        r"\bDate of Issue\s*:\s*([^<\n]+)",
        r"\bReleased\s*:\s*([^<\n]+)",
    ):
        match = re.search(pattern, combined, re.IGNORECASE)
        if match:
            dates = extract_textual_dates(match.group(1))
            if dates:
                return dates[0]
    for line in lines[:4]:
        dates = extract_textual_dates(line)
        if dates:
            return dates[0]
    return ""


def extract_case_hearing_date(lines: Sequence[str]) -> str:
    combined = "\n".join(lines)
    for pattern in (
        r"\bHearing\s*:\s*([^<\n]+)",
        r"\bHeard on\s*:\s*([^<\n]+)",
    ):
        match = re.search(pattern, combined, re.IGNORECASE)
        if match:
            dates = extract_textual_dates(match.group(1))
            if dates:
                return dates[0]
    return ""


def extract_case_court(lines: Sequence[str]) -> str:
    for line in lines[:20]:
        cleaned = normalize_whitespace(line)
        upper = cleaned.upper()
        if "DUBAI INTERNATIONAL FINANCIAL CENTRE COURTS" in upper:
            return re.sub(r"^IN THE\s+", "", cleaned, flags=re.IGNORECASE)
        if "COURTS OF DUBAI INTERNATIONAL FINANCIAL CENTRE" in upper:
            return re.sub(r"^IN THE\s+", "", cleaned, flags=re.IGNORECASE)
    return ""


def extract_case_court_division(lines: Sequence[str]) -> str:
    for line in lines[:20]:
        cleaned = normalize_whitespace(line)
        upper = cleaned.upper()
        if re.match(r"^IN THE COURT OF ", upper):
            return re.sub(r"^IN THE\s+", "", cleaned, flags=re.IGNORECASE)
        if re.match(r"^IN THE SMALL CLAIMS TRIBUNAL$", upper):
            return re.sub(r"^IN THE\s+", "", cleaned, flags=re.IGNORECASE)
    for line in lines[:5]:
        cleaned = normalize_whitespace(line)
        match = re.search(
            rf"\b(?:{MONTH_NAME_RE})\s+\d{{1,2}},\s+\d{{4}}\s+(.+?)\s*-\s*(?:ORDERS?|JUDGMENTS?)\b",
            cleaned,
            re.IGNORECASE,
        )
        if match:
            return normalize_whitespace(match.group(1))
    return ""


def _party_lines_to_values(lines: Sequence[str], allow_split_enumerated: bool) -> List[str]:
    if not lines:
        return []
    cleaned_lines = []
    for line in lines:
        cleaned = normalize_whitespace(line)
        if not cleaned:
            continue
        cleaned = re.sub(
            r"\b(?:Claimant|Claimants|Defendant|Defendants|Respondent|Respondents|Applicant|Applicants|Appellant|Appellants)\b$",
            "",
            cleaned,
            flags=re.IGNORECASE,
        ).strip(" ,;:")
        if cleaned:
            cleaned_lines.append(cleaned)
    if not cleaned_lines:
        return []
    if allow_split_enumerated:
        joined = " ".join(cleaned_lines)
        matches = [
            normalize_whitespace(match.group(0))
            for match in re.finditer(r"\(\d+\)\s*[^()]+?(?=(?:\(\d+\)\s*|$))", joined)
        ]
        if matches:
            return dedupe_preserve(matches)
        if len(cleaned_lines) > 1:
            return dedupe_preserve(cleaned_lines)
    return [normalize_whitespace(" ".join(cleaned_lines))]


def extract_case_parties(lines: Sequence[str]) -> Tuple[str, List[str]]:
    claimant_lines: List[str] = []
    defendant_lines: List[str] = []
    active: Optional[str] = None
    started = False
    stop_prefixes = (
        "ORDER WITH REASONS",
        "AMENDED ORDER WITH REASONS",
        "REASONS FOR THE ORDER",
        "JUDGMENT OF",
        "BEFORE ",
        "UPON ",
        "AND UPON ",
        "IT IS HEREBY ORDERED",
    )

    for line in lines:
        cleaned = normalize_whitespace(line)
        upper = cleaned.upper()
        if not started:
            if upper == "BETWEEN":
                started = True
                active = "claimant"
            continue
        if upper in {"AND", "-AND-"}:
            active = "defendant"
            continue
        if any(upper.startswith(prefix) for prefix in stop_prefixes):
            break
        if re.search(r"\bCLAIMANT\b|\bPLAINTIFF\b|\bAPPELLANT\b|\bAPPLICANT\b", upper):
            continue
        if re.search(r"\bDEFENDANT\b|\bRESPONDENT\b", upper):
            continue
        if active == "claimant":
            claimant_lines.append(cleaned)
        elif active == "defendant":
            defendant_lines.append(cleaned)

    claimant_values = _party_lines_to_values(claimant_lines, allow_split_enumerated=False)
    defendant_values = _party_lines_to_values(defendant_lines, allow_split_enumerated=True)
    claimant = claimant_values[0] if claimant_values else ""
    return claimant, defendant_values


def extract_case_law_firms(lines: Sequence[str]) -> Tuple[str, str]:
    combined = " ".join(lines)
    claimant_patterns = (
        r"instructed by\s+(.+?)\s+for the\s+(?:claimant|appellant/claimant|claimant/appellant|claimant/respondent)",
        r"instructed by\s+(.+?)\s+for the\s+(?:applicant|plaintiff)",
    )
    defendant_patterns = (
        r"instructed by\s+(.+?)\s+for the\s+(?:defendant|respondent|defendant/appellant|applicant/respondent|defendant/respondent)",
        r"instructed by\s+(.+?)\s+for the\s+applicants/respondents",
    )

    def extract(patterns: Sequence[str]) -> str:
        for pattern in patterns:
            match = re.search(pattern, combined, re.IGNORECASE)
            if match:
                value = normalize_whitespace(match.group(1).strip(" ,.;"))
                if value:
                    return value
        return ""

    return extract(claimant_patterns), extract(defendant_patterns)


def build_case_payload(blocks: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    lines = title_page_lines(blocks)
    claimant, defendants = extract_case_parties(lines)
    return {
        "title_page_lines": lines[:25],
        "early_blocks": [normalize_whitespace(block["text"]) for block in blocks[:25]],
        "claim_number": extract_claim_number(" ".join(lines[:15])),
        "neutral_citation": extract_neutral_citation(" ".join(lines[:15])),
        "case_name_candidate": extract_case_name(lines, claimant, defendants),
        "court_candidate": extract_case_court(lines),
        "court_division_candidate": extract_case_court_division(lines),
        "judgment_date_candidate": extract_case_issue_date(lines),
        "hearing_date_candidate": extract_case_hearing_date(lines),
        "claimant_candidate": claimant,
        "defendant_candidates": defendants,
        "page_count": page_count_from_blocks(blocks),
    }


def classify_document(
    content_items: Sequence[Dict[str, Any]],
    structured_items: Dict[int, Dict[str, Any]],
    existing_metadata: Optional[Dict[str, Any]] = None,
) -> str:
    if existing_metadata:
        existing_type = normalize_whitespace(str(existing_metadata.get("doc_type") or ""))
        if existing_type in {"case", "law"}:
            return existing_type
    blocks = build_blocks(content_items, structured_items)
    lines = title_page_lines(blocks)
    head_text = " ".join(lines[:15])
    if extract_claim_number(head_text):
        return "case"
    if extract_neutral_citation(head_text):
        return "case"
    if re.search(r"\bIN THE\b.*\bCOURT\b|\bJUDGMENT\b|\bCLAIMANT\b|\bDEFENDANT\b", head_text, re.IGNORECASE):
        return "case"
    number, year = extract_law_number_components(head_text)
    if number and year:
        return "law"
    if re.search(r"\bLAW\b", head_text, re.IGNORECASE) and re.search(r"\bPART\b|\bCONTENTS\b", " ".join(lines[1:20]), re.IGNORECASE):
        return "law"
    return "case"


def build_case_metadata(
    content_items: Sequence[Dict[str, Any]],
    structured_items: Dict[int, Dict[str, Any]],
    existing_metadata: Optional[Dict[str, Any]] = None,
    llm_metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    blocks = build_blocks(content_items, structured_items)
    lines = title_page_lines(blocks)
    page_count = page_count_from_blocks(blocks)
    metadata = empty_case_metadata(page_count)

    merged = {}
    for source in (existing_metadata or {}, llm_metadata or {}):
        for key, value in source.items():
            if key in CASE_FIELDS and value not in (None, "", []):
                merged[key] = value

    first_page_text = "\n".join(lines)
    claimant, defendants = extract_case_parties(lines)
    claimant_law_firm, defendant_law_firm = extract_case_law_firms(lines)
    derived_issue_date = extract_case_issue_date(lines)
    metadata.update(
        {
            "case_name": normalize_whitespace(
                str(merged.get("case_name") or extract_case_name(lines, claimant, defendants))
            ),
            "neutral_citation": normalize_whitespace(
                str(merged.get("neutral_citation") or extract_neutral_citation(first_page_text))
            ),
            "claim_number": normalize_whitespace(
                str(merged.get("claim_number") or extract_claim_number(first_page_text))
            ).upper(),
            "court": normalize_whitespace(str(merged.get("court") or extract_case_court(lines))),
            "court_division": normalize_whitespace(str(merged.get("court_division") or extract_case_court_division(lines))),
            "hearing_date": normalize_whitespace(str(merged.get("hearing_date") or extract_case_hearing_date(lines))),
            "judgment_date": normalize_whitespace(str(merged.get("judgment_date") or derived_issue_date)),
            "judgment_release_date": normalize_whitespace(str(merged.get("judgment_release_date") or "")),
            "claimant": normalize_whitespace(str(merged.get("claimant") or claimant)),
            "defendants": dedupe_preserve(coerce_text_list(merged.get("defendants") or defendants)),
            "claimant_counsel": dedupe_preserve(coerce_text_list(merged.get("claimant_counsel") or [])),
            "defendant_counsel": dedupe_preserve(coerce_text_list(merged.get("defendant_counsel") or [])),
            "claimant_law_firm": normalize_whitespace(str(merged.get("claimant_law_firm") or claimant_law_firm)),
            "defendant_law_firm": normalize_whitespace(str(merged.get("defendant_law_firm") or defendant_law_firm)),
        }
    )
    if not metadata["judgment_release_date"] and metadata["judgment_date"]:
        metadata["judgment_release_date"] = metadata["judgment_date"] if "ORDER" not in first_page_text.upper() else ""
    metadata["case_name_normalized"] = normalize_lookup_key(metadata["case_name"])
    metadata["neutral_citation_normalized"] = normalize_lookup_key(metadata["neutral_citation"])
    metadata["claim_number_normalized"] = normalize_lookup_key(metadata["claim_number"])
    return metadata


def merge_case_llm_metadata(base: Dict[str, Any], llm_metadata: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not llm_metadata:
        return base
    merged = dict(base)
    for key in [
        "case_name",
        "neutral_citation",
        "court",
        "court_division",
        "claim_number",
        "hearing_date",
        "judgment_date",
        "judgment_release_date",
        "claimant",
        "claimant_law_firm",
        "defendant_law_firm",
    ]:
        if llm_metadata.get(key):
            merged[key] = normalize_whitespace(str(llm_metadata[key]))
    if llm_metadata.get("defendants"):
        merged["defendants"] = dedupe_preserve(coerce_text_list(llm_metadata["defendants"]))
    if llm_metadata.get("claimant_counsel"):
        merged["claimant_counsel"] = dedupe_preserve(coerce_text_list(llm_metadata["claimant_counsel"]))
    if llm_metadata.get("defendant_counsel"):
        merged["defendant_counsel"] = dedupe_preserve(coerce_text_list(llm_metadata["defendant_counsel"]))
    return merged


def merge_law_llm_metadata(base: Dict[str, Any], llm_metadata: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not llm_metadata:
        return base
    merged = dict(base)
    for key in [
        "official_title",
        "short_title",
        "law_number",
        "law_year",
        "official_citation",
        "citation_title_from_text",
        "consolidated_version_date",
        "administering_authority",
        "jurisdiction",
    ]:
        if llm_metadata.get(key):
            merged[key] = normalize_whitespace(str(llm_metadata[key]))
    if llm_metadata.get("amending_laws"):
        merged["amending_laws"] = dedupe_preserve(coerce_text_list(llm_metadata["amending_laws"]))
    return merged


def build_alias_keys(
    official_title: str,
    short_title: str,
    law_year: str,
    official_citation: str,
    citation_title_from_text: str,
) -> List[str]:
    aliases: set[str] = set()

    def add(value: str) -> None:
        cleaned = normalize_whitespace(value)
        normalized = normalize_lookup_key(cleaned)
        if not cleaned or not normalized or normalized in GENERIC_ALIAS_HEADINGS:
            return
        aliases.add(cleaned)
        collapsed = clean_ocr_spacing(cleaned)
        if collapsed and collapsed != cleaned:
            aliases.add(collapsed)

    for title in [official_title, short_title, citation_title_from_text]:
        if not title:
            continue
        add(title)
        stripped = strip_trailing_year(title)
        add(stripped)
        if law_year and stripped:
            add(f"{stripped} {law_year}")
        if title.lower().startswith("the "):
            add(title[4:])

    add(official_citation)
    return sorted(aliases, key=lambda item: (len(item), item.lower()))


def _make_index_entry(blocks: Sequence[Dict[str, Any]], heading: str, article_number: str = "") -> Dict[str, Any]:
    page_numbers = sorted({int(block["page_number"]) for block in blocks})
    return {
        "heading": heading,
        "article_number": article_number,
        "page_start": min(page_numbers) if page_numbers else 0,
        "page_end": max(page_numbers) if page_numbers else 0,
        "block_start": int(blocks[0]["block_index"]) if blocks else 0,
        "block_end": int(blocks[-1]["block_index"]) if blocks else 0,
    }


def build_article_and_section_indexes(blocks: Sequence[Dict[str, Any]]) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, List[Dict[str, Any]]]]:
    article_index: Dict[str, Dict[str, Any]] = {}
    section_heading_index: Dict[str, List[Dict[str, Any]]] = {}

    heading_positions: List[Tuple[int, str, str]] = []
    for index, block in enumerate(blocks):
        match = SECTION_HEADING_RE.match(normalize_whitespace(block["text"]))
        if not match:
            continue
        heading_positions.append((index, match.group("number"), to_display_title(match.group("title"))))

    for position_index, (block_pos, section_number, heading_title) in enumerate(heading_positions):
        next_block_pos = heading_positions[position_index + 1][0] if position_index + 1 < len(heading_positions) else len(blocks)
        section_blocks = list(blocks[block_pos:next_block_pos])
        article_key = f"Article {section_number}"
        article_index[article_key] = _make_index_entry(section_blocks, heading_title, section_number)
        section_heading_index.setdefault(heading_title, []).append(_make_index_entry(section_blocks, heading_title, section_number))

        subsection_start: Optional[int] = None
        subsection_key: Optional[str] = None
        alpha_start: Optional[int] = None
        alpha_key: Optional[str] = None

        for offset, block in enumerate(section_blocks[1:], start=1):
            text = normalize_whitespace(block["text"])
            numeric_match = NUMERIC_SUBSECTION_RE.match(text)
            alpha_match = ALPHA_SUBSECTION_RE.match(text)
            roman_match = ROMAN_SUBSECTION_RE.match(text)

            if numeric_match:
                if alpha_start is not None and alpha_key is not None:
                    article_index[alpha_key] = _make_index_entry(section_blocks[alpha_start:offset], heading_title, section_number)
                    alpha_start = None
                    alpha_key = None
                if subsection_start is not None and subsection_key is not None:
                    article_index[subsection_key] = _make_index_entry(section_blocks[subsection_start:offset], heading_title, section_number)
                subsection_start = offset
                subsection_key = f"Article {section_number}({numeric_match.group(1)})"
                continue

            if alpha_match and subsection_key is not None:
                if alpha_start is not None and alpha_key is not None:
                    article_index[alpha_key] = _make_index_entry(section_blocks[alpha_start:offset], heading_title, section_number)
                alpha_start = offset
                alpha_key = f"{subsection_key}({alpha_match.group(1).lower()})"
                continue

            if roman_match and alpha_key is not None:
                nested_key = f"{alpha_key}({roman_match.group(1).lower()})"
                article_index[nested_key] = _make_index_entry([block], heading_title, section_number)

        if alpha_start is not None and alpha_key is not None:
            article_index[alpha_key] = _make_index_entry(section_blocks[alpha_start:], heading_title, section_number)
        if subsection_start is not None and subsection_key is not None:
            article_index[subsection_key] = _make_index_entry(section_blocks[subsection_start:], heading_title, section_number)

    return article_index, section_heading_index


def build_law_payload(blocks: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    lines = title_page_lines(blocks)
    heading_snippets = [normalize_whitespace(block["text"]) for block in blocks[0:25]]
    return {
        "title_page_lines": lines[:20],
        "early_headings": heading_snippets,
        "title_clause_snippet": find_title_text_snippet(blocks),
        "page_count": page_count_from_blocks(blocks),
    }


def build_law_metadata(
    content_items: Sequence[Dict[str, Any]],
    structured_items: Dict[int, Dict[str, Any]],
    existing_metadata: Optional[Dict[str, Any]] = None,
    llm_metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    blocks = build_blocks(content_items, structured_items)
    lines = title_page_lines(blocks)
    page_count = page_count_from_blocks(blocks)
    metadata = empty_law_metadata(page_count)

    title_page_text = " ".join(lines)
    law_number, law_year = extract_law_number_components(title_page_text)
    official_title = extract_title_page_title(lines)
    citation_title_from_text = extract_citation_title_from_text(blocks)
    short_title = infer_short_title(official_title, citation_title_from_text, law_year)

    metadata.update(
        {
            "official_title": official_title,
            "short_title": short_title,
            "law_number": law_number,
            "law_year": law_year,
            "official_citation": format_official_citation(law_number, law_year),
            "citation_title_from_text": citation_title_from_text,
            "consolidated_version_date": extract_consolidated_version_date(lines),
            "amending_laws": extract_amending_laws(lines),
            "administering_authority": infer_administering_authority(blocks),
            "jurisdiction": infer_jurisdiction(lines, blocks),
        }
    )
    article_index, section_heading_index = build_article_and_section_indexes(blocks)
    metadata["article_index"] = article_index
    metadata["section_heading_index"] = section_heading_index

    metadata = merge_law_llm_metadata(metadata, llm_metadata)
    metadata["official_title"] = to_display_title(metadata.get("official_title") or official_title)
    metadata["citation_title_from_text"] = to_display_title(metadata.get("citation_title_from_text") or citation_title_from_text)
    metadata["short_title"] = to_display_title(
        metadata.get("short_title")
        or infer_short_title(metadata["official_title"], metadata["citation_title_from_text"], metadata.get("law_year") or law_year)
    )
    merged_law_number = normalize_whitespace(str(metadata.get("law_number") or law_number))
    merged_law_year = normalize_whitespace(str(metadata.get("law_year") or law_year))
    parsed_number, parsed_year = extract_law_number_components(
        " ".join(
            value
            for value in [merged_law_number, merged_law_year, str(metadata.get("official_citation") or "")]
            if value
        )
    )
    metadata["law_number"] = parsed_number or merged_law_number
    metadata["law_year"] = parsed_year or merged_law_year
    metadata["official_citation"] = (
        normalize_whitespace(str(metadata.get("official_citation") or ""))
        or format_official_citation(metadata["law_number"], metadata["law_year"])
    )
    metadata["consolidated_version_date"] = normalize_whitespace(str(metadata.get("consolidated_version_date") or ""))
    metadata["amending_laws"] = dedupe_preserve(coerce_text_list(metadata.get("amending_laws") or []))
    metadata["administering_authority"] = normalize_whitespace(str(metadata.get("administering_authority") or ""))
    metadata["jurisdiction"] = normalize_whitespace(str(metadata.get("jurisdiction") or ""))
    metadata["official_title_normalized"] = normalize_lookup_key(metadata["official_title"])
    metadata["short_title_normalized"] = normalize_lookup_key(metadata["short_title"])
    metadata["official_citation_normalized"] = normalize_lookup_key(metadata["official_citation"])
    metadata["alias_keys"] = build_alias_keys(
        metadata["official_title"],
        metadata["short_title"],
        metadata["law_year"],
        metadata["official_citation"],
        metadata["citation_title_from_text"],
    )
    return metadata
