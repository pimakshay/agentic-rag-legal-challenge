from __future__ import annotations

import ast
import json
import logging
import os
import subprocess
import traceback
from typing import Any, Dict, Optional

from tqdm import tqdm

try:
    from .legal_metadata import (
        LAW_METADATA_PROMPT,
        build_blocks,
        build_case_metadata,
        build_law_metadata,
        build_law_payload,
        classify_document,
        parse_llm_json,
        title_page_lines,
    )
    from .utils import call_openai_llm as call_llm
except ImportError:
    from legal_metadata import (  # type: ignore
        LAW_METADATA_PROMPT,
        build_blocks,
        build_case_metadata,
        build_law_metadata,
        build_law_payload,
        classify_document,
        parse_llm_json,
        title_page_lines,
    )
    from utils import call_openai_llm as call_llm  # type: ignore


folder = "../public_dataset/docs_corpus"
logger = logging.getLogger(__name__)
prompt_template = """/no_think
# Task
You are an expert document structure in Legal domain.
Given documentation paragraphs (type: text or table, text: content of that paragraphs, page_idx: page index (starting from 0)),
your task is to assign correct heading levels to all paragraphs in that document so that they follow a natural structure.

# Input Format
The input is a dictionary where:
- Each key is a unique heading ID (integer).
- Each value is dict, which contains three keys: 'type', 'text' and 'page_idx'. The 'type' should be 'text' or 'table'. The 'page_idx' is integer starting from 0.

# Instructions
1. **Primary rules**
    - Do not add, remove, merge, or change any headings.
    - The number of output elements must exactly match the input.
2. **Heading Hierarchy Rules**
    - The level 1 must be the highest heading level, and gradually decrease to 5.
    - Maximum depth is **5 levels**. Heading level must be 1, 2, 3, 4 or 5. NO OTHER NUMBERS.
    - Headings that follow the same textual pattern should generally be assigned to the same level.
    - When increasing depth, levels can only increase by 1 step.
3. **More strict rules**
    - On the first page, case title, ORDER WITH REASONS headings, UPON/AND UPON headings and IT IS HEREBY ORDERED THAT behave as top-level headings.
    - "SCHEDULE OF REASONS" should be level 1.
    - Consecutive numbered sections such as "1.", "2.", "3." should usually be the same level.

# Output Format
- Return only a valid Python dictionary in the format {{HeadingID: HeadingLevel}}.
- Every heading level must be 1, 2, 3, 4 or 5.

# Input
{input_dict}

# Result
"""

CASE_METADATA_PROMPT = """You are a legal document information extraction system.

Your task is to extract key metadata from a legal judgment title page.

Rules:
- Only extract information explicitly present in the text.
- Do not infer missing information.
- If a field is not present, return NOT_FOUND.
- Preserve original wording.
- Return JSON only.

Text:
{text}

Return:
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
"""

RE_THINK = __import__("re").compile(r"<think>(.*?)</think>", __import__("re").DOTALL)
RE_DICT = __import__("re").compile(r"(\d+)\s*:\s*(\d+)")


def parse_heading_level(text: str) -> Optional[Dict[int, int]]:
    try:
        cleaned = RE_THINK.sub("", (text or "")).replace("`", "").strip()
        try:
            parsed = ast.literal_eval(cleaned)
            if not isinstance(parsed, dict):
                return None
            return {int(key): max(1, min(5, int(value))) for key, value in parsed.items()}
        except Exception:
            matches = list(RE_DICT.finditer(cleaned))
            result: Dict[int, int] = {}
            for match in matches:
                result[int(match.group(1))] = max(1, min(5, int(match.group(2))))
            return result or None
    except Exception:
        return None


class Ingestion:
    def __init__(self, folder: str, output_path: str):
        self.folder = folder
        self.files = sorted(file for file in os.listdir(folder) if file.endswith(".pdf"))
        self.output_path = output_path
        self.use_llm = os.getenv("LEGAL_INGEST_USE_LLM", "0").strip().lower() not in {"0", "false", "no"}
        os.makedirs(self.output_path, exist_ok=True)

    @staticmethod
    def _get_parse_cmd(file: str, output_path: str) -> list[str]:
        return f"mineru -p {file} -l en -m txt -b pipeline -o {output_path}".split()

    @staticmethod
    def _read_json(path: str, default: Any) -> Any:
        if not os.path.isfile(path):
            return default
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)

    @staticmethod
    def _write_json(path: str, value: Any) -> None:
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=4, ensure_ascii=False)

    @staticmethod
    def refine_parse_result(file: str) -> None:
        data = Ingestion._read_json(file, [])
        new_data = []
        for element in data:
            if "text" not in element and "table_body" not in element:
                continue
            text = (element.get("text") or element.get("table_body") or "").strip()
            if not text:
                continue
            if element["type"] in {"text", "table"}:
                new_data.append(element.copy())
            elif element["type"] == "discarded":
                normalized = element.copy()
                normalized["type"] = "text"
                new_data.append(normalized)
        Ingestion._write_json(file, new_data)

    def parse(self) -> bool:
        try:
            for file in tqdm(self.files, desc="Parsing documents"):
                cmd = Ingestion._get_parse_cmd(os.path.join(self.folder, file), self.output_path)
                process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
                assert process.stdout is not None
                for line in process.stdout:
                    print(line, end="")
                process.wait()
                self.refine_parse_result(
                    os.path.join(self.output_path, file[:-4], "txt", f"{file[:-4]}_content_list.json")
                )
            return True
        except Exception:
            logger.error(traceback.format_exc())
            return False

    @staticmethod
    def _structure_analysis(parse_file: str, structure_file: str) -> bool:
        def filter_elements(data: list[dict[str, Any]]) -> list[dict[str, Any]]:
            normalized = []
            for element in data:
                if element["type"] not in {"text", "table"}:
                    continue
                copied = element.copy()
                if copied["type"] == "table":
                    copied["text"] = copied.get("table_body", "")
                    copied.pop("table_body", None)
                normalized.append(copied)
            return normalized

        data = filter_elements(Ingestion._read_json(parse_file, []))
        input_dict: Dict[int, Dict[str, Any]] = {}
        for index, element in enumerate(data):
            input_dict[index] = {
                "type": element["type"],
                "page_idx": element["page_idx"],
                "text": element["text"],
            }

        prompt_input = {key: value.copy() for key, value in input_dict.items()}
        for value in prompt_input.values():
            text = value["text"]
            if len(text) > 200:
                value["text"] = text[:200] + "..."

        result = parse_heading_level(call_llm(prompt_template.format(input_dict=prompt_input)))
        if not result or set(result.keys()) != set(input_dict.keys()):
            return False

        for key in input_dict:
            input_dict[key]["level"] = result[key]
        Ingestion._write_json(structure_file, input_dict)
        return True

    @staticmethod
    def _structure_analysis_for_long_file(parse_file: str, structure_file: str) -> bool:
        def filter_elements(data: list[dict[str, Any]]) -> list[dict[str, Any]]:
            normalized = []
            for element in data:
                if element["type"] not in {"text", "table"}:
                    continue
                copied = element.copy()
                if copied["type"] == "table":
                    copied["text"] = copied.get("table_body", "")
                    copied.pop("table_body", None)
                normalized.append(copied)
            return normalized

        data = filter_elements(Ingestion._read_json(parse_file, []))
        input_dict: Dict[int, Dict[str, Any]] = {}
        for index, element in enumerate(data):
            input_dict[index] = {
                "type": element["type"],
                "page_idx": element["page_idx"],
                "text": element["text"],
            }
        batches_input_dict = []
        appeared_page_idx = []
        all_common_keys = []
        for key, element in input_dict.items():
            page_idx = element["page_idx"]
            if page_idx % 20 == 0 and (page_idx not in appeared_page_idx):
                batches_input_dict.append({})
                if page_idx > 0:
                    common_keys = []
                    for key2, element2 in batches_input_dict[-2].items():
                        page_idx2 = element2['page_idx']
                        assert page_idx2 < page_idx
                        if page_idx2 >= page_idx - 2:  # Overlapping page
                            batches_input_dict[-1][key2] = element2.copy()
                            common_keys.append(key2)
                    all_common_keys.append(common_keys)


            batches_input_dict[-1][key] = element.copy()
            appeared_page_idx.append(page_idx)

        pre_info = ''
        for i, batch_input_dict in tqdm(enumerate(batches_input_dict)):
            prompt_input = {key: value.copy() for key, value in batch_input_dict.items()}
            for value in prompt_input.values():
                text = value["text"]
                if len(text) > 200:
                    value["text"] = text[:200] + "..."

            result = parse_heading_level(call_llm(prompt_template.format(input_dict=prompt_input) + pre_info))
            if not result or set(result.keys()) != set(batch_input_dict.keys()):
                return False

            for key in batch_input_dict:
                batch_input_dict[key]["level"] = result[key]
            structure_file_i = structure_file[:-5] + str(i) + ".json"
            Ingestion._write_json(structure_file_i, batch_input_dict)

            if i != len(batches_input_dict) - 1:
                pre_info = "{" + ", ".join([f"{key}: {result[key]}" for key in all_common_keys[i] ]) + ', '
        result = {}
        for i in range(len(batches_input_dict)):
            structure_file_i = structure_file[:-5] + str(i) + ".json"
            assert os.path.isfile(structure_file_i)
            data = json.load(open(structure_file_i))
            for key, element in data.items():
                if key in result:
                    continue
                result[key] = element
        Ingestion._write_json(structure_file, result)

        return True

    def structure_analysis(self) -> bool:
        try:
            for file in tqdm(self.files, desc="Structure analysis"):
                file_name = file[:-4]
                parse_file = os.path.join(self.output_path, file_name, "txt", f"{file_name}_content_list.json")
                structure_file = os.path.join(self.output_path, file_name, "txt", f"{file_name}_structure.json")
                if not os.path.isfile(parse_file):
                    raise FileNotFoundError(parse_file)
                success = self._structure_analysis(parse_file, structure_file)
                if not success:
                    success = self._structure_analysis_for_long_file(parse_file, structure_file)
                logger.info("structure %s success=%s", file_name, success)
            return True
        except Exception:
            logger.error(traceback.format_exc())
            return False

    @staticmethod
    def _case_llm_metadata(content_items: list[dict[str, Any]]) -> Dict[str, Any]:
        first_page_text = "\n".join(
            (item.get("text") or item.get("table_body") or "").strip()
            for item in content_items
            if int(item.get("page_idx", 0)) == 0
        )
        if not first_page_text.strip():
            return {}
        return parse_llm_json(call_llm(CASE_METADATA_PROMPT.format(text=first_page_text)))

    @staticmethod
    def _law_llm_metadata(blocks: list[dict[str, Any]]) -> Dict[str, Any]:
        payload = build_law_payload(blocks)
        return parse_llm_json(call_llm(LAW_METADATA_PROMPT.format(payload=json.dumps(payload, indent=2, ensure_ascii=False))))

    def _metadata_extraction(self, parse_file: str, structure_file: str, metadata_file: str) -> Dict[str, Any] | bool:
        try:
            content_items = self._read_json(parse_file, [])
            structured_items = self._read_json(structure_file, {})
            existing_metadata = self._read_json(metadata_file, {})
            blocks = build_blocks(content_items, structured_items)
            doc_type = classify_document(content_items, structured_items, existing_metadata)

            if doc_type == "law":
                llm_metadata = self._law_llm_metadata(blocks) if self.use_llm else {}
                result = build_law_metadata(content_items, structured_items, existing_metadata, llm_metadata)
            else:
                existing_case = existing_metadata if existing_metadata.get("doc_type") == "case" or existing_metadata.get("claim_number") else existing_metadata
                llm_metadata = {}
                if self.use_llm and (not existing_case or not existing_case.get("case_name") or not existing_case.get("claim_number")):
                    llm_metadata = self._case_llm_metadata(content_items)
                result = build_case_metadata(content_items, structured_items, existing_case, llm_metadata)

            self._write_json(metadata_file, result)
            logger.info("metadata %s doc_type=%s", os.path.basename(metadata_file), result.get("doc_type"))
            return result
        except Exception:
            logger.error(traceback.format_exc())
            return False

    def metadata_extraction(self) -> bool:
        try:
            for file in tqdm(self.files, desc="Metadata extraction"):
                file_name = file[:-4]
                parse_file = os.path.join(self.output_path, file_name, "txt", f"{file_name}_content_list.json")
                structure_file = os.path.join(self.output_path, file_name, "txt", f"{file_name}_structure.json")
                metadata_file = os.path.join(self.output_path, file_name, "txt", f"{file_name}_metadata.json")
                success = self._metadata_extraction(parse_file, structure_file, metadata_file)
                if not success:
                    logger.error("metadata extraction failed for %s", file)
            return True
        except Exception:
            logger.error(traceback.format_exc())
            return False

    def ingest(self) -> None:
        self.parse()
        self.structure_analysis()
        self.metadata_extraction()


if __name__ == "__main__":
    pipeline = Ingestion(folder=folder, output_path="docs_corpus_ingest_result")
    pipeline.ingest()
