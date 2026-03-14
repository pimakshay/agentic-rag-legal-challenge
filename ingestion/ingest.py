import os
import subprocess
import json
import re
import ast
from tqdm import tqdm
import traceback
from loguru import logger
from utils import call_llm

folder = "../public_dataset/docs_corpus"
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
	- Headings that follow the same textual pattern (e.g., numbered articles, chapters, or sections) should generally be assigned to the same level.
	- Level progression constraints:
		+ When increasing depth, levels can only increase by **1** step (e.g., 1 → 2 valid, 1 → 3 invalid).
		+ When decreasing depth, levels can drop by any number (e.g., 3 → 1 valid).
3. **More strictly rules**
        - In first page (page_idx = 0):
               + there are usually case title between claimant and defendants, their heading level should be 1.
               + "ORDER WITH REASONS OF ..." should be level 1
               + "UPON ..." and "AND UPON ..." should be level 1
               + "IT IS HEREBY ORDERED THAT:" should be level 1 and its following paragraphs belongs to that section should be level 2.
        - In legal domain, if text is "SCHEDULE OF REASONS", then heading level should be 1.
        - There are usually sections starting with CONSECUTIVE numbers: "1.", "2.", "3.", .... They should be SAME LEVEL 2.
              + Note: But there are also sub-sections maybe starting with numbers: "(1)", "(2)", ... or  "1.", "2." ... or "(a)", "(b)", ... or "1.1", "3.1", "2.2", ... or [7.], [8.], (i), (ii), (iii), ... Read content carefully and please assign them level 3, 4 or 5.
              + Between sections, sometimes there are important titles (For example: The Release Agreement, Introduction, Summary, Discussion, The grounds of appeal ... ). They should have level 1.

# Output Format
- Return only the optimized structure as a valid **Python dictionary** in the format: {{HeadingID: HeadingLevel}}:
- EVERY Heading level must be 1, 2, 3, 4 or 5. NO OTHER NUMBERS. No null values.
For example:
{{
  0:1,
  1:1,
  2:2,
  3:3,
  4:2
}}
- No explanations, reasoning, or additional text.
- No code generation or commentary.

# Input
{input_dict}

# Result
"""


prompt_template_kie = """You are a legal document information extraction system.

Your task is to extract key metadata from a legal judgment page.


# Rules:
- Only extract information explicitly present in the text.
- Do NOT infer or hallucinate missing information.
- If a field is not present, return NOT_FOUND.
- Preserve the original wording of names.
- Return the output in JSON format only.

#Field descriptions:

- case_name: Full case title including all parties (e.g., "A v B").
- neutral_citation: Official court citation such as "[2025] DIFC SCT 169".
- court: Name of the court institution.
- court_division: Specific division or chamber of the court (e.g., Court of First Instance).
- claim_number: Case filing number (e.g., "DEC 001/2025").
- hearing_date: Date of the hearing if stated.
- judgment_date: Date the judgment was issued or signed.
- judgment_release_date: Date the judgment was publicly released or published.
- claimant: The party bringing the claim.
- defendants: List of all defendants in the case.
- claimant_counsel: Lawyers representing the claimant.
- defendant_counsel: Lawyers representing the defendants.
- claimant_law_firm: Law firm representing the claimant.
- defendant_law_firm: Law firm representing the defendants.

# Text:
{text}

If the value cannot be found exactly in the text, return NOT_FOUND.
Do not rephrase or summarize extracted values.

Return JSON in the following format:

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

RE_THINK = re.compile(r"<think>(.*?)</think>", re.DOTALL)
RE_DICT = re.compile(r"\b(\d+)\:(\d+)\,")
def parse_heading_level(text):
    try:
        text = RE_THINK.sub("", text).strip()
        text = text.replace('`', '')
        try:
            result = ast.literal_eval(text)
            result = {int(x): min(5, max(y, 1)) for x, y in result.items()}
            return result
        except:
            matched = list(RE_DICT.finditer(text))
            result = {}
            prev_key = -1
            for m in matched:
                key = m.group(0)
                level = m.group(1)
                if prev_key + 1 != key:  # Make sure consecutive
                    return None
                result[key] = min(5, max(level, 1))
                prev_key = key
            return result
    except:
        return None
    
def parse_metadata(text):
    try:
        text = RE_THINK.sub("", text).strip()
        text = text.replace("```json", "")
        text = text.replace('`', '')
        text = text[text.find('{'):text.rfind('}') + 1]
        text = text.replace('"NOT_FOUND"', '""').replace('NOT_FOUND', '""')
        text = text.strip()
        try:
            result = ast.literal_eval(text)
            if not isinstance(result, dict):
                return {}
            return result
        except Exception as e:
            print(e)
            return {}
    except:
        return {}

class Ingestion:
    def __init__(self, folder, output_path):
        self.folder = folder
        self.files = [file for file in os.listdir(folder) if file.endswith(".pdf")]
        self.output_path = output_path
        os.makedirs(self.output_path, exist_ok=True)

    @staticmethod
    def _get_parse_cmd(file, output_path):
        cmd = f"mineru -p {file} -l en -m txt -b pipeline -o {output_path}"
        return cmd.split()

    def parse(self):
        try:
            for file in tqdm(self.files, desc="Parsing documents"):
                cmd = Ingestion._get_parse_cmd(os.path.join(self.folder, file), self.output_path)
                process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True
                )

                # stream log realtime
                for line in process.stdout:
                    print(line, end="")

                process.wait()
            return True
        except:
            return False

    @staticmethod
    def _structure_analysis(parse_file, structure_file):
        print('>' * 10 + parse_file)
        def filter_elements(data):
            new_data = []
            for element in data:
                if "text" in element and element["text"] == "":
                    continue
                if element["type"] == "table":
                    if "table_body" in element:
                        element["text"] = element["table_body"]
                        del element["table_body"]
                    else:
                        continue
                if element["type"] not in ["table", "text"]:
                    continue
                new_data.append(element.copy())
            return new_data
        data = json.load(open(parse_file))
        data = filter_elements(data)


        input_dict = {}
        for i, element in enumerate(data):
            page_idx = element["page_idx"]
            text = element["text"]
            if i not in input_dict:
                input_dict[i] = {}
            input_dict[i]["type"] = element["type"]
            input_dict[i]["page_idx"] = element["page_idx"]
            input_dict[i]["text"] = text

        input_dict_prompting = input_dict.copy()
        for key in input_dict_prompting:
            text = input_dict_prompting[key]["text"]
            if len(text) > 200:
                text = text[:200] + "..."
            input_dict_prompting[key]["text"] = text

        prompt = prompt_template.format(input_dict=input_dict_prompting)
        result = call_llm(prompt)
        print(f'R1 = {result}')
        result = parse_heading_level(result)
        print(f'R2 = {result}')
        if result == None:
            result = {}
        if result.keys() != input_dict.keys():
            result = {}
        if result == {}:
            return False

        for key in input_dict:
            input_dict[key]["level"] = result[key]
        with open(structure_file, 'w') as f:
            json.dump(input_dict, f, indent=4, ensure_ascii=False)
        return True
        
    
    def structure_analysis(self):
        try:
            for file in tqdm(self.files, desc="Structure analysis"):
                file_name = file[:-4]
                print(file_name)
                parse_file = os.path.join(self.output_path,  file_name, "txt", f"{file_name}_content_list.json")
                assert os.path.isfile(parse_file)
                structure_file = os.path.join(self.output_path, file_name, "txt", f"{file_name}_structure.json")
                success = Ingestion._structure_analysis(parse_file, structure_file)
                if not success:
                    print(file)
                print("success", success)
        except Exception as e:
            logger.error(traceback.format_exc())
            return False
        
    def _metadata_extaction(file, metadata_file):
        try:
            data = json.load(open(file))
            text = "\n".join([element.get("text", element.get("table_body", "")).strip() for element in data if element["page_idx"] == 0])
            prompt = prompt_template_kie.format(text=text)
            result = call_llm(prompt)
            result = parse_metadata(result)
            with open(metadata_file, 'w') as f:
                json.dump(result, f, indent=4, ensure_ascii=False)
            logger.info(f"result = {result}")
            return result
        except Exception as e:
            logger.error(traceback.format_exc())
            return False

    def metadata_extraction(self):
        try:
            for file in tqdm(self.files, desc="Metadata extraction"):
                file_name = file[:-4]
                logger.info(file_name)
                parse_file = os.path.join(self.output_path,  file_name, "txt", f"{file_name}_content_list.json")
                metadata_file = os.path.join(self.output_path,  file_name, "txt", f"{file_name}_metadata.json")
                success = Ingestion._metadata_extaction(parse_file, metadata_file)  
                if not success:
                    print(file)              

        
        except Exception as e:
            logger.error(traceback.format_exc())
            return False

    def ingest(self):
        success = self.parse()
        success = self.structure_analysis()
        success = self.metadata_extraction()

if __name__ == "__main__":
    pipeline = Ingestion(folder=folder, output_path=f"docs_corpus_ingest_result")
    pipeline.ingest()

    

