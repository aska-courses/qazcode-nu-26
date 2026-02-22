import json
import re
from pathlib import Path
from utils.generator import LLMGenerator

import json
import re
import os
from dotenv import load_dotenv
from openai import OpenAI
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--start", type=int, default=0)
parser.add_argument("--end", type=int, default=None)
args = parser.parse_args()



API_KEY="EMPTY"
HUB_URL=os.getenv("VLLM_120B_URL_5") 
MODEL="gpt-oss-120b-gpu5"

client = OpenAI(base_url=HUB_URL, api_key=API_KEY)

generator = LLMGenerator(client=client, model_name=MODEL)

# -------- utils --------

def parse_llm_response(raw: str) -> dict:
    clean = re.sub(r"```json|```", "", raw).strip()
    return json.loads(clean)


def get_protocol_text(corpus_path: Path, protocol_id: str) -> str | None:
    with corpus_path.open("r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            if obj.get("protocol_id") == protocol_id:
                return obj.get("text")
    return None


# -------- prompts --------

FINAL_SYSTEM = """
Ты — медицинский эксперт.
Используй ТОЛЬКО предоставленные протоколы.
Отвечай СТРОГО в JSON. Без пояснений вне JSON.
"""

FINAL_USER_TMPL = """
Запрос пользователя:
{query}

Релевантные протоколы:
{protocols_json}

Задание:
Определи наиболее вероятный диагноз.

Верни строго JSON:
{
  "icd_code": "код МКБ-10",
  "explanation": "краткое медицинское обоснование с указанием protocol_id"
}
"""


# -------- pipeline function --------

def analyze_protocol(
    corpus_path: str,
    protocol_id: str,
    query: str
) -> dict:
    """
    Full pipeline:
    protocol_id -> load text -> LLM -> parsed JSON
    """

    # 1) load protocol text
    text = get_protocol_text(Path(corpus_path), protocol_id)
    if text is None:
        raise ValueError(f"Protocol {protocol_id} not found")

    # 2) prepare input
    protocols_dict = {protocol_id: text}

    # 3) LLM call
    messages = [
        {"role": "system", "content": FINAL_SYSTEM},
        {"role": "user", "content": FINAL_USER_TMPL.format(
            query=query,
            protocols_json=json.dumps(protocols_dict, ensure_ascii=False)
        )}
    ]

    resp = generator.generate(
        messages=messages,
        reasoning_effort="medium",
        temperature=0.2,
        max_new_tokens=1024
    )

    # 4) parse + return
    return parse_llm_response(resp)


# -------- usage --------
# result = analyze_protocol(
#     generator,
#     "../data/corpus/protocols_corpus.jsonl",
#     "P_123456",
#     "Пациент с лихорадкой, кашлем и одышкой"
# )