"""
Symptom extractor – uses the local GPT-OSS LLM to parse free-text into
a clean list of Russian symptom strings ready for vector search.
"""

from __future__ import annotations

import json
import re

from openai import OpenAI

SYSTEM = """
Ты — клинический NLP-ассистент.
Твоя задача — извлечь ВСЕ клинические симптомы из текста пациента.
Отвечай СТРОГО валидным JSON, без пояснений и без markdown.
"""

USER_TMPL = """
Текст пациента:
{query}

Инструкция:
1. Извлеки только симптомы, жалобы и клинические проявления.
2. НЕ включай диагнозы, причины, процедуры, лекарства, анализы.
3. Каждый симптом — краткая фраза на русском (1–5 слов).

Формат ответа (строго):
{{
  "symptoms": ["симптом 1", "симптом 2", ...]
}}

"""


class SymptomExtractor:
    def __init__(
        self,
        client: OpenAI,
        model: str = "gpt-oss-120b",
        temperature: float = 0.5,
        max_new_tokens: int = 1024,
    ):
        self.client         = client
        self.model          = model
        self.temperature    = temperature
        self.max_new_tokens = max_new_tokens

    # ------------------------------------------------------------------
    def extract(self, query: str) -> list[str]:
        """Return list of symptom strings extracted from raw user query."""
        messages = [
            {"role": "system", "content": SYSTEM},
            {"role": "user",   "content": USER_TMPL.format(query=query)},
        ]

        response = self.client.chat.completions.create(
            model       = self.model,
            messages    = messages,
            temperature = self.temperature,
            max_tokens  = self.max_new_tokens,
        )
        raw = response.choices[0].message.content or ""
        return self._parse(raw)

    # ------------------------------------------------------------------
    @staticmethod
    def _parse(raw: str) -> list[str]:
        # List of generic words the LLM often mistakenly includes as symptoms
        BLACKLIST = {
            "симптом", "симптомы", "symptoms", "жалобы", "проявления", 
            "клинические проявления", "текст пациента", "диагноз", "жалоба"
        }

        clean = re.sub(r"```json|```", "", raw).strip()
        symptoms = []

        try:
            data = json.loads(clean)
            symptoms = data.get("symptoms", [])
        except json.JSONDecodeError:
            # fallback: try to grab quoted strings
            symptoms = re.findall(r'"([^"]+)"', clean)

        # --- CLEANING LOGIC ---
        processed = []
        for s in symptoms:
            if not isinstance(s, str):
                continue
            
            s_clean = s.strip().lower()
            
            # 1. Skip if the string is just one of the blacklisted words
            if s_clean in BLACKLIST:
                continue
            
            # 2. Skip very short noise (e.g., ".", "и", "в")
            if len(s_clean) < 3:
                continue

            # 3. Skip if the LLM hallucinated the key name as a symptom (e.g. "symptoms:")
            if s_clean.endswith(':'):
                continue

            processed.append(s.strip())

        # Return unique symptoms only
        return list(dict.fromkeys(processed))
