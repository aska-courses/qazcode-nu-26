"""
Explanation generator – takes top-N scored diagnoses and generates
a clinical explanation grounded in the evidence from the graph.
"""

from __future__ import annotations

import json
import re

from openai import OpenAI

SYSTEM = """
Ты — клинический ИИ-ассистент казахстанской системы здравоохранения.
Объясняй диагнозы кратко, точно, со ссылкой на клинические признаки.
Отвечай строго на русском языке. Отвечай ТОЛЬКО валидным JSON без markdown.
"""

USER_TMPL = """
Симптомы пациента: {symptoms}

Кандидаты диагнозов (ранжированы по вероятности):
{candidates}

Для каждого диагноза из списка сгенерируй краткое клиническое объяснение (2-3 предложения),
опираясь на представленные доказательства и казахстанские клинические протоколы.

Верни JSON строго в формате:
{{
  "explanations": {{
    "<uid>": "объяснение..."
  }}
}}
"""


class ExplanationGenerator:
    def __init__(
        self,
        client: OpenAI,
        model: str = "gpt-oss-120b",
        temperature: float = 0.5,
        max_new_tokens: int = 2048,
    ):
        self.client         = client
        self.model          = model
        self.temperature    = temperature
        self.max_new_tokens = max_new_tokens

    # ------------------------------------------------------------------
    def generate(self, symptoms: list[str], candidates: list[dict]) -> dict[str, str]:
        """
        Args:
            symptoms:   original extracted symptoms
            candidates: list of dicts from GraphScorer (rank, uid, diagnosis, …)
        Returns:
            {uid: explanation_str}
        """
        cand_text = "\n".join(
            f"  {c['rank']}. {c['diagnosis']} [{c['icd10_code']}] "
            f"(score={c['score']}) | evidence: {'; '.join(c['evidence'][:2]) or 'нет'}"
            for c in candidates
        )

        messages = [
            {"role": "system", "content": SYSTEM},
            {
                "role": "user",
                "content": USER_TMPL.format(
                    symptoms=", ".join(symptoms),
                    candidates=cand_text,
                ),
            },
        ]

        response = self.client.chat.completions.create(
            model       = self.model,
            messages    = messages,
            temperature = self.temperature,
            max_tokens  = self.max_new_tokens,
        )
        raw = response.choices[0].message.content or ""
        return self._parse(raw, candidates)

    # ------------------------------------------------------------------
    @staticmethod
    def _parse(raw: str, candidates: list[dict]) -> dict[str, str]:
        clean = re.sub(r"```json|```", "", raw).strip()
        try:
            data = json.loads(clean)
            return data.get("explanations", {})
        except json.JSONDecodeError:
            # fallback: empty explanations
            return {c["uid"]: "" for c in candidates}
