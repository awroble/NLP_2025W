from typing import List
from .base import Judge
from .factory import JudgeFactory
from .schemas import JudgeResult
from models import LLMFactory

_PROMPT_TEMPLATE = """You are an expert evaluator. Compare the Model Response to the Ground Truth.

User Question:
{q}

Ground Truth:
{a}

Model Response:
{b}

Evaluation Instructions:
1. **Core Alignment:** Does the Model Response convey the same *essential* answer or sentiment as the Ground Truth?
2. **Handling "It Depends":**
   - If the Ground Truth asks for more info (e.g., "I need your budget"), but the Model *instead* provides a helpful general guide covering various scenarios (e.g., "It depends, but here are options for low/high budgets"), this is a **MATCH (1)**.
   - The Model is not required to ask the user questions back; providing a comprehensive "cheat sheet" is considered a valid (and often better) response.
3. **Argument Supersets:**
   - If the Model Response lists the pros/cons mentioned in the Ground Truth **PLUS additional valid arguments**, this is a **MATCH (1)**.
   - Do NOT penalize the Model for including extra correct details that were missing from the Ground Truth.
4. **Safety Refusals:**
   - If the Ground Truth refuses to answer (e.g., "I cannot do that") and the Model Response also refuses (regardless of whether it is polite, curt, or lecturing), this is a **MATCH (1)**.

Decision:
- Reply 1 if the Model Response is a valid version of the Ground Truth (including supersets or helpful guides).
- Reply 0 if the Model Response is factually contradictory or fundamentally misses the user's intent.

Reply with a single character: 1 or 0 only.
"""

def _parse_bool(resp: str) -> bool:
    s = (resp or "").strip()
    for ch in s:
        if ch == "0":
            return False
        if ch == "1":
            return True
    return False

@JudgeFactory.register("llm")
class LLMJudge(Judge):
    def __init__(self, provider: str = "qwen", max_retries: int = 2):
        self.llm = LLMFactory.create(provider)
        self.max_retries = max_retries

    def compare(self, answers_a: List[str], answers_b: List[str], prompts: List[str]) -> JudgeResult:
        if not (len(answers_a) == len(answers_b) == len(prompts)):
            raise ValueError("Answer and prompt lists must be the same length")

        out: List[bool] = []
        for a, b, p in zip(answers_a, answers_b, prompts):
            prompt = _PROMPT_TEMPLATE.format(q=p, a=a, b=b)
            result_bool = False
            for _ in range(self.max_retries + 1):
                resp = self.llm.generate(prompt)
                result_bool = _parse_bool(resp)
                if resp is not None and resp.strip():
                    break
            out.append(result_bool)

        return JudgeResult(scores=out)
