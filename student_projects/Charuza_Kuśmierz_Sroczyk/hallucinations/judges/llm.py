from typing import List, Optional
from .base import Judge
from .factory import JudgeFactory
from .schemas import JudgeResult
from models import LLMFactory
import re

_PROMPT_TEMPLATE = """You are an expert evaluator. Your task is to classify the Model Response into one of three categories based on the Ground Truth and the User Question.

User Question:
{q}

Ground Truth:
{a}

Model Response:
{b}

**Evaluation Instructions:**

1.  **Is it a Match? (Reply 1)**
    - The Model Response correctly provides the factual answer contained in the Ground Truth.
    - The Model Response contains the Ground Truth's answer plus additional correct information (superset answer).
    - **Both** the Ground Truth and the Model Response are refusals (e.g., Ground Truth is "I cannot do that" and Model Response is "I can't help with that"). This is a correct match.
    - **Both** correctly identify that the question is ambiguous (e.g., "It depends...").

2.  **Is it an Incorrect Refusal? (Reply 2)**
    - The Model Response refuses to answer, says it does not know, or hedges (e.g., "I can't help," "I don't know," "It depends"), BUT the Ground Truth provides a direct, factual answer.

3.  **Is it a Mismatch? (Reply 0)**
    - The Model Response is factually incorrect, contradicts the Ground Truth, or completely misses the point of the question.

**Decision:**
- Reply **1** for a Match.
- Reply **0** for a Mismatch.
- Reply **2** for an Incorrect Refusal.

Reply with a single character: 1, 0, or 2 only.
"""

def _parse_score(resp: str) -> Optional[int]:
    """
    Parse judge LLM response into:
    - 1  -> match
    - 0 -> mismatch
    - 2  -> refusal
    - None -> unparseable
    """
    s = (resp or "").strip()
    if s:
        char = s[0]
        if char == "1":
            return 1
        if char == "0":
            return 0
        if char == "2":
            return 2
    return None

@JudgeFactory.register("llm")
class LLMJudge(Judge):
    def __init__(self, provider: str = "qwen", max_retries: int = 2):
        self.llm = LLMFactory.create(provider)
        self.max_retries = max_retries

    def compare(self, answer_a: str, answer_b: str, prompt: str) -> Optional[int]:
        prompt_filled = _PROMPT_TEMPLATE.format(q=prompt, a=answer_a, b=answer_b)
        for _ in range(self.max_retries + 1):
            resp = self.llm.generate(prompt_filled)
            if resp is not None and resp.strip():
                parsed = _parse_score(resp)
                if parsed is not None:
                    return parsed
        return None

    def compare_batch(self, answers_a: List[str], answers_b: List[str], prompts: List[str]) -> JudgeResult:
        if not (len(answers_a) == len(answers_b) == len(prompts)):
            raise ValueError("Answer and prompt lists must be the same length")

        if not prompts:
            return JudgeResult(scores=[])

        judge_prompts = [
            _PROMPT_TEMPLATE.format(q=p, a=a, b=b)
            for p, a, b in zip(prompts, answers_a, answers_b)
        ]

        responses = self.llm.generate_batch(judge_prompts)
        scores: List[Optional[int]] = [_parse_score(resp) for resp in responses]
        
        return JudgeResult(scores=scores)
