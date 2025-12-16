from typing import List
from .base import Judge
from .factory import JudgeFactory
from .schemas import JudgeResult


def _normalize(text: str) -> str:
    return " ".join(text.strip().lower().split())


@JudgeFactory.register("simple")
class SimpleJudge(Judge):
    """Heuristic judge: token-normalized equality and boolean equivalence.

    Rules:
    - exact normalized match => equivalent=True
    - else use a simple overlap heuristic; equivalent=True if overlap ratio >= 0.8
    - score is always boolean and mirrors `equivalent`
    """

    def compare(self, answers_a: List[str], answers_b: List[str], prompts: List[str]) -> JudgeResult:
        if not (len(answers_a) == len(answers_b) == len(prompts)):
            raise ValueError("Answer and prompt lists must be the same length")

        results: List[bool] = []

        for a, b in zip(answers_a, answers_b):
            na, nb = _normalize(a), _normalize(b)
            if na == nb:
                eq = True
            else:
                # very simple similarity: overlap of characters / max length
                overlap = len(set(na) & set(nb))
                denom = max(len(na), len(nb)) or 1
                ratio = overlap / denom
                eq = ratio >= 0.8

            results.append(eq)

        return JudgeResult(scores=results)
