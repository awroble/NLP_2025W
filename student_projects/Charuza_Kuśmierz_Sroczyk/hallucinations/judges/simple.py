from typing import List, Optional
from .base import Judge
from .factory import JudgeFactory
from .schemas import JudgeResult


def _normalize(text: str) -> str:
	return " ".join(text.strip().lower().split())


@JudgeFactory.register("simple")
class SimpleJudge(Judge):
	"""Heuristic judge: token-normalized equality and boolean equivalence.

	Rules:
	- exact normalized match => equivalent=1
	- else use a simple overlap heuristic; equivalent=1 if overlap ratio >= 0.8, else 0
	- score is always 0 or 1.
	"""

	def compare(self, answer_a: str, answer_b: str, prompt: str) -> Optional[int]:
		na, nb = _normalize(answer_a), _normalize(answer_b)
		if na == nb:
			return 1
		else:
			# very simple similarity: overlap of characters / max length
			overlap = len(set(na) & set(nb))
			denom = max(len(na), len(nb)) or 1
			ratio = overlap / denom
			return 1 if ratio >= 0.8 else 0

	def compare_batch(self, answers_a: List[str], answers_b: List[str], prompts: List[str]) -> JudgeResult:
		return super().compare_batch(answers_a, answers_b, prompts)
