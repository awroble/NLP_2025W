from abc import ABC, abstractmethod
from typing import List
from .schemas import JudgeResult


class Judge(ABC):
    """Abstract judge interface.

    Contract:
    - compare(answers_a, answers_b, prompts): returns a `JudgeResult` which is a list of
      booleans (one per pair) indicating equivalence
    """

    @abstractmethod
    def compare(self, answers_a: List[str], answers_b: List[str], prompts: List[str]) -> JudgeResult:
        pass
