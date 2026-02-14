from abc import ABC, abstractmethod
from typing import List, Optional
from .schemas import JudgeResult
import concurrent.futures
from tqdm import tqdm


class Judge(ABC):
    """Abstract judge interface.

    Contract:
    - compare(answer_a, answer_b, prompt): returns Optional[int] indicating equivalence (1),
      mismatch (0), refusal (2), or error (None).
    - compare_batch(answers_a, answers_b, prompts): returns a `JudgeResult` which is a list of
      Optional[int] (one per pair), processed in parallel.
    """

    @abstractmethod
    def compare(self, answer_a: str, answer_b: str, prompt: str) -> Optional[int]:
        pass

    def compare_batch(self, answers_a: List[str], answers_b: List[str], prompts: List[str]) -> JudgeResult:
        if not (len(answers_a) == len(answers_b) == len(prompts)):
            raise ValueError("Answer and prompt lists must be the same length")

        if not prompts:
            return JudgeResult(scores=[])

        max_workers = min(32, len(prompts))
        scores: List[Optional[int]] = [None] * len(prompts)

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_index = {
                executor.submit(self.compare, a, b, p): i
                for i, (a, b, p) in enumerate(zip(answers_a, answers_b, prompts))
            }

            for future in tqdm(concurrent.futures.as_completed(future_to_index), total=len(prompts), desc="Judging"):
                index = future_to_index[future]
                try:
                    scores[index] = future.result()
                except Exception:
                    scores[index] = None

        return JudgeResult(scores=scores)
