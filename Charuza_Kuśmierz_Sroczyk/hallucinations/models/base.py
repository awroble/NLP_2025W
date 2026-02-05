from abc import ABC, abstractmethod
import os
from typing import List
import concurrent.futures
from tqdm import tqdm

class LLMProvider(ABC):
    @abstractmethod
    def generate(self, prompt: str) -> str:
        pass

    # New: batch generation using thread pool and existing generate()
    def generate_batch(self, prompts: List[str]) -> List[str]:
        """
        Invoke self.generate for each prompt in parallel, preserve order.
        Exceptions from individual generate calls are captured and returned
        as a string starting with "ERROR:" for that prompt.
        """
        if not prompts:
            return []

        max_workers = min(32, len(prompts))
        results: List[str] = ["" for _ in prompts]
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
            future_to_index = {ex.submit(self.generate, p): i for i, p in enumerate(prompts)}
            for future in tqdm(concurrent.futures.as_completed(future_to_index), total=len(prompts), desc="Generating"):
                index = future_to_index[future]
                try:
                    results[index] = future.result()
                except Exception as e:
                    results[index] = f"ERROR: {e}"
        return results