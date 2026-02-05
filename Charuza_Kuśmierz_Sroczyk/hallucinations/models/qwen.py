import os
from typing import List
from .base import LLMProvider
from huggingface_hub import InferenceClient
from .factory import LLMFactory

@LLMFactory.register("qwen")
class QwenAdapter(LLMProvider):
    def __init__(self):
        self.client = InferenceClient(api_key=os.environ["HF_TOKEN"])
        self.model = "Qwen/Qwen3-8B:nscale"

    def generate(self, prompt: str) -> str:
        completion = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}]
        )
        return completion.choices[0].message.content

    def generate_batch(self, prompts: List[str]) -> List[str]:
        return super().generate_batch(prompts)