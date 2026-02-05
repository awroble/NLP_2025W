import os
import asyncio
from typing import List
from google import genai
from .base import LLMProvider
from .factory import LLMFactory

@LLMFactory.register("gemini-2.5-flash")
class GeminiFlash2_5Adapter(LLMProvider):
    def __init__(self):
        super().__init__()
        genai.configure(api_key=os.environ["GEMINI_API_KEY"])
        self.model = genai.GenerativeModel("gemini-1.5-flash")

    def generate(self, prompt: str) -> str:
        response = self.model.generate_content(contents=prompt)
        return response.text

    def generate_batch(self, prompts: List[str]) -> List[str]:
        return super().generate_batch(prompts)
