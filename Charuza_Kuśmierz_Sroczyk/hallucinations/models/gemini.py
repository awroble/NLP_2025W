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
        async def _generate_one(prompt: str):
            response = await self.model.generate_content_async(contents=prompt)
            return response.text

        async def _generate_batch():
            tasks = [_generate_one(p) for p in prompts]
            results = await asyncio.gather(*tasks)
            return results

        return asyncio.run(_generate_batch())