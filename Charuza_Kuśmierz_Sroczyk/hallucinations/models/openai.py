import os
from openai import OpenAI
from .base import LLMProvider
from .factory import LLMFactory

@LLMFactory.register("gpt-5-nano")
class OpenAIAdapter(LLMProvider):
    def __init__(self):
        self.client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        self.model = "gpt-5-nano"

    def generate(self, prompt: str) -> str:
        completion = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}]
        )
        return completion.choices[0].message.content

@LLMFactory.register("gpt-5-mini")
class OpenAIAdapter(LLMProvider):
    def __init__(self):
        self.client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        self.model = "gpt-5-mini"

    def generate(self, prompt: str) -> str:
        completion = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}]
        )
        return completion.choices[0].message.content
