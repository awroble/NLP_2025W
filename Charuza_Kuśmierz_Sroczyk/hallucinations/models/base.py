from abc import ABC, abstractmethod
import os

class LLMProvider(ABC):
    @abstractmethod
    def generate(self, prompt: str) -> str:
        pass