from abc import ABC, abstractmethod
import requests
import base64
import logging
from typing import Optional

class LLM(ABC):
    """Abstract Base Class for Language Models."""
    @abstractmethod
    def generate(self, prompt: str, image_path: str = None, **kwargs) -> str:
        pass

class OllamaModel(LLM):
    """
    Concrete implementation for local Ollama models.
    Supports both text-only and multimodal (LLaVA) generation.
    """
    def __init__(self, model_name: str, api_url: str = "http://localhost:11434/api/generate"):
        self.model_name = model_name
        self.api_url = api_url

    def generate(self, prompt: str, image_path: Optional[str] = None, **kwargs) -> str:
        # Pass parameters like temperature via kwargs options
        data = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "options": kwargs
        }

        # Handle multimodal input (Visual Prompt Injection)
        if image_path:
            try:
                with open(image_path, "rb") as img_file:
                    encoded_string = base64.b64encode(img_file.read()).decode('utf-8')
                data["images"] = [encoded_string]
            except FileNotFoundError:
                logging.error(f"Image not found: {image_path}")
                return "ERROR: Image file missing"

        try:
            response = requests.post(self.api_url, json=data)
            response.raise_for_status()
            return response.json().get("response", "")
        except Exception as e:
            logging.error(f"Inference Error for {self.model_name}: {e}")
            return f"ERROR: {str(e)}"

class ModelFactory:
    """Factory to create model instances based on configuration."""
    @staticmethod
    def create_model(model_name: str) -> LLM:
        # Easy to extend for GPT-4/Claude in the future
        return OllamaModel(model_name)