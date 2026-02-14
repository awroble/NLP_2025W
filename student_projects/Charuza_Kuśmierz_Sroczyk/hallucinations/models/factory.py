from .base import LLMProvider

class LLMFactory:
    _registry = {}

    @classmethod
    def register(cls, name: str):
        """Decorator to register a class"""
        def inner_wrapper(wrapped_class):
            cls._registry[name] = wrapped_class
            return wrapped_class
        return inner_wrapper

    @classmethod
    def available_models(cls):
        return list(cls._registry.keys())

    @classmethod
    def create(cls, name: str) -> LLMProvider:
        if name not in cls._registry:
            raise ValueError(f"Provider '{name}' not found.")
        return cls._registry[name]()