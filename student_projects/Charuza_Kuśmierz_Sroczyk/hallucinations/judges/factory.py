from .base import Judge

class JudgeFactory:
    _registry = {}

    @classmethod
    def register(cls, name: str):
        """Decorator to register a judge class"""
        def inner_wrapper(wrapped_class):
            cls._registry[name] = wrapped_class
            return wrapped_class
        return inner_wrapper

    @classmethod
    def create(cls, name: str, **kwargs) -> Judge:
        if name not in cls._registry:
            raise ValueError(f"Judge '{name}' not found.")
        return cls._registry[name](**kwargs)
