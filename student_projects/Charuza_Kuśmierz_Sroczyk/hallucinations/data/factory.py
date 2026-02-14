from .base import DataProvider


class DataFactory:
    _registry = {}

    @classmethod
    def register(cls, name: str):
        """Decorator to register a data provider class"""

        def inner_wrapper(wrapped_class):
            cls._registry[name] = wrapped_class
            return wrapped_class

        return inner_wrapper

    @classmethod
    def create(cls, name: str, **kwargs) -> DataProvider:
        if name not in cls._registry:
            raise ValueError(f"Data provider '{name}' not found.")
        return cls._registry[name](**kwargs)
