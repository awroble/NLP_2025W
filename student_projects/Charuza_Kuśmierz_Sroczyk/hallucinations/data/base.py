from abc import ABC, abstractmethod
from typing import List, Optional, Union
from .schemas import Sample


class DataProvider(ABC):
    """Abstract data provider interface.

    Contract:
        - categories(): returns available category names
        - load(categories=None, limit=None): returns a list of `Sample` pydantic models
            with fields: id, category, prompt, expected_response
    """

    @abstractmethod
    def categories(self) -> List[str]:
        pass

    @abstractmethod
    def load(
        self,
        categories: Optional[Union[str, List[str]]] = None,
        limit: Optional[int] = None,
    ) -> List[Sample]:
        pass
