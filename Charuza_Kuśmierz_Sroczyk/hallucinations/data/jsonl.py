import json
from pathlib import Path
from typing import List, Optional, Union, Dict

from .base import DataProvider
from .schemas import Sample
from .factory import DataFactory


@DataFactory.register("jsonl")
class JSONLDataProvider(DataProvider):
    """Load JSONL files where each line is a JSON object with keys:
    id, category, prompt, expected_response

    Parameters
    - data_dir: directory containing files like <category>.jsonl
    - categories_map: optional map of category -> filename (without .jsonl)
    """

    def __init__(self, data_dir: Union[str, Path] = "dummy_data", categories_map: Optional[Dict[str, str]] = None):
        self.data_dir = Path(data_dir)
        self._categories_map = categories_map or self._discover_categories()

    def _discover_categories(self) -> Dict[str, str]:
        mapping: Dict[str, str] = {}
        if not self.data_dir.exists():
            return mapping
        for p in self.data_dir.glob("*.jsonl"):
            # filename without suffix is the category name by default
            cat = p.stem
            mapping[cat] = cat
        return mapping

    def categories(self) -> List[str]:
        return sorted(self._categories_map.keys())

    def _resolve_targets(self, categories: Optional[Union[str, List[str]]]) -> List[str]:
        if categories is None:
            return self.categories()
        if isinstance(categories, str):
            categories = [categories]
        unknown = [c for c in categories if c not in self._categories_map]
        if unknown:
            raise ValueError(f"Unknown categories: {unknown}. Available: {self.categories()}")
        return categories

    def _iter_file(self, category: str):
        fname = self._categories_map[category] + ".jsonl"
        path = self.data_dir / fname
        if not path.exists():
            # If mapping present but file missing, skip gracefully
            return
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                # Ensure required fields and backfill category if missing
                obj.setdefault("category", category)
                yield obj

    def load(
        self,
        categories: Optional[Union[str, List[str]]] = None,
        limit: Optional[int] = None,
    ) -> List[Sample]:
        targets = self._resolve_targets(categories)
        out: List[Sample] = []
        for cat in targets:
            for obj in self._iter_file(cat):
                try:
                    sample = Sample.model_validate(obj)
                except Exception:
                    # Skip invalid rows
                    continue
                out.append(sample)
                if limit is not None and len(out) >= limit:
                    return out
        return out
