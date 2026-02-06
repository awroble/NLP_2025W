from typing import List, Optional
from pydantic import BaseModel

class EvaluationResultItem(BaseModel):
    model: str
    sample_id: str
    score: Optional[int]
    prompt: str
    expected_response: str
    model_response: str

class EvaluationOutput(BaseModel):
    results: List[EvaluationResultItem]
