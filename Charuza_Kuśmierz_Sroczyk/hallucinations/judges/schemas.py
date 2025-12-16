from typing import List, Optional
from pydantic import BaseModel, Field

class JudgeResult(BaseModel):
    scores: List[bool] = Field(..., description="Per-pair boolean equivalence scores")