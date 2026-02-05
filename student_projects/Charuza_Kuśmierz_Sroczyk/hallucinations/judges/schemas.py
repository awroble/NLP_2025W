from typing import List, Optional
from pydantic import BaseModel, Field

class JudgeResult(BaseModel):
    # scores: 1 = match, 0 = mismatch, 2 = refusal, None = error
    scores: List[Optional[int]] = Field(..., description="Per-pair integer scores; 1=match, 0=mismatch, 2=refusal, None=error")