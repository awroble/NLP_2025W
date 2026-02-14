from typing import Optional
from pydantic import BaseModel, Field

class Sample(BaseModel):
    id: str = Field(..., description="Unique sample identifier")
    category: str = Field(..., description="Category name")
    prompt: str = Field(..., description="Prompt text for the LLM")
    expected_response: str = Field(..., description="Reference answer for evaluation")
