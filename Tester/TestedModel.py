
from typing import Sequence
import numpy as np
class TestedModel:
    def predict(self,texts:Sequence[str])->float:
        pass
    def get_name(self)->str:
        pass


class ClassicalModel(TestedModel):
    def __init__(self,model,tokenizer,name):
        self.model=model
        self.tokenizer=tokenizer
        self.name=name

    def predict(self,texts:Sequence[str])->Sequence[int]:
        vectorized=self.tokenizer.transform(texts)
        return self.model.predict(vectorized)
    def get_name(self)->str:
        return self.name