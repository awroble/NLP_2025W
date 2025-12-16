
from typing import Sequence
import numpy as np
import torch
from numpy.typing import NDArray
class TestedModel:
    def predict(self,texts:NDArray[str])->float:
        pass
    def get_name(self)->str:
        pass


class ClassicalModel(TestedModel):
    def __init__(self,model,name):
        self.model=model
        self.name=name

    def predict(self,texts:NDArray[str])->Sequence[int]:
        return self.model.predict(texts).astype(int)
    def get_name(self)->str:
        return self.name

class BertModel(TestedModel):

    def __init__(self,model,tokenizer,name):
        self.model=model
        self.tokenizer=tokenizer
        self.name=name

    def predict(self,texts:NDArray[str])->Sequence[int]:
        t_vec=self.tokenizer(texts.tolist(),
                            return_tensors = "pt",
                            truncation = True,
                            padding = True,
                            )
        with torch.no_grad():
            result=self.model(**t_vec)
        return torch.argmax(result.logits, dim=1).detach().numpy()
    def get_name(self)->str:
        return self.name