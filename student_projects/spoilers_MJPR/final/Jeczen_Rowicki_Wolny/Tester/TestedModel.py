
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

    def __init__(self, model, tokenizer, name: str):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.tokenizer = tokenizer
        self.name = name

        self.model.eval()

    def predict(self, texts: NDArray[str]) -> Sequence[int]:
        inputs = self.tokenizer(
            texts.tolist(),
            return_tensors="pt",
            truncation=True,
            padding=True,
        )

        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)

        preds = torch.argmax(outputs.logits, dim=1)

        return preds.cpu().numpy()

    def get_name(self) -> str:
        return self.name