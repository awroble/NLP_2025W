from typing import List,Dict,Sequence
from  TestedModel import TestedModel

class ModelsTestingSuite:
    def __init__(self):
        self.models:List[TestedModel] = []

    def add(self, model: TestedModel):
        self.models.append(model)
    def evaluate(self, metric:str) -> Dict[str, Sequence[float]]:
        return None