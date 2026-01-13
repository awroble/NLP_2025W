from typing import List, Dict
from TestedModel import TestedModel
from numpy.typing import NDArray
import numpy as np
import sklearn.metrics as metrics
import time
import torch


def evaluate_metric(y_pred, y_true, metric: str):
    match metric:
        case "accuracy":
            return metrics.accuracy_score(y_true, y_pred)
        case "f1_score":
            return metrics.f1_score(y_true, y_pred)
        case "mcc":
            return metrics.matthews_corrcoef(y_true, y_pred)
        case _:
            raise ValueError(f"Invalid metric: {metric}")


class ModelsTestingSuite:


    def __init__(self, X: NDArray, y: NDArray):
        self.models: List[TestedModel] = []
        self.X: NDArray = X
        self.y: NDArray = y

    def change_ds(self, X: NDArray, y: NDArray):
        self.X = X
        self.y = y

    def add(self, model: TestedModel):
        self.models.append(model)

    def _init_results(self, batch_nr: int) -> Dict[str, np.ndarray]:
        return {model.get_name(): np.zeros(batch_nr) for model in self.models}

    def _init_splits(self, batch_nr: int, seed: int):
        N = self.X.shape[0]
        idx = np.arange(N)
        np.random.seed(seed)
        np.random.shuffle(idx)
        return np.array_split(idx, batch_nr)

    def evaluate(self, metric: str, batch_nr: int, seed: int = 42) -> Dict[str, np.ndarray]:

        results = self._init_results(batch_nr)
        splits = self._init_splits(batch_nr, seed)

        for i, split in enumerate(splits):
            X_b = self.X[split]
            y_b = self.y[split]

            for model in self.models:
                y_pred = model.predict(X_b)
                results[model.get_name()][i] = evaluate_metric(
                    y_pred, y_b, metric
                )

            print(f"Batch {i + 1}/{batch_nr} finished")

        return results

    def measure_time(self, batch_nr: int, seed: int = 42) -> Dict[str, np.ndarray]:

        results = self._init_results(batch_nr)
        splits = self._init_splits(batch_nr, seed)

        for i, split in enumerate(splits):
            X_b = self.X[split]

            for model in self.models:
                if torch.cuda.is_available():
                    torch.cuda.synchronize()

                start_time = time.perf_counter()
                model.predict(X_b)

                if torch.cuda.is_available():
                    torch.cuda.synchronize()

                end_time = time.perf_counter()
                results[model.get_name()][i] = end_time - start_time

            print(f"Batch {i + 1}/{batch_nr} finished")

        return results
