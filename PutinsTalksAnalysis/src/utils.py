import time
import tracemalloc
import logging
import random
import numpy as np
import torch
import os
import json
from datetime import datetime


class ResourceTracker:
    """
    Context manager to measure execution time and peak memory usage.
    """

    def __init__(self, operation_name="Operation"):
        self.operation_name = operation_name

    def __enter__(self):
        self.start_time = time.time()
        tracemalloc.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        self.end_time = time.time()
        self.duration = self.end_time - self.start_time
        self.peak_memory_mb = peak / 10 ** 6

        print(f"[{self.operation_name}] Finished.")
        print(f"   Duration: {self.duration:.4f} seconds")
        print(f"   Peak Memory: {self.peak_memory_mb:.2f} MB")


def set_global_seed(seed=42):
    """Sets random seed for Python, NumPy, and PyTorch for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    print(f"Global seed set to: {seed}")


class ExperimentLogger:
    """Logs experiment details to a JSON file."""

    def __init__(self, log_dir="reports", experiment_name="experiment"):
        os.makedirs(log_dir, exist_ok=True)
        self.log_file = os.path.join(log_dir, f"{experiment_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        self.log_data = {
            "timestamp": datetime.now().isoformat(),
            "operations": []
        }
        print(f"Logging experiments to: {self.log_file}")

    def log_operation(self, name, duration, memory_mb, metrics=None):
        entry = {
            "name": name,
            "duration_sec": duration,
            "memory_mb": memory_mb,
            "metrics": metrics or {}
        }
        self.log_data["operations"].append(entry)
        self._save()

    def _save(self):
        with open(self.log_file, 'w') as f:
            json.dump(self.log_data, f, indent=4)
