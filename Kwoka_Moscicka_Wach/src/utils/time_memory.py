import torch
import time
from functools import wraps

def profile_gpu(func):
    """
    Decorator to measure GPU time and peak memory for a function.
    """
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        # GPU warmup / reset
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()

        start = time.perf_counter()
        output = func(self, *args, **kwargs)
        end = time.perf_counter()

        metrics = {
            "latency_sec": end - start,
            "gpu_peak_mb": torch.cuda.max_memory_allocated() / 1e6 if torch.cuda.is_available() else None
        }

        if isinstance(output, dict):
            output.setdefault("metrics", {}).update(metrics)
        else:
            output = {"output": output, "metrics": metrics}

        return output

    return wrapper
