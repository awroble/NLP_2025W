import random
import os
import platform
import psutil
import torch
import numpy as np
import logging


def set_seed(seed: int):
    """
    Sets seeds for all random number generators to ensure reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

    if torch.cuda.is_available():
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def log_system_info():
    """
    Logs computing infrastructure details.
    """
    logging.info("=== Reproducibility: System Information ===")
    logging.info(f"OS: {platform.system()} {platform.release()}")
    logging.info(f"Python Version: {platform.python_version()}")
    logging.info(f"CPU Cores: {os.cpu_count()}")
    logging.info(f"RAM: {round(psutil.virtual_memory().total / (1024.0 ** 3))} GB")

    if torch.cuda.is_available():
        logging.info(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        logging.info("GPU: Not available (Running on CPU)")
    logging.info("==========================================")