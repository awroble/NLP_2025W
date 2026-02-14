"""
Reproducibility utilities for diffusion LLM evaluation.

Ensures deterministic behavior and proper experiment tracking.

Authors: Based on work by Kisiel, Kosakowski, Franczak, and Koniecko
Institution: Warsaw University of Technology, NLP Course Winter 2025
"""

import random
import os
import json
import logging
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime
import platform
import sys

import numpy as np
import torch
import transformers


logger = logging.getLogger(__name__)


def set_seed(seed: int = 42, use_deterministic_algorithms: bool = False):
    """
    Set random seeds for reproducibility.

    Args:
        seed: Random seed value
        use_deterministic_algorithms: If True, enables strict deterministic algorithms
                                     (VERY SLOW - not recommended for diffusion models)

    Note:
        Setting seed alone provides good reproducibility for most cases.
        Full deterministic mode adds 10-30% overhead and is rarely needed.

        For diffusion models with temperature=0, seed alone is usually sufficient.
    """
    logger.info(f"Setting random seed: {seed}")

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    logger.info("✓ Random seeds set, cuDNN benchmark disabled")

    if use_deterministic_algorithms:
        logger.warning(
            "Enabling STRICT deterministic mode - this is VERY SLOW (10-30% overhead). "
            "Only use for debugging reproducibility issues."
        )

        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.deterministic = True
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

        logger.info("Strict deterministic algorithms enabled")
    else:
        logger.info("Strict deterministic mode OFF (recommended for performance)")


def get_environment_info() -> Dict[str, Any]:
    """
    Collect environment and version information.

    Returns:
        Dictionary with system information
    """

    info = {
        "timestamp": datetime.now().isoformat(),
        "python_version": sys.version,
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "numpy_version": np.__version__,
        "cuda_available": torch.cuda.is_available(),
    }

    if torch.cuda.is_available():
        info.update({
            "cuda_version": torch.version.cuda,
            "cudnn_version": torch.backends.cudnn.version(),
            "gpu_count": torch.cuda.device_count(),
            "gpu_names": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
        })

    return info


def save_config(
    config: Dict[str, Any],
    output_file: Path,
    include_env: bool = True
) -> None:
    """
    Save configuration to JSON file for reproducibility.

    Args:
        config: Configuration dictionary
        output_file: Path to save config
        include_env: Include environment information
    """
    full_config = config.copy()

    if include_env:
        full_config["environment"] = get_environment_info()

    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(full_config, f, indent=2, default=str)

    logger.info(f"Configuration saved to {output_file}")


def create_run_config(
    model_name: str,
    model_type: str,
    device: str,
    steps: int,
    gen_length: int,
    block_length: int,
    temperature: float,
    seed: Optional[int],
    judge_model: Optional[str],
    domains: list,
    **kwargs
) -> Dict[str, Any]:
    """
    Create standardized configuration dictionary.

    Args:
        model_name: Model identifier
        model_type: Type of model ('llada' or 'mmada')
        device: Computation device
        steps: Diffusion steps
        gen_length: Generation length
        block_length: Block length
        temperature: Sampling temperature
        seed: Random seed (None if not set)
        judge_model: Judge model name
        domains: Evaluation domains
        **kwargs: Additional parameters

    Returns:
        Configuration dictionary
    """
    config = {
        "model": {
            "name": model_name,
            "type": model_type,
        },
        "generation": {
            "steps": steps,
            "gen_length": gen_length,
            "block_length": block_length,
            "temperature": temperature,
        },
        "evaluation": {
            "judge_model": judge_model,
            "domains": domains,
        },
        "system": {
            "device": device,
            "seed": seed,
        },
        "additional": kwargs
    }

    return config


def check_reproducibility(temperature: float, seed: Optional[int]) -> None:
    """
    Check and warn about reproducibility settings.

    Args:
        temperature: Sampling temperature
        seed: Random seed (None if not set)
    """
    issues = []

    if seed is None:
        issues.append("No random seed set - results will vary between runs")

    if temperature > 0:
        issues.append(f"Temperature={temperature} > 0 - sampling is stochastic")

    if not issues:
        logger.info("✓ Reproducibility: GOOD (seed set, temperature=0)")
        if torch.backends.cudnn.deterministic:
            logger.info("  Note: Strict deterministic mode enabled (slower)")
        else:
            logger.info("  Note: Using seed-based reproducibility (recommended)")
    else:
        logger.warning("⚠ Reproducibility warnings:")
        for issue in issues:
            logger.warning(f"  - {issue}")
        logger.warning("To ensure reproducibility:")
        if seed is None:
            logger.warning("  1. Add --seed 42 (or any fixed number)")
        if temperature > 0:
            logger.warning("  2. Use --temperature 0.0 for deterministic sampling")
