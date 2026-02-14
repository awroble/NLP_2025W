"""
LLaDA Safety Evaluation Script

Runs safety evaluations on LLaDA-8B-Instruct using the LSB framework.

Usage:
    python run_evaluation.py --output results/llada
    python run_evaluation.py --steps 256 --judge Qwen/Qwen2.5-3B-Instruct
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime
import logging
import torch
import transformers
import traceback

# Check version
VERSION = transformers.__version__
REQUIRED_VERSION = "4.38.2"

if VERSION != REQUIRED_VERSION:
    print(f"\nERROR: LLaDA requires transformers=={REQUIRED_VERSION}")
    print(f"   Current version: {VERSION}")
    print(f"\n   To fix: pip install -r requirements.txt")
    sys.exit(1)

from adapters import create_llada_evaluator
from reproducibility import (
    set_seed,
    check_reproducibility,
    create_run_config,
    save_config,
    get_environment_info
)

# Setup paths
DATA_DIR = Path(__file__).parent.parent.parent / "data"
RESULTS_DIR = Path(__file__).parent.parent.parent / "results"


def get_prompt_files(domains=None):
    """Get list of prompt files to evaluate."""
    available = {
        "health": DATA_DIR / "prompts_health.json",
        "misinformation": DATA_DIR / "prompts_misinformation.json",
        "disinformation": DATA_DIR / "prompts_disinformation.json"
    }

    if domains is None:
        domains = list(available.keys())

    files = []
    for domain in domains:
        if domain not in available:
            raise ValueError(f"Unknown domain: {domain}")
        if not available[domain].exists():
            raise FileNotFoundError(f"Dataset not found: {available[domain]}")
        files.append(available[domain])

    return files


def main():
    parser = argparse.ArgumentParser(description="Evaluate LLaDA on LSB safety benchmark")

    parser.add_argument("--model-path", default="GSAI-ML/LLaDA-8B-Instruct")
    parser.add_argument("--device", default=None)
    parser.add_argument("--steps", type=int, default=128)
    parser.add_argument("--gen-length", type=int, default=128)
    parser.add_argument("--block-length", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility (recommended)")
    parser.add_argument("--strict-deterministic", action="store_true", help="Enable STRICT deterministic algorithms (VERY SLOW - not recommended)")
    parser.add_argument("--judge", default=None, help="Judge model (e.g., Qwen/Qwen2.5-3B-Instruct)")
    parser.add_argument("--domains", nargs="+", choices=["health", "misinformation", "disinformation"], default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--log-level", default="INFO")

    args = parser.parse_args()

    if args.output is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output = str(RESULTS_DIR / f"llada_{timestamp}")
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    log_file = output_dir / f"evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    logger = logging.getLogger(__name__)

    logger.info("="*70)
    logger.info("LLaDA Safety Evaluation")
    logger.info("="*70)
    logger.info(f"Model: {args.model_path}")
    logger.info(f"Device: {args.device or 'auto-detect'}")
    logger.info(f"Steps: {args.steps}")
    logger.info(f"Generation length: {args.gen_length}")
    logger.info(f"Block length: {args.block_length}")
    logger.info(f"Temperature: {args.temperature}")
    logger.info(f"Seed: {args.seed if args.seed is not None else 'Not set (not reproducible)'}")
    logger.info(f"Strict deterministic: {args.strict_deterministic}")
    logger.info(f"Judge: {args.judge or 'None'}")
    logger.info(f"Domains: {args.domains or 'All'}")
    logger.info(f"Output: {output_dir}")
    logger.info("="*70)

    if args.seed is not None:
        set_seed(args.seed, use_deterministic_algorithms=args.strict_deterministic)

    check_reproducibility(temperature=args.temperature, seed=args.seed)

    try:
        # Create evaluator
        logger.info("Initializing evaluator...")
        evaluator = create_llada_evaluator(
            model_path=args.model_path,
            device=args.device,
            steps=args.steps,
            gen_length=args.gen_length,
            block_length=args.block_length,
            temperature=args.temperature,
            judge_model_name=args.judge,
            log_level=args.log_level
        )
        logger.info("Evaluator initialized")

        prompt_files = get_prompt_files(args.domains)
        logger.info(f"Evaluating {len(prompt_files)} domains")

        run_config = create_run_config(
            model_name=args.model_path,
            model_type="llada",
            device=evaluator.device,
            steps=args.steps,
            gen_length=args.gen_length,
            block_length=args.block_length,
            temperature=args.temperature,
            seed=args.seed,
            judge_model=args.judge,
            domains=args.domains or ["health", "misinformation", "disinformation"]
        )
        save_config(run_config, output_dir / "config.json")

        logger.info("Starting evaluation...")
        evaluator.run_evaluation(
            prompts_files=[str(f) for f in prompt_files],
            output_dir=str(output_dir)
        )

        logger.info("="*70)
        logger.info("Evaluation Complete!")
        logger.info(f"Results: {output_dir}")
        logger.info(f"Configuration: {output_dir / 'config.json'}")
        logger.info("="*70)

        return 0

    except Exception as e:
        logger.error(f"Evaluation failed: {type(e).__name__}: {str(e)}")
        logger.error(traceback.format_exc())
        return 1


if __name__ == "__main__":
    sys.exit(main())
