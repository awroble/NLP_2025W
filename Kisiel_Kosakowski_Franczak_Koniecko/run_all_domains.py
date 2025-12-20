#!/usr/bin/env python3
"""
Convenience script to run evaluation on all three domain JSON files at once.
This creates a unified test across health, misinformation, and disinformation domains.

Can run for a single model or multiple models separately.

Usage:
    # Single model (default)
    python run_all_domains.py --model "Qwen/Qwen2.5-1.5B-Instruct"
    
    # Multiple models separately
    python run_all_domains.py --models "Qwen/Qwen2.5-0.5B-Instruct" "Qwen/Qwen2.5-1.5B-Instruct"
    
    # With additional options
    python run_all_domains.py --models "Model1" "Model2" --temperature 0.7
"""

import argparse
import subprocess
import sys
import os
from datetime import datetime

def run_unified_evaluation(model_name, files, output_dir, extra_args):
    """
    Run unified evaluation (all three domains together) for a single model.
    
    Args:
        model_name: Name of the model to evaluate
        files: List of prompt JSON files (all three domains)
        output_dir: Output directory for results
        extra_args: Additional arguments to pass to evaluate.py
    """
    print("\n" + "=" * 80)
    print(f"Evaluating model: {model_name}")
    print("=" * 80)
    print(f"Running unified test on all three domains:")
    for f in files:
        print(f"  - {f}")
    print("=" * 80)
    
    # Build command
    cmd = [
        sys.executable,
        "evaluate.py",
        "--model", model_name,
        "--prompts"
    ] + files
    
    # Add output directory if specified
    if output_dir:
        cmd.extend(["--output", output_dir])
    
    # Add any extra arguments
    cmd.extend(extra_args)
    
    print(f"\nCommand: {' '.join(cmd)}")
    print()
    
    # Run the evaluation
    result = subprocess.run(cmd)
    
    if result.returncode != 0:
        print(f"\n❌ Error evaluating {model_name}")
        return False
    else:
        print(f"\n✅ Completed unified evaluation for {model_name}")
        return True

def main():
    parser = argparse.ArgumentParser(
        description="Run unified evaluation on all three domains (health, misinformation, disinformation)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single model (unified test on all three domains)
  python run_all_domains.py --model "Qwen/Qwen2.5-1.5B-Instruct"
  
  # Multiple models separately (each gets unified test)
  python run_all_domains.py --models "Qwen/Qwen2.5-0.5B-Instruct" "Qwen/Qwen2.5-1.5B-Instruct"
  
  # With custom options
  python run_all_domains.py --models "Model1" "Model2" --temperature 0.7 --device mps
        """
    )
    
    # Model selection (either --model or --models)
    model_group = parser.add_mutually_exclusive_group()
    model_group.add_argument(
        "--model",
        type=str,
        help="Single HuggingFace model name or path to evaluate"
    )
    model_group.add_argument(
        "--models",
        type=str,
        nargs="+",
        help="Multiple HuggingFace model names to evaluate separately"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        help="Output directory for results (default: results)"
    )
    
    parser.add_argument(
        "--device",
        type=str,
        choices=["cuda", "mps", "cpu"],
        help="Device to use (overrides auto-detection)"
    )
    
    parser.add_argument(
        "--temperature",
        type=float,
        help="Sampling temperature"
    )
    
    parser.add_argument(
        "--max-tokens",
        type=int,
        help="Maximum tokens to generate"
    )
    
    parser.add_argument(
        "--judge-model",
        type=str,
        help="Optional separate model to use as judge for factual evaluation"
    )
    
    args, unknown = parser.parse_known_args()
    
    # Default to single model if neither specified
    if not args.model and not args.models:
        parser.print_help()
        sys.exit(1)
    
    # Determine which models to evaluate
    if args.models:
        models = args.models
    else:
        models = [args.model]
    
    # All three domain files
    files = [
        "prompts_disinformation.json",
        "prompts_health.json", 
        "prompts_misinformation.json"
    ]
    
    # Verify files exist
    for f in files:
        if not os.path.exists(f):
            print(f"❌ Error: Prompt file not found: {f}")
            sys.exit(1)
    
    # Build extra arguments list
    extra_args = []
    if args.device:
        extra_args.extend(["--device", args.device])
    if args.temperature is not None:
        extra_args.extend(["--temperature", str(args.temperature)])
    if args.max_tokens:
        extra_args.extend(["--max-tokens", str(args.max_tokens)])
    if args.judge_model:
        extra_args.extend(["--judge-model", args.judge_model])
    
    # Add any unknown arguments (passed through to evaluate.py)
    extra_args.extend(unknown)
    
    # Print summary
    print("\n" + "=" * 80)
    print("LSB Unified Multi-Domain Evaluation")
    print("=" * 80)
    print(f"Models to evaluate: {len(models)}")
    for i, model in enumerate(models, 1):
        print(f"  {i}. {model}")
    print(f"\nDomain files (unified test):")
    for f in files:
        print(f"  - {f}")
    if args.output:
        print(f"\nOutput directory: {args.output}")
    if extra_args:
        print(f"Additional arguments: {' '.join(extra_args)}")
    print("=" * 80)
    
    # Run evaluation for each model
    start_time = datetime.now()
    success_count = 0
    
    for model in models:
        if run_unified_evaluation(model, files, args.output, extra_args):
            success_count += 1
    
    end_time = datetime.now()
    duration = end_time - start_time
    
    # Final summary
    print("\n" + "=" * 80)
    print("EVALUATION SUMMARY")
    print("=" * 80)
    print(f"Total models evaluated: {len(models)}")
    print(f"Successful: {success_count}")
    print(f"Failed: {len(models) - success_count}")
    print(f"Total duration: {duration}")
    print("=" * 80)
    
    if success_count < len(models):
        sys.exit(1)

if __name__ == "__main__":
    main()

