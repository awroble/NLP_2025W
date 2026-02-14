"""
LLaDA Single Sample Test

Test the complete evaluation pipeline with just 1 prompt.

Usage:
    python test_single_sample.py
    python test_single_sample.py --domain health --steps 64 --no-judge
    python test_single_sample.py --prompt "What is AI?"
"""

import argparse
import sys
import json
from pathlib import Path
import torch
import transformers

# Check version
VERSION = transformers.__version__
REQUIRED_VERSION = "4.38.2"

if VERSION != REQUIRED_VERSION:
    print(f"\n❌ ERROR: LLaDA requires transformers=={REQUIRED_VERSION}")
    print(f"   Current version: {VERSION}")
    sys.exit(1)

from adapters import create_llada_evaluator

# Setup paths
DATA_DIR = Path(__file__).parent.parent.parent / "data"


def load_sample_prompt(domain="health"):
    """Load one sample prompt from dataset."""
    prompt_file = DATA_DIR / f"prompts_{domain}.json"

    if not prompt_file.exists():
        raise FileNotFoundError(f"Dataset not found: {prompt_file}")

    with open(prompt_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    if not data.get('prompts'):
        raise ValueError(f"No prompts found in {prompt_file}")

    return data['prompts'][0]


def main():
    parser = argparse.ArgumentParser(description="Test LLaDA pipeline with single sample")
    parser.add_argument("--domain", choices=["health", "misinformation", "disinformation"], default="health")
    parser.add_argument("--prompt", type=str, default=None, help="Custom prompt")
    parser.add_argument("--no-judge", action="store_true", help="Skip judge evaluation")
    parser.add_argument("--steps", type=int, default=64, help="Diffusion steps")

    args = parser.parse_args()

    print("="*70)
    print("LLaDA Single Sample Test")
    print("="*70)
    print(f"Transformers: {VERSION}")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA: {torch.cuda.is_available()}")
    print("="*70)

    try:
        if args.prompt:
            prompt_data = {
                "prompt": args.prompt,
                "category": "custom",
                "id": "test_001"
            }
        else:
            prompt_data = load_sample_prompt(args.domain)
            print(f"\nLoaded from {args.domain}:")
            print(f"  ID: {prompt_data.get('id', 'N/A')}")
            print(f"  Prompt: {prompt_data['prompt'][:100]}...")

        print("\n1. Initializing evaluator...")
        evaluator = create_llada_evaluator(
            device="cuda" if torch.cuda.is_available() else "cpu",
            steps=args.steps,
            gen_length=128,
            block_length=32,
            judge_model_name="Qwen/Qwen2.5-3B-Instruct" if not args.no_judge else None
        )
        print("   ✓ Initialized")

        print("\n2. Testing formatting...")
        formatted_prompt = evaluator.format_prompt(prompt_data)
        print(f"   Original: {prompt_data['prompt'][:80]}...")
        print(f"   Formatted: {formatted_prompt[:80]}...")
        print("   ✓ OK")

        print("\n3. Generating response...")
        response = evaluator.generate_response(formatted_prompt)
        print(f"\n   PROMPT: {prompt_data['prompt']}")
        print(f"\n   RESPONSE: {response}")
        print(f"\n   Length: {len(response)} chars")
        print("   ✓ OK")

        print("\n4. Testing refusal detection...")
        is_refusal = evaluator.detect_refusal(response)
        print(f"   Refusal: {is_refusal}")
        print(f"   → {'SAFE (refused)' if is_refusal else 'Check harmful content'}")

        print("\n5. Testing harmful content detection...")
        is_harmful = evaluator.detect_harmful_content(response, prompt_data.get('domain', 'health'))
        print(f"   Harmful: {is_harmful}")
        print(f"   → {'⚠️ WARNING' if is_harmful else '✓ OK'}")

        print("\n" + "="*70)
        print("Test: SUCCESS ✓")
        print("="*70)
        return 0

    except Exception as e:
        print(f"\n❌ Test failed: {type(e).__name__}: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
