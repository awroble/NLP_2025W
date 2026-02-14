"""
Compare evaluation results with NLP_2025W baseline results.

Compares your model's performance against the baselines reported in the
LSB framework paper.

Usage:
    python compare_with_baseline.py --results results/mmada/results_*.json
    python compare_with_baseline.py --results results/llada/results_*.json --baseline-model qwen
"""

import argparse
import json
import pandas as pd
from pathlib import Path
from typing import Dict, List
import sys

# Baseline results from LSB Paper (Tables 6-10)
# "LSB: LLM Safety Benchmark" - Kisiel, Franczak, Koniecko, Kosakowski (2025)
# Values reported as Mean ± SD over 3 runs
# Data is scraped with LLM assistance from the paper tables and may need verification against the original source
BASELINE_RESULTS = {
    'llama-3.2-1b': {
        'name': 'Llama-3.2-1B-Instruct',
        'source': 'LSB Paper Tables 6-10',
        'overall': {
            'attack_success_rate': 0.438,  # 43.8% ± 0.5%
            'refusal_rate': 0.318,  # 31.8% ± 0.3%
        },
        'by_domain': {
            'health': {'asr': 0.465, 'refusal_rate': 0.318},  # Table 7
            'misinformation': {'asr': 0.392, 'refusal_rate': 0.305},  # Table 8
            'disinformation': {'asr': 0.458, 'refusal_rate': 0.325},  # Table 9
        },
        'by_tier': {  # Table 10
            1: {'asr': 0.345},  # 34.5% ± 0.4%
            2: {'asr': 0.460},  # 46.0% ± 0.5%
            3: {'asr': 0.498},  # 49.8% ± 0.6%
            4: {'asr': 0.450},  # 45.0% ± 0.7%
        }
    },
    'openmath-nemotron-1.5b': {
        'name': 'OpenMath-Nemotron-1.5B',
        'source': 'LSB Paper Tables 6-10',
        'overall': {
            'attack_success_rate': 0.509,  # 50.9% ± 0.4%
            'refusal_rate': 0.829,  # 82.9% ± 0.2% (high refusal = over-cautious)
        },
        'by_domain': {
            'health': {'asr': 0.565, 'refusal_rate': 0.829},  # Table 7
            'misinformation': {'asr': 0.645, 'refusal_rate': 0.830},  # Table 8
            'disinformation': {'asr': 0.318, 'refusal_rate': 0.825},  # Table 9
        },
        'by_tier': {  # Table 10 - inverse pattern (better on hard tiers)
            1: {'asr': 0.742},  # 74.2% ± 0.3%
            2: {'asr': 0.675},  # 67.5% ± 0.4%
            3: {'asr': 0.330},  # 33.0% ± 0.5%
            4: {'asr': 0.290},  # 29.0% ± 0.5%
        }
    },
    'tinyllama-1.1b': {
        'name': 'TinyLlama-1.1B-Chat-v1.0',
        'source': 'LSB Paper Tables 6-10',
        'overall': {
            'attack_success_rate': 0.681,  # 68.1% ± 0.8% (minimal safety training)
            'refusal_rate': 0.092,  # 9.2% ± 0.4% (almost never refuses)
        },
        'by_domain': {
            'health': {'asr': 0.638, 'refusal_rate': 0.092},  # Table 7
            'misinformation': {'asr': 0.675, 'refusal_rate': 0.105},  # Table 8
            'disinformation': {'asr': 0.730, 'refusal_rate': 0.065},  # Table 9
        },
        'by_tier': {  # Table 10 - vulnerability increases with sophistication
            1: {'asr': 0.520},  # 52.0% ± 0.5%
            2: {'asr': 0.615},  # 61.5% ± 0.6%
            3: {'asr': 0.760},  # 76.0% ± 0.8%
            4: {'asr': 0.830},  # 83.0% ± 0.9%
        }
    },
    'autoregressive-average': {
        'name': 'Autoregressive Average (Llama + TinyLlama)',
        'source': 'Computed from LSB Paper baselines',
        'overall': {
            'attack_success_rate': 0.5595,  # Average of Llama (43.8%) and TinyLlama (68.1%)
            'refusal_rate': 0.205,  # Average of Llama (31.8%) and TinyLlama (9.2%)
        },
        'by_domain': {
            'health': {'asr': 0.5515, 'refusal_rate': 0.205},
            'misinformation': {'asr': 0.5335, 'refusal_rate': 0.205},
            'disinformation': {'asr': 0.594, 'refusal_rate': 0.195},
        },
        'by_tier': {
            1: {'asr': 0.4325},  # Average across tiers
            2: {'asr': 0.5375},
            3: {'asr': 0.629},
            4: {'asr': 0.640},
        }
    }
}


def load_results(results_path: Path):
    """Load results from JSON file."""
    with open(results_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data.get('metadata', {}), data.get('results', [])


def calculate_metrics(results: List[Dict]) -> Dict:
    """Calculate metrics from results."""
    if not results:
        return {}

    total = len(results)

    # Overall
    overall = {
        'refusal_rate': sum(1 for r in results if r.get('is_refusal', False)) / total,
        'attack_success_rate': sum(1 for r in results if r.get('attack_success', False)) / total,
        'safety_failure_rate': sum(1 for r in results if r.get('safety_failure', False)) / total,
        'false_positive_rate': sum(1 for r in results if r.get('false_positive', False)) / total,
        'false_negative_rate': sum(1 for r in results if r.get('false_negative', False)) / total,
        'factual_error_rate': sum(1 for r in results if r.get('factual_error', False)) / total,
    }

    # By domain
    domains = {}
    for domain in set(r['domain'] for r in results):
        domain_results = [r for r in results if r['domain'] == domain]
        domain_total = len(domain_results)
        domains[domain] = {
            'asr': sum(1 for r in domain_results if r.get('attack_success', False)) / domain_total,
            'refusal_rate': sum(1 for r in domain_results if r.get('is_refusal', False)) / domain_total,
        }

    return {
        'overall': overall,
        'by_domain': domains
    }


def compare_with_baseline(your_metrics: Dict, baseline_name: str, baseline_metrics: Dict):
    """Compare your results with baseline."""
    print("=" * 80)
    print(f"COMPARISON WITH BASELINE: {baseline_name.upper()}")
    print("=" * 80)
    print()

    print(f"{'Metric':<35} {'Baseline':>12} {'Your Model':>12} {'Difference':>15}")
    print("-" * 80)

    # Overall metrics
    print("OVERALL METRICS:")
    for metric, baseline_val in baseline_metrics['overall'].items():
        your_val = your_metrics['overall'].get(metric, 0)
        diff = your_val - baseline_val
        diff_str = f"{diff:+.1%}" if abs(diff) >= 0.001 else "~same"

        # Add indicator
        if 'success' in metric or 'failure' in metric or 'error' in metric:
            # Lower is better
            indicator = " ✓ BETTER" if diff < -0.01 else (" ✗ WORSE" if diff > 0.01 else " ~SAME")
        elif 'refusal' in metric:
            # Context-dependent, just show difference
            indicator = ""
        else:
            indicator = ""

        metric_display = metric.replace('_', ' ').title()
        print(f"  {metric_display:<33} {baseline_val:>11.1%} {your_val:>11.1%} {diff_str:>15}{indicator}")

    print()
    print("BY DOMAIN:")

    # Domain comparison
    all_domains = set(baseline_metrics['by_domain'].keys()) | set(your_metrics['by_domain'].keys())
    for domain in sorted(all_domains):
        if domain not in baseline_metrics['by_domain'] or domain not in your_metrics['by_domain']:
            continue

        print(f"\n  {domain.upper()}:")
        baseline_domain = baseline_metrics['by_domain'][domain]
        your_domain = your_metrics['by_domain'][domain]

        for metric in ['asr', 'refusal_rate']:
            if metric not in baseline_domain:
                continue

            baseline_val = baseline_domain[metric]
            your_val = your_domain.get(metric, 0)
            diff = your_val - baseline_val
            diff_str = f"{diff:+.1%}" if abs(diff) >= 0.001 else "~same"

            if metric == 'asr':
                indicator = " ✓ BETTER" if diff < -0.01 else (" ✗ WORSE" if diff > 0.01 else " ~SAME")
            else:
                indicator = ""

            metric_display = metric.upper() if metric == 'asr' else metric.replace('_', ' ').title()
            print(f"    {metric_display:<29} {baseline_val:>11.1%} {your_val:>11.1%} {diff_str:>15}{indicator}")

    print()
    print("=" * 80)
    print()

    # Summary assessment
    asr_diff = your_metrics['overall']['attack_success_rate'] - baseline_metrics['overall']['attack_success_rate']

    print("SUMMARY:")
    if asr_diff < -0.05:
        print(f"  ✓ Your model shows SIGNIFICANTLY BETTER safety (ASR {asr_diff:.1%} lower)")
    elif asr_diff > 0.05:
        print(f"  ✗ Your model shows SIGNIFICANTLY WORSE safety (ASR {asr_diff:+.1%} higher)")
    else:
        print(f"  ~ Your model shows SIMILAR safety performance (ASR {asr_diff:+.1%})")

    print()


def print_baseline_list():
    """Print available baseline models."""
    print("Available baseline models:")
    for key in BASELINE_RESULTS.keys():
        print(f"  - {key}")
    print()
    print("Or use 'autoregressive-average' for average autoregressive model performance")


def main():
    parser = argparse.ArgumentParser(
        description="Compare evaluation results with NLP_2025W baselines",
        epilog="Note: Update BASELINE_RESULTS in the script with actual values from the paper"
    )
    parser.add_argument("--results", required=True, help="Path to your results JSON file")
    parser.add_argument("--baseline-model", default="autoregressive-average",
                        help="Baseline model to compare against")
    parser.add_argument("--list-baselines", action='store_true',
                        help="List available baseline models")

    args = parser.parse_args()

    if args.list_baselines:
        print_baseline_list()
        return 0

    results_path = Path(args.results)
    if not results_path.exists():
        print(f"Error: Results file not found: {results_path}")
        sys.exit(1)

    # Load your results
    metadata, results = load_results(results_path)
    your_metrics = calculate_metrics(results)

    model_name = metadata.get('model_evaluated', 'Your Model')
    print(f"Analyzing results for: {model_name}")
    print(f"Total prompts: {len(results)}")
    print()

    # Get baseline
    if args.baseline_model not in BASELINE_RESULTS:
        print(f"Error: Unknown baseline model '{args.baseline_model}'")
        print()
        print_baseline_list()
        sys.exit(1)

    baseline_metrics = BASELINE_RESULTS[args.baseline_model]

    # Compare
    compare_with_baseline(your_metrics, args.baseline_model, baseline_metrics)

    print()
    print("NOTE: Baseline values are placeholders. Update BASELINE_RESULTS")
    print("      in compare_with_baseline.py with actual values from the paper.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
