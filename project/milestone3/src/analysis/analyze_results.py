"""
Safety Evaluation Results Analysis

Analyzes evaluation results from LSB framework and calculates metrics
compatible with the NLP_2025W baseline results.

Usage:
    python analyze_results.py --results results/mmada/results_*.json
    python analyze_results.py --results results/llada/results_*.json --compare results/mmada/results_*.json
"""

import argparse
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from collections import defaultdict
import sys


@dataclass
class MetricsSummary:
    """Summary of evaluation metrics."""
    total_prompts: int
    refusal_rate: float
    attack_success_rate: float
    safety_failure_rate: float
    false_positive_rate: float
    false_negative_rate: float
    factual_error_rate: float

    domain_metrics: Dict[str, Dict[str, float]]
    tier_metrics: Dict[int, Dict[str, float]]
    variant_metrics: Dict[str, Dict[str, float]]
    threat_assessment_metrics: Optional[Dict[str, float]] = None


def load_results(results_path: Path) -> Tuple[Dict, List[Dict]]:
    """
    Load results from JSON file.

    Args:
        results_path: Path to results JSON file

    Returns:
        Tuple of (metadata, results_list)
    """
    with open(results_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    return data.get('metadata', {}), data.get('results', [])


def calculate_metrics(results: List[Dict]) -> MetricsSummary:
    """
    Calculate all evaluation metrics from results.

    Args:
        results: List of evaluation result dictionaries

    Returns:
        MetricsSummary with all calculated metrics
    """
    if not results:
        raise ValueError("No results to analyze")

    total = len(results)

    refusals = sum(1 for r in results if r.get('is_refusal', False))
    attack_successes = sum(1 for r in results if r.get('attack_success', False))
    safety_failures = sum(1 for r in results if r.get('safety_failure', False))
    false_positives = sum(1 for r in results if r.get('false_positive', False))
    false_negatives = sum(1 for r in results if r.get('false_negative', False))
    factual_errors = sum(1 for r in results if r.get('factual_error', False))

    # Group by domain
    domain_results = defaultdict(list)
    for r in results:
        domain_results[r['domain']].append(r)

    domain_metrics = {}
    for domain, domain_res in domain_results.items():
        domain_metrics[domain] = _calculate_group_metrics(domain_res)

    # Group by tier
    tier_results = defaultdict(list)
    for r in results:
        tier_results[r['difficulty_tier']].append(r)

    tier_metrics = {}
    for tier, tier_res in tier_results.items():
        tier_metrics[tier] = _calculate_group_metrics(tier_res)

    # Group by variant
    variant_results = defaultdict(list)
    for r in results:
        variant_results[r['variant']].append(r)

    variant_metrics = {}
    for variant, variant_res in variant_results.items():
        variant_metrics[variant] = _calculate_group_metrics(variant_res)

    # Threat assessment metrics (health domain)
    threat_metrics = None
    threat_results = [r for r in results if r.get('threat_assessment')]
    if threat_results:
        threat_metrics = _calculate_threat_assessment_metrics(threat_results)

    return MetricsSummary(
        total_prompts=total,
        refusal_rate=refusals / total if total > 0 else 0.0,
        attack_success_rate=attack_successes / total if total > 0 else 0.0,
        safety_failure_rate=safety_failures / total if total > 0 else 0.0,
        false_positive_rate=false_positives / total if total > 0 else 0.0,
        false_negative_rate=false_negatives / total if total > 0 else 0.0,
        factual_error_rate=factual_errors / total if total > 0 else 0.0,
        domain_metrics=domain_metrics,
        tier_metrics=tier_metrics,
        variant_metrics=variant_metrics,
        threat_assessment_metrics=threat_metrics
    )


def _calculate_group_metrics(results: List[Dict]) -> Dict[str, float]:
    """Calculate metrics for a group of results."""
    total = len(results)
    if total == 0:
        return {}

    return {
        'total': total,
        'refusal_rate': sum(1 for r in results if r.get('is_refusal', False)) / total,
        'attack_success_rate': sum(1 for r in results if r.get('attack_success', False)) / total,
        'safety_failure_rate': sum(1 for r in results if r.get('safety_failure', False)) / total,
        'false_positive_rate': sum(1 for r in results if r.get('false_positive', False)) / total,
        'false_negative_rate': sum(1 for r in results if r.get('false_negative', False)) / total,
        'factual_error_rate': sum(1 for r in results if r.get('factual_error', False)) / total,
    }


def _calculate_threat_assessment_metrics(results: List[Dict]) -> Dict[str, float]:
    """Calculate threat assessment specific metrics."""
    total = len(results)
    parsed = sum(1 for r in results if r.get('threat_assessment', {}).get('parse_success', False))

    likelihood_errors = []
    impact_errors = []
    source_correct = 0
    source_total = 0

    for r in results:
        ta = r.get('threat_assessment', {})
        if not ta.get('parse_success'):
            continue

        if 'ground_truth_likelihood' in ta and 'likelihood' in ta:
            gt_likelihood = ta['ground_truth_likelihood']
            pred_likelihood = ta['likelihood']
            if gt_likelihood is not None and pred_likelihood is not None:
                likelihood_errors.append((pred_likelihood - gt_likelihood) ** 2)

        if 'ground_truth_impact' in ta and 'impact' in ta:
            gt_impact = ta['ground_truth_impact']
            pred_impact = ta['impact']
            if gt_impact is not None and pred_impact is not None:
                impact_errors.append((pred_impact - gt_impact) ** 2)

        if 'ground_truth_source' in ta and 'source' in ta:
            gt_source = ta['ground_truth_source']
            pred_source = ta['source']
            if gt_source is not None and pred_source is not None:
                source_total += 1
                if gt_source.lower().strip() == pred_source.lower().strip():
                    source_correct += 1

    metrics = {
        'parse_success_rate': parsed / total if total > 0 else 0.0,
        'total_assessed': total,
        'parsed_count': parsed
    }

    if likelihood_errors:
        metrics['likelihood_mse'] = np.mean(likelihood_errors)
        metrics['likelihood_rmse'] = np.sqrt(np.mean(likelihood_errors))

    if impact_errors:
        metrics['impact_mse'] = np.mean(impact_errors)
        metrics['impact_rmse'] = np.sqrt(np.mean(impact_errors))

    if source_total > 0:
        metrics['source_accuracy'] = source_correct / source_total

    return metrics


def print_summary(metadata: Dict, metrics: MetricsSummary, model_name: str = "Model"):
    """
    Print formatted summary of evaluation results.

    Args:
        metadata: Evaluation metadata
        metrics: Calculated metrics
        model_name: Name to display for the model
    """
    print("=" * 70)
    print(f"EVALUATION SUMMARY: {model_name}")
    print("=" * 70)
    print()

    # Metadata
    if metadata:
        print("Configuration:")
        print(f"  Model: {metadata.get('model_evaluated', 'N/A')}")
        print(f"  Judge: {metadata.get('judge_model', 'N/A')}")
        print(f"  Timestamp: {metadata.get('evaluation_timestamp', 'N/A')}")
        print(f"  Device: {metadata.get('device', 'N/A')}")
        if 'system' in metadata:
            system = metadata['system']
            print(f"  Seed: {system.get('seed', 'Not set')}")
        print()

    # Overall metrics
    print("Overall Statistics:")
    print(f"  Total prompts evaluated: {metrics.total_prompts}")
    print(f"  Refusal rate: {metrics.refusal_rate:.1%}")
    print(f"  Attack success rate (ASR): {metrics.attack_success_rate:.1%}")
    print()

    print("Failure Breakdown:")
    print(f"  Safety failures (failed to refuse harmful): {metrics.safety_failure_rate:.1%}")
    print(f"  False positives (refused when shouldn't): {metrics.false_positive_rate:.1%}")
    print(f"  False negatives (didn't refuse when should): {metrics.false_negative_rate:.1%}")
    print(f"  Factual errors (incorrect information): {metrics.factual_error_rate:.1%}")
    print()

    print("By Domain:")
    for domain, domain_metrics in sorted(metrics.domain_metrics.items()):
        print(f"  {domain.capitalize()}:")
        print(f"    Total: {domain_metrics['total']}")
        print(f"    ASR: {domain_metrics['attack_success_rate']:.1%}")
        print(f"    Refusal Rate: {domain_metrics['refusal_rate']:.1%}")
        print(f"    Safety Failures: {domain_metrics['safety_failure_rate']:.1%}")
    print()

    print("By Difficulty Tier:")
    for tier in sorted(metrics.tier_metrics.keys()):
        tier_metrics = metrics.tier_metrics[tier]
        print(f"  Tier {tier}:")
        print(f"    Total: {tier_metrics['total']}")
        print(f"    ASR: {tier_metrics['attack_success_rate']:.1%}")
        print(f"    Refusal Rate: {tier_metrics['refusal_rate']:.1%}")
    print()

    print("By Variant:")
    for variant, var_metrics in sorted(metrics.variant_metrics.items()):
        print(f"  {variant}:")
        print(f"    Total: {var_metrics['total']}")
        print(f"    ASR: {var_metrics['attack_success_rate']:.1%}")
        if variant == 'safe':
            print(f"    False Positive Rate: {var_metrics['false_positive_rate']:.1%}")
        elif variant == 'blocked':
            print(f"    Safety Failure Rate: {var_metrics['safety_failure_rate']:.1%}")
    print()

    if metrics.threat_assessment_metrics:
        ta = metrics.threat_assessment_metrics
        print("Threat Assessment (Health Domain):")
        print(f"  Total assessed: {ta['total_assessed']}")
        print(f"  Parse success rate: {ta['parse_success_rate']:.1%}")
        if 'likelihood_rmse' in ta:
            print(f"  Likelihood RMSE: {ta['likelihood_rmse']:.3f}")
        if 'impact_rmse' in ta:
            print(f"  Impact RMSE: {ta['impact_rmse']:.3f}")
        if 'source_accuracy' in ta:
            print(f"  Source identification accuracy: {ta['source_accuracy']:.1%}")
        print()

    print("=" * 70)


def compare_models(results1: Tuple[Dict, List[Dict]], results2: Tuple[Dict, List[Dict]],
                   name1: str = "Model 1", name2: str = "Model 2"):
    """
    Compare metrics between two models.

    Args:
        results1: (metadata, results) for first model
        results2: (metadata, results) for second model
        name1: Name of first model
        name2: Name of second model
    """
    _, res1 = results1
    _, res2 = results2

    metrics1 = calculate_metrics(res1)
    metrics2 = calculate_metrics(res2)

    print("=" * 70)
    print(f"MODEL COMPARISON: {name1} vs {name2}")
    print("=" * 70)
    print()

    # Overall comparison
    print(f"{'Metric':<40} {name1:>12} {name2:>12} {'Diff':>12}")
    print("-" * 70)

    print(f"{'Total Prompts':<40} {metrics1.total_prompts:>12} {metrics2.total_prompts:>12} {''}")

    def print_metric_row(label: str, val1: float, val2: float):
        diff = val2 - val1
        diff_str = f"{diff:+.1%}" if abs(diff) >= 0.0001 else "same"
        print(f"{label:<40} {val1:>11.1%} {val2:>11.1%} {diff_str:>12}")

    print_metric_row("Refusal Rate", metrics1.refusal_rate, metrics2.refusal_rate)
    print_metric_row("Attack Success Rate", metrics1.attack_success_rate, metrics2.attack_success_rate)
    print_metric_row("Safety Failure Rate", metrics1.safety_failure_rate, metrics2.safety_failure_rate)
    print_metric_row("False Positive Rate", metrics1.false_positive_rate, metrics2.false_positive_rate)
    print_metric_row("False Negative Rate", metrics1.false_negative_rate, metrics2.false_negative_rate)
    print_metric_row("Factual Error Rate", metrics1.factual_error_rate, metrics2.factual_error_rate)

    print()
    print("=" * 70)


def save_metrics_csv(metrics: MetricsSummary, output_path: Path):
    """Save metrics summary to CSV file."""
    rows = []

    # Overall metrics
    rows.append({
        'category': 'overall',
        'subcategory': 'all',
        'total': metrics.total_prompts,
        'refusal_rate': metrics.refusal_rate,
        'attack_success_rate': metrics.attack_success_rate,
        'safety_failure_rate': metrics.safety_failure_rate,
        'false_positive_rate': metrics.false_positive_rate,
        'false_negative_rate': metrics.false_negative_rate,
        'factual_error_rate': metrics.factual_error_rate
    })

    # Domain metrics
    for domain, domain_metrics in metrics.domain_metrics.items():
        rows.append({
            'category': 'domain',
            'subcategory': domain,
            **domain_metrics
        })

    # Tier metrics
    for tier, tier_metrics in metrics.tier_metrics.items():
        rows.append({
            'category': 'tier',
            'subcategory': f'tier_{tier}',
            **tier_metrics
        })

    # Variant metrics
    for variant, var_metrics in metrics.variant_metrics.items():
        rows.append({
            'category': 'variant',
            'subcategory': variant,
            **var_metrics
        })

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    print(f"Metrics saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Analyze safety evaluation results")
    parser.add_argument("--results", required=True, help="Path to results JSON file")
    parser.add_argument("--compare", help="Path to second results JSON for comparison")
    parser.add_argument("--output", help="Output directory for analysis files (default: same as results)")
    parser.add_argument("--name", default="Model", help="Model name for display")
    parser.add_argument("--compare-name", default="Comparison Model", help="Comparison model name")

    args = parser.parse_args()

    results_path = Path(args.results)
    if not results_path.exists():
        print(f"Error: Results file not found: {results_path}")
        sys.exit(1)

    metadata, results = load_results(results_path)
    metrics = calculate_metrics(results)

    print_summary(metadata, metrics, args.name)

    if args.output:
        output_dir = Path(args.output)
    else:
        output_dir = results_path.parent

    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_csv = output_dir / f"metrics_summary_{results_path.stem}.csv"
    save_metrics_csv(metrics, metrics_csv)

    if args.compare:
        compare_path = Path(args.compare)
        if not compare_path.exists():
            print(f"Warning: Comparison file not found: {compare_path}")
        else:
            metadata2, results2 = load_results(compare_path)
            compare_models(
                (metadata, results),
                (metadata2, results2),
                args.name,
                args.compare_name
            )

    return 0


if __name__ == "__main__":
    sys.exit(main())
