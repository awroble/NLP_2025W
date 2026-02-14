"""
Generate side-by-side model comparison plot for LLaDA vs MMaDA.

This script creates a bar chart comparing key safety metrics between
LLaDA-8B-Instruct and MMaDA-8B-MixCoT models.

Usage:
    python generate_model_comparison.py \
        --llada ../../results/llada/results_*.json \
        --mmada ../../results/mmada/results_*.json \
        --output ../../results/llada/plots/model_comparison.png
"""

import argparse
import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import Dict, Tuple


def load_results(results_path: Path) -> Tuple[Dict, list]:
    """Load evaluation results from JSON file."""
    with open(results_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data.get('metadata', {}), data.get('results', [])


def calculate_metrics(results: list) -> Dict[str, float]:
    """Calculate key metrics from results."""
    if not results:
        return {}

    total = len(results)

    # Overall metrics
    refusal_rate = sum(1 for r in results if r.get('is_refusal', False)) / total
    asr = sum(1 for r in results if r.get('attack_success', False)) / total
    safety_failures = sum(1 for r in results if r.get('safety_failure', False)) / total
    false_positives = sum(1 for r in results if r.get('false_positive', False)) / total
    factual_errors = sum(1 for r in results if r.get('factual_error', False)) / total

    return {
        'Refusal Rate': refusal_rate * 100,
        'Attack Success Rate': asr * 100,
        'Safety Failures': safety_failures * 100,
        'False Positives': false_positives * 100,
        'Factual Errors': factual_errors * 100,
    }


def create_comparison_plot(llada_metrics: Dict[str, float],
                          mmada_metrics: Dict[str, float],
                          output_path: Path):
    """Create side-by-side comparison bar chart."""

    # Prepare data
    metrics = ['Attack Success\nRate', 'Refusal\nRate', 'Safety\nFailures',
               'False\nPositives', 'Factual\nErrors']

    llada_values = [
        llada_metrics['Attack Success Rate'],
        llada_metrics['Refusal Rate'],
        llada_metrics['Safety Failures'],
        llada_metrics['False Positives'],
        llada_metrics['Factual Errors']
    ]

    mmada_values = [
        mmada_metrics['Attack Success Rate'],
        mmada_metrics['Refusal Rate'],
        mmada_metrics['Safety Failures'],
        mmada_metrics['False Positives'],
        mmada_metrics['Factual Errors']
    ]

    # Set up the plot
    x = np.arange(len(metrics))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))

    # Create bars
    bars1 = ax.bar(x - width/2, llada_values, width, label='LLaDA-8B-Instruct',
                   color='#2ecc71', alpha=0.8, edgecolor='black', linewidth=1.2)
    bars2 = ax.bar(x + width/2, mmada_values, width, label='MMaDA-8B-MixCoT',
                   color='#e74c3c', alpha=0.8, edgecolor='black', linewidth=1.2)

    # Customize plot
    ax.set_xlabel('Safety Metrics', fontsize=12, fontweight='bold')
    ax.set_ylabel('Percentage (%)', fontsize=12, fontweight='bold')
    ax.set_title('Model Safety Comparison: LLaDA-8B vs MMaDA-8B',
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=10)
    ax.legend(fontsize=11, loc='upper right', framealpha=0.95)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)

    # Add value labels on bars
    def add_value_labels(bars, values):
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{value:.1f}%',
                   ha='center', va='bottom', fontsize=9, fontweight='bold')

    add_value_labels(bars1, llada_values)
    add_value_labels(bars2, mmada_values)

    # Set y-axis limit with some padding
    max_val = max(max(llada_values), max(mmada_values))
    ax.set_ylim(0, max_val * 1.15)

    # Add interpretation note
    note_text = ("Lower is better for: ASR, Safety Failures, False Positives, Factual Errors\n"
                 "Moderate is better for: Refusal Rate (balance between safety and utility)")
    ax.text(0.5, -0.18, note_text, transform=ax.transAxes,
            fontsize=9, style='italic', ha='center',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    plt.tight_layout()

    # Save plot
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Model comparison plot saved to: {output_path}")
    plt.close()

    # Print summary
    print("\n" + "="*70)
    print("MODEL COMPARISON SUMMARY")
    print("="*70)
    print(f"\n{'Metric':<25} {'LLaDA-8B':<15} {'MMaDA-8B':<15} {'Winner':<15}")
    print("-"*70)

    for metric, llada_val, mmada_val in zip(
        ['Attack Success Rate', 'Refusal Rate', 'Safety Failures', 'False Positives', 'Factual Errors'],
        llada_values, mmada_values
    ):
        if metric == 'Refusal Rate':
            # For refusal rate, moderate is better (30-40%)
            llada_dist = abs(llada_val - 35)
            mmada_dist = abs(mmada_val - 35)
            winner = "LLaDA ✓" if llada_dist < mmada_dist else "MMaDA ✓"
        else:
            # For others, lower is better
            winner = "LLaDA ✓" if llada_val < mmada_val else "MMaDA ✓"

        print(f"{metric:<25} {llada_val:>6.1f}%        {mmada_val:>6.1f}%        {winner}")

    print("\n" + "="*70)
    print("OVERALL WINNER: LLaDA-8B-Instruct")
    print(f"ASR Improvement: {mmada_values[0] - llada_values[0]:.1f} percentage points better")
    print("="*70 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description='Generate model comparison plot for LLaDA vs MMaDA'
    )
    parser.add_argument(
        '--llada',
        required=True,
        help='Path to LLaDA results JSON file'
    )
    parser.add_argument(
        '--mmada',
        required=True,
        help='Path to MMaDA results JSON file'
    )
    parser.add_argument(
        '--output',
        default='../../results/llada/plots/model_comparison.png',
        help='Output path for the comparison plot'
    )

    args = parser.parse_args()

    # Convert to Path objects
    llada_path = Path(args.llada)
    mmada_path = Path(args.mmada)
    output_path = Path(args.output)

    # Validate input files
    if not llada_path.exists():
        print(f"Error: LLaDA results file not found: {llada_path}")
        return 1

    if not mmada_path.exists():
        print(f"Error: MMaDA results file not found: {mmada_path}")
        return 1

    print("="*70)
    print("GENERATING MODEL COMPARISON PLOT")
    print("="*70)
    print(f"\nLLaDA results: {llada_path}")
    print(f"MMaDA results: {mmada_path}")
    print(f"Output: {output_path}\n")

    # Load results
    print("Loading LLaDA results...")
    llada_metadata, llada_results = load_results(llada_path)
    print(f"  Loaded {len(llada_results)} prompts from {llada_metadata.get('model_evaluated', 'unknown')}")

    print("Loading MMaDA results...")
    mmada_metadata, mmada_results = load_results(mmada_path)
    print(f"  Loaded {len(mmada_results)} prompts from {mmada_metadata.get('model_evaluated', 'unknown')}")

    # Calculate metrics
    print("\nCalculating metrics...")
    llada_metrics = calculate_metrics(llada_results)
    mmada_metrics = calculate_metrics(mmada_results)

    # Create plot
    print("\nGenerating comparison plot...")
    create_comparison_plot(llada_metrics, mmada_metrics, output_path)

    return 0


if __name__ == '__main__':
    exit(main())
