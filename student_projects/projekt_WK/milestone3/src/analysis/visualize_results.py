"""
Visualization for Safety Evaluation Results

Creates comprehensive visualizations of evaluation metrics.

Usage:
    python visualize_results.py --results results/mmada/results_*.json
    python visualize_results.py --results results/llada/results_*.json results/mmada/results_*.json --compare
"""

import argparse
import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
import sys

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10


def load_results(results_path: Path) -> Tuple[Dict, pd.DataFrame]:
    """Load results and convert to DataFrame."""
    with open(results_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    metadata = data.get('metadata', {})
    results_df = pd.DataFrame(data.get('results', []))

    return metadata, results_df


def plot_overall_metrics(df: pd.DataFrame, model_name: str, output_dir: Path):
    """Plot overall safety metrics as bar chart."""
    metrics = {
        'Refusal Rate': df['is_refusal'].mean(),
        'Attack Success\nRate (ASR)': df['attack_success'].mean(),
        'Safety Failure\nRate': df['safety_failure'].mean(),
        'False Positive\nRate': df['false_positive'].mean(),
        'False Negative\nRate': df['false_negative'].mean(),
        'Factual Error\nRate': df['factual_error'].mean()
    }

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(metrics.keys(), [v * 100 for v in metrics.values()],
                   color=['#2ecc71', '#e74c3c', '#e74c3c', '#f39c12', '#f39c12', '#9b59b6'])

    ax.set_ylabel('Percentage (%)', fontsize=12)
    ax.set_title(f'Safety Metrics Overview - {model_name}', fontsize=14, fontweight='bold')
    ax.set_ylim(0, 100)

    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%',
                ha='center', va='bottom', fontweight='bold')

    plt.xticks(rotation=0, ha='center')
    plt.tight_layout()
    plt.savefig(output_dir / f'overall_metrics_{model_name.replace("/", "_")}.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: overall_metrics_{model_name.replace('/', '_')}.png")


def plot_domain_breakdown(df: pd.DataFrame, model_name: str, output_dir: Path):
    """Plot metrics by domain."""
    domains = df['domain'].unique()
    metrics = ['attack_success', 'is_refusal', 'safety_failure', 'false_positive']
    metric_labels = ['ASR', 'Refusal', 'Safety Failure', 'False Positive']

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for idx, (metric, label) in enumerate(zip(metrics, metric_labels)):
        ax = axes[idx]

        domain_values = []
        for domain in sorted(domains):
            domain_df = df[df['domain'] == domain]
            value = domain_df[metric].mean() * 100
            domain_values.append(value)

        bars = ax.bar(sorted(domains), domain_values, color=sns.color_palette("Set2", len(domains)))
        ax.set_ylabel('Percentage (%)', fontsize=11)
        ax.set_title(f'{label} by Domain', fontsize=12, fontweight='bold')
        ax.set_ylim(0, max(domain_values) * 1.2 if max(domain_values) > 0 else 10)

        # Add value labels
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.1f}%',
                        ha='center', va='bottom', fontsize=9)

        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

    plt.suptitle(f'Domain-Level Analysis - {model_name}', fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(output_dir / f'domain_breakdown_{model_name.replace("/", "_")}.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: domain_breakdown_{model_name.replace('/', '_')}.png")


def plot_tier_analysis(df: pd.DataFrame, model_name: str, output_dir: Path):
    """Plot ASR by difficulty tier."""
    tier_metrics = df.groupby('difficulty_tier').agg({
        'attack_success': 'mean',
        'is_refusal': 'mean',
        'safety_failure': 'mean'
    }).reset_index()

    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(tier_metrics))
    width = 0.25

    ax.bar(x - width, tier_metrics['attack_success'] * 100, width,
           label='Attack Success Rate', color='#e74c3c')
    ax.bar(x, tier_metrics['is_refusal'] * 100, width,
           label='Refusal Rate', color='#2ecc71')
    ax.bar(x + width, tier_metrics['safety_failure'] * 100, width,
           label='Safety Failure Rate', color='#f39c12')

    ax.set_xlabel('Difficulty Tier', fontsize=12)
    ax.set_ylabel('Percentage (%)', fontsize=12)
    ax.set_title(f'Metrics by Difficulty Tier - {model_name}', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f'Tier {int(t)}' for t in tier_metrics['difficulty_tier']])
    ax.legend(loc='best')
    ax.set_ylim(0, 100)

    plt.tight_layout()
    plt.savefig(output_dir / f'tier_analysis_{model_name.replace("/", "_")}.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: tier_analysis_{model_name.replace('/', '_')}.png")


def plot_variant_comparison(df: pd.DataFrame, model_name: str, output_dir: Path):
    """Plot metrics by prompt variant."""
    variants = df['variant'].unique()

    variant_data = []
    for variant in sorted(variants):
        variant_df = df[df['variant'] == variant]
        variant_data.append({
            'Variant': variant,
            'ASR': variant_df['attack_success'].mean() * 100,
            'Refusal Rate': variant_df['is_refusal'].mean() * 100,
            'Safety Failure': variant_df['safety_failure'].mean() * 100,
            'False Positive': variant_df['false_positive'].mean() * 100
        })

    variant_df = pd.DataFrame(variant_data)

    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(variant_df))
    width = 0.2

    ax.bar(x - 1.5*width, variant_df['ASR'], width, label='ASR', color='#e74c3c')
    ax.bar(x - 0.5*width, variant_df['Refusal Rate'], width, label='Refusal Rate', color='#2ecc71')
    ax.bar(x + 0.5*width, variant_df['Safety Failure'], width, label='Safety Failure', color='#f39c12')
    ax.bar(x + 1.5*width, variant_df['False Positive'], width, label='False Positive', color='#9b59b6')

    ax.set_xlabel('Prompt Variant', fontsize=12)
    ax.set_ylabel('Percentage (%)', fontsize=12)
    ax.set_title(f'Metrics by Prompt Variant - {model_name}', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(variant_df['Variant'])
    ax.legend(loc='best')
    ax.set_ylim(0, 100)

    plt.tight_layout()
    plt.savefig(output_dir / f'variant_comparison_{model_name.replace("/", "_")}.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: variant_comparison_{model_name.replace('/', '_')}.png")


def plot_model_comparison(results_list: List[Tuple[Dict, pd.DataFrame]], names: List[str], output_dir: Path):
    """Compare multiple models side by side."""
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()

    metrics = [
        ('is_refusal', 'Refusal Rate', '#2ecc71'),
        ('attack_success', 'Attack Success Rate', '#e74c3c'),
        ('safety_failure', 'Safety Failure Rate', '#f39c12'),
        ('false_positive', 'False Positive Rate', '#9b59b6'),
        ('false_negative', 'False Negative Rate', '#e67e22'),
        ('factual_error', 'Factual Error Rate', '#3498db')
    ]

    for idx, (metric, label, color) in enumerate(metrics):
        ax = axes[idx]

        values = []
        for metadata, df in results_list:
            value = df[metric].mean() * 100
            values.append(value)

        bars = ax.bar(names, values, color=color, alpha=0.8)
        ax.set_ylabel('Percentage (%)', fontsize=10)
        ax.set_title(label, fontsize=11, fontweight='bold')
        ax.set_ylim(0, max(values) * 1.2 if max(values) > 0 else 10)

        # Add value labels
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.1f}%',
                        ha='center', va='bottom', fontsize=9, fontweight='bold')

        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

    plt.suptitle('Model Comparison', fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(output_dir / 'model_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved: model_comparison.png")


def plot_response_length_distribution(df: pd.DataFrame, model_name: str, output_dir: Path):
    """Plot distribution of response lengths."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Overall distribution
    ax1.hist(df['response_length'], bins=50, color='#3498db', alpha=0.7, edgecolor='black')
    ax1.set_xlabel('Response Length (characters)', fontsize=11)
    ax1.set_ylabel('Frequency', fontsize=11)
    ax1.set_title('Overall Response Length Distribution', fontsize=12, fontweight='bold')
    ax1.axvline(df['response_length'].mean(), color='red', linestyle='--',
                linewidth=2, label=f'Mean: {df["response_length"].mean():.0f}')
    ax1.legend()

    # By refusal status
    refusal_lengths = df[df['is_refusal'] == True]['response_length']
    non_refusal_lengths = df[df['is_refusal'] == False]['response_length']

    ax2.hist([refusal_lengths, non_refusal_lengths], bins=30,
             color=['#e74c3c', '#2ecc71'], alpha=0.6, label=['Refusal', 'Non-refusal'])
    ax2.set_xlabel('Response Length (characters)', fontsize=11)
    ax2.set_ylabel('Frequency', fontsize=11)
    ax2.set_title('Response Length by Refusal Status', fontsize=12, fontweight='bold')
    ax2.legend()

    plt.suptitle(f'Response Length Analysis - {model_name}', fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(output_dir / f'response_length_{model_name.replace("/", "_")}.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: response_length_{model_name.replace('/', '_')}.png")


def main():
    parser = argparse.ArgumentParser(description="Visualize safety evaluation results")
    parser.add_argument("--results", nargs='+', required=True, help="Path(s) to results JSON file(s)")
    parser.add_argument("--output", help="Output directory for plots (default: same as first results file)")
    parser.add_argument("--names", nargs='+', help="Model names for display (default: from metadata)")
    parser.add_argument("--compare", action='store_true', help="Create comparison plots if multiple results provided")

    args = parser.parse_args()

    if args.output:
        output_dir = Path(args.output)
    else:
        output_dir = Path(args.results[0]).parent / "plots"

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving plots to: {output_dir}")
    print()

    # Load all results
    results_list = []
    names = args.names if args.names else []

    for idx, results_path in enumerate(args.results):
        path = Path(results_path)
        if not path.exists():
            print(f"Error: Results file not found: {path}")
            continue

        metadata, df = load_results(path)
        results_list.append((metadata, df))

        # Get model name
        if idx < len(names):
            model_name = names[idx]
        else:
            model_name = metadata.get('model_evaluated', f'Model_{idx+1}')
            names.append(model_name)

        # Create individual plots
        print(f"Creating plots for {model_name}...")
        plot_overall_metrics(df, model_name, output_dir)
        plot_domain_breakdown(df, model_name, output_dir)
        plot_tier_analysis(df, model_name, output_dir)
        plot_variant_comparison(df, model_name, output_dir)
        plot_response_length_distribution(df, model_name, output_dir)
        print()

    # Comparison plots
    if args.compare and len(results_list) > 1:
        print("Creating comparison plots...")
        plot_model_comparison(results_list, names, output_dir)
        print()

    print(f"All plots saved to: {output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
