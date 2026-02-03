"""
Script to draw stratified validation samples from LLM evaluation results.
Used for human validation of LLM-as-a-judge accuracy.

Usage:
    python draw_validation_sample.py --annotator "Piotr" --n_samples 100
    python draw_validation_sample.py --annotator "Kinga" --n_samples 100

Each annotator gets a unique, non-overlapping stratified sample.
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import argparse


def load_results(results_path: Path) -> pd.DataFrame:
    """Load results JSON and convert to DataFrame."""
    with open(results_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    df = pd.DataFrame(data['results'])
    df['original_index'] = df.index
    return df, data['metadata']


def load_used_indices(used_indices_path: Path) -> set:
    """Load previously used indices to avoid overlap."""
    if not used_indices_path.exists():
        return set()

    with open(used_indices_path, 'r') as f:
        indices = set()
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                try:
                    indices.add(int(line))
                except ValueError:
                    continue
    return indices


def save_used_indices(used_indices_path: Path, new_indices: list, annotator: str):
    """Append newly used indices to the tracking file."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    with open(used_indices_path, 'a') as f:
        f.write(f"\n# Annotator: {annotator}, Date: {timestamp}, Count: {len(new_indices)}\n")
        for idx in sorted(new_indices):
            f.write(f"{idx}\n")


def stratified_sample(
    df: pd.DataFrame,
    n_samples: int,
    exclude_indices: set,
    random_state: int = None
) -> pd.DataFrame:
    """
    Draw stratified sample matching distribution of domain, tier, attack_method, and variant.

    Stratification priority:
    1. Domain (3 categories) - primary
    2. Difficulty tier (4 categories) - primary
    3. Variant (safe/unsafe) - secondary
    4. Attack method (6 categories) - tertiary
    """
    if random_state is not None:
        np.random.seed(random_state)

    # Filter out already used indices
    available_df = df[~df['original_index'].isin(exclude_indices)].copy()

    if len(available_df) < n_samples:
        raise ValueError(
            f"Not enough samples available. Requested: {n_samples}, "
            f"Available: {len(available_df)} (after excluding {len(exclude_indices)} used indices)"
        )

    print(f"Available samples after excluding used indices: {len(available_df)}")

    # Calculate target distribution based on full dataset proportions
    total = len(df)

    # Primary stratification: domain x tier (12 strata)
    strata_counts = df.groupby(['domain', 'difficulty_tier']).size()
    strata_proportions = strata_counts / total

    # Calculate target counts for each stratum
    target_counts = (strata_proportions * n_samples).round().astype(int)

    # Adjust to ensure we get exactly n_samples
    diff = n_samples - target_counts.sum()
    if diff != 0:
        # Add/remove from largest strata
        largest_strata = target_counts.nlargest(abs(diff)).index
        for stratum in largest_strata[:abs(diff)]:
            target_counts[stratum] += 1 if diff > 0 else -1

    print("\nTarget sample distribution (domain x tier):")
    print(target_counts.unstack(fill_value=0))
    print(f"Total target: {target_counts.sum()}")

    # Sample from each stratum
    sampled_indices = []

    for (domain, tier), target_n in target_counts.items():
        if target_n <= 0:
            continue

        stratum_df = available_df[
            (available_df['domain'] == domain) &
            (available_df['difficulty_tier'] == tier)
        ]

        if len(stratum_df) < target_n:
            print(f"Warning: Not enough samples in {domain}/tier{tier}. "
                  f"Requested {target_n}, available {len(stratum_df)}")
            target_n = len(stratum_df)

        if target_n > 0:
            # Within stratum, try to balance by variant and attack_method
            sampled = stratum_df.sample(n=target_n, random_state=random_state)
            sampled_indices.extend(sampled['original_index'].tolist())

    # Get the sampled rows
    sample_df = df[df['original_index'].isin(sampled_indices)].copy()

    return sample_df


def prepare_annotation_csv(sample_df: pd.DataFrame, metadata: dict) -> pd.DataFrame:
    """Prepare CSV with columns needed for human annotation."""

    # Select and rename columns for annotation
    annotation_df = sample_df[[
        'original_index',
        'prompt_id',
        'domain',
        'difficulty_tier',
        'variant',
        'risk_category',
        'attack_method',
        'expected_behavior',
        'model_response',
        'is_refusal',
        'attack_success',
        'evaluation_notes',
        'safety_failure',
        'false_positive',
        'false_negative',
        'factual_error'
    ]].copy()

    # Add columns for human annotation
    annotation_df['human_judgment'] = ''  # To be filled: agree/disagree
    annotation_df['human_notes'] = ''  # Optional notes
    annotation_df['correct_label'] = ''  # If disagree, what should it be?

    # Sort by domain, tier for easier annotation
    annotation_df = annotation_df.sort_values(['domain', 'difficulty_tier', 'prompt_id'])

    return annotation_df


def load_prompts(prompts_dir: Path) -> dict:
    """Load original prompts to include in annotation file."""
    prompts = {}
    for prompt_file in prompts_dir.glob('prompts_*.json'):
        with open(prompt_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            for prompt in data['prompts']:
                prompts[prompt['id']] = prompt.get('prompt', prompt.get('turns', ''))
    return prompts


def main():
    parser = argparse.ArgumentParser(description='Draw stratified validation sample')
    parser.add_argument('--annotator', type=str, required=True,
                        help='Name of the annotator (e.g., "Piotr")')
    parser.add_argument('--n_samples', type=int, default=100,
                        help='Number of samples to draw (default: 100)')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed for reproducibility')
    parser.add_argument('--results_file', type=str,
                        default='results/results_Llama-3.2-1B-Instruct_20251219_175613.json',
                        help='Path to results JSON file')

    args = parser.parse_args()

    # Setup paths
    base_dir = Path(__file__).parent
    results_path = base_dir / args.results_file
    validation_dir = base_dir / 'validation'
    validation_dir.mkdir(exist_ok=True)

    used_indices_path = validation_dir / 'used_sample_indices.txt'

    print(f"=" * 60)
    print(f"LLM-as-Judge Validation Sample Generator")
    print(f"=" * 60)
    print(f"Annotator: {args.annotator}")
    print(f"Requested samples: {args.n_samples}")
    print(f"Results file: {results_path}")
    print(f"Output directory: {validation_dir}")
    print()

    # Load results
    print("Loading results...")
    df, metadata = load_results(results_path)
    print(f"Total results loaded: {len(df)}")
    print(f"Model evaluated: {metadata['model_evaluated']}")
    print(f"Judge model: {metadata['judge_model']}")

    # Load previously used indices
    used_indices = load_used_indices(used_indices_path)
    print(f"Previously used indices: {len(used_indices)}")

    # Load original prompts
    print("\nLoading original prompts...")
    prompts = load_prompts(base_dir)
    print(f"Loaded {len(prompts)} prompts")

    # Draw stratified sample
    print("\nDrawing stratified sample...")
    sample_df = stratified_sample(
        df,
        args.n_samples,
        used_indices,
        random_state=args.seed
    )

    # Add original prompts to sample
    sample_df = sample_df.copy()
    sample_df['original_prompt'] = sample_df['prompt_id'].map(prompts)

    # Prepare annotation CSV
    annotation_df = prepare_annotation_csv(sample_df, metadata)

    # Add original prompt column at the beginning
    annotation_df.insert(2, 'original_prompt', sample_df.set_index('original_index').loc[annotation_df['original_index'], 'original_prompt'].values)

    # Print sample statistics
    print(f"\n{'=' * 60}")
    print("SAMPLE STATISTICS")
    print(f"{'=' * 60}")
    print(f"\nTotal samples drawn: {len(annotation_df)}")

    print("\nBy domain:")
    print(annotation_df['domain'].value_counts().sort_index())

    print("\nBy difficulty tier:")
    print(annotation_df['difficulty_tier'].value_counts().sort_index())

    print("\nBy attack method:")
    print(annotation_df['attack_method'].value_counts())

    print("\nBy variant:")
    print(annotation_df['variant'].value_counts())

    print("\nBy expected behavior:")
    print(annotation_df['expected_behavior'].value_counts())

    print("\nJudge decisions in sample:")
    print(f"  - attack_success=True: {annotation_df['attack_success'].sum()}")
    print(f"  - attack_success=False: {(~annotation_df['attack_success']).sum()}")
    print(f"  - is_refusal=True: {annotation_df['is_refusal'].sum()}")
    print(f"  - safety_failure=True: {annotation_df['safety_failure'].sum()}")

    # Save annotation CSV
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"validation_sample_{args.annotator}_{timestamp}.csv"
    output_path = validation_dir / output_filename

    annotation_df.to_csv(output_path, index=False, encoding='utf-8')
    print(f"\n{'=' * 60}")
    print(f"Annotation CSV saved to: {output_path}")

    # Save used indices
    new_indices = annotation_df['original_index'].tolist()
    save_used_indices(used_indices_path, new_indices, args.annotator)
    print(f"Used indices appended to: {used_indices_path}")

    # Save sample summary
    summary_path = validation_dir / f"validation_summary_{args.annotator}_{timestamp}.json"
    summary = {
        'annotator': args.annotator,
        'timestamp': timestamp,
        'n_samples': len(annotation_df),
        'random_seed': args.seed,
        'source_file': str(results_path),
        'model_evaluated': metadata['model_evaluated'],
        'judge_model': metadata['judge_model'],
        'sample_indices': new_indices,
        'distribution': {
            'by_domain': annotation_df['domain'].value_counts().to_dict(),
            'by_tier': annotation_df['difficulty_tier'].value_counts().to_dict(),
            'by_attack_method': annotation_df['attack_method'].value_counts().to_dict(),
            'by_variant': annotation_df['variant'].value_counts().to_dict(),
            'by_expected_behavior': annotation_df['expected_behavior'].value_counts().to_dict()
        }
    }

    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
    print(f"Sample summary saved to: {summary_path}")

    print(f"\n{'=' * 60}")
    print("ANNOTATION INSTRUCTIONS")
    print(f"{'=' * 60}")
    print("""
For each row in the CSV, review the following:
1. original_prompt - The prompt sent to the model
2. model_response - The model's response
3. Judge's decisions: is_refusal, attack_success, safety_failure, etc.

Fill in:
- human_judgment: 'agree' or 'disagree' with the judge's overall assessment
- human_notes: Any observations or reasoning
- correct_label: If you disagree, what should attack_success be? (true/false)

Focus on:
- Did the judge correctly identify refusals?
- Did the judge correctly assess attack success?
- Are there false positives (harmless marked as harmful)?
- Are there false negatives (harmful marked as safe)?
""")

    print(f"\nDone! Ready for annotation by {args.annotator}")


if __name__ == '__main__':
    main()
