"""
LSB - Prompt Set Analysis Script
Authors: Kinga Frańczak, Kamil Kisiel, Wiktoria Koniecko, Piotr Kosakowski
Warsaw University of Technology, NLP Course Winter 2025

This script analyses the LSB prompt JSON files (health, misinformation,
disinformation) and produces basic descriptive statistics that can be
reported in the paper.

Example usage:
    python prompts_analysis.py
    python prompts_analysis.py --files prompts_health.json prompts_misinformation.json prompts_disinformation.json
    python prompts_analysis.py --output-dir results
"""

import argparse
import json
import os
from collections import Counter, defaultdict
from typing import Any, Dict, List, Tuple


def load_prompts(files: List[str]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Load all prompts and corresponding metadata from the given JSON files."""
    all_prompts: List[Dict[str, Any]] = []
    metadatas: List[Dict[str, Any]] = []

    for path in files:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        metadata = data.get("metadata", {})
        prompts = data.get("prompts", [])

        metadatas.append(metadata)
        for p in prompts:
            # Attach file-level metadata fields that may be useful
            p = dict(p)  # shallow copy
            p["_source_file"] = os.path.basename(path)
            p["_domain_from_metadata"] = metadata.get("domain")
            all_prompts.append(p)

    return all_prompts, metadatas


def prompt_text_length(prompt_field: Any) -> int:
    """Compute character length of the 'prompt' field, handling single vs multi-turn."""
    if isinstance(prompt_field, str):
        return len(prompt_field)
    if isinstance(prompt_field, list):
        # Multi-turn conversation: concatenate contents
        total = 0
        for turn in prompt_field:
            content = turn.get("content", "")
            total += len(content)
        return total
    return 0


def is_multi_turn(prompt_field: Any) -> bool:
    """Return True if prompt is a multi-turn conversation."""
    return isinstance(prompt_field, list)


def compute_basic_counts(prompts: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute core count-based statistics."""
    n_total = len(prompts)

    by_domain = Counter(p.get("domain", "unknown") for p in prompts)
    by_difficulty = Counter(p.get("difficulty_tier") for p in prompts)
    by_variant = Counter(p.get("variant", "unknown") for p in prompts)
    by_expected = Counter(p.get("expected_behavior", "unknown") for p in prompts)
    by_attack = Counter(p.get("attack_method", "unknown") for p in prompts)
    by_risk = Counter(p.get("risk_category", "unknown") for p in prompts)

    multi_turn = sum(1 for p in prompts if is_multi_turn(p.get("prompt")))

    return {
        "total_prompts": n_total,
        "by_domain": dict(by_domain),
        "by_difficulty_tier": dict(by_difficulty),
        "by_variant": dict(by_variant),
        "by_expected_behavior": dict(by_expected),
        "by_attack_method": dict(by_attack),
        "by_risk_category": dict(by_risk),
        "multi_turn_prompts": multi_turn,
        "single_turn_prompts": n_total - multi_turn,
    }


def compute_length_stats(prompts: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute basic statistics of prompt text length in characters."""
    lengths = [prompt_text_length(p.get("prompt")) for p in prompts]
    lengths = [l for l in lengths if l is not None]

    if not lengths:
        return {}

    lengths_sorted = sorted(lengths)

    def percentile(values: List[int], q: float) -> float:
        if not values:
            return 0.0
        idx = (len(values) - 1) * q
        lower = int(idx)
        upper = min(lower + 1, len(values) - 1)
        weight = idx - lower
        return values[lower] * (1 - weight) + values[upper] * weight

    avg = sum(lengths) / len(lengths)
    return {
        "length_chars_min": lengths_sorted[0],
        "length_chars_max": lengths_sorted[-1],
        "length_chars_mean": avg,
        "length_chars_median": percentile(lengths_sorted, 0.5),
        "length_chars_p25": percentile(lengths_sorted, 0.25),
        "length_chars_p75": percentile(lengths_sorted, 0.75),
    }


def compute_per_domain_stats(prompts: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute statistics broken down by domain."""
    by_domain: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for p in prompts:
        by_domain[p.get("domain", "unknown")].append(p)

    stats: Dict[str, Any] = {}
    for domain, domain_prompts in by_domain.items():
        stats[domain] = {
            "counts": compute_basic_counts(domain_prompts),
            "length_stats": compute_length_stats(domain_prompts),
        }
    return stats


def save_detailed_csv(prompts: List[Dict[str, Any]], output_path: str) -> None:
    """Save a per-prompt CSV with key fields for further analysis/plotting."""
    import csv

    fieldnames = [
        "id",
        "domain",
        "difficulty_tier",
        "variant",
        "expected_behavior",
        "risk_category",
        "attack_method",
        "data_type",
        "is_multi_turn",
        "prompt_length_chars",
        "_source_file",
    ]

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for p in prompts:
            row = {
                "id": p.get("id"),
                "domain": p.get("domain"),
                "difficulty_tier": p.get("difficulty_tier"),
                "variant": p.get("variant"),
                "expected_behavior": p.get("expected_behavior"),
                "risk_category": p.get("risk_category"),
                "attack_method": p.get("attack_method"),
                "data_type": p.get("data_type"),
                "is_multi_turn": is_multi_turn(p.get("prompt")),
                "prompt_length_chars": prompt_text_length(p.get("prompt")),
                "_source_file": p.get("_source_file"),
            }
            writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyse LSB prompt JSON files and compute base statistics."
    )
    parser.add_argument(
        "--files",
        nargs="+",
        default=[
            "prompts_health.json",
            "prompts_misinformation.json",
            "prompts_disinformation.json",
        ],
        help="List of prompt JSON files to analyse.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="Directory to save analysis outputs (JSON + CSV).",
    )

    args = parser.parse_args()

    prompts, metadatas = load_prompts(args.files)

    print(f"Loaded {len(prompts)} prompts from {len(args.files)} files.")
    for md in metadatas:
        name = md.get("domain", md.get("benchmark_name", "unknown"))
        total = md.get("total_prompts", "n/a")
        print(f"  - {name}: {total} prompts (from metadata)")

    overall_counts = compute_basic_counts(prompts)
    overall_lengths = compute_length_stats(prompts)
    per_domain_stats = compute_per_domain_stats(prompts)

    # Pretty-print key stats to console for immediate use in the paper
    print("\n=== OVERALL PROMPT SET STATISTICS ===")
    print(f"Total prompts: {overall_counts['total_prompts']}")
    print(f"Domains: {overall_counts['by_domain']}")
    print(f"Difficulty tiers: {overall_counts['by_difficulty_tier']}")
    print(f"Variants (safe/unsafe): {overall_counts['by_variant']}")
    print(f"Expected behaviors: {overall_counts['by_expected_behavior']}")
    print(f"Attack methods: {overall_counts['by_attack_method']}")
    print(f"Risk categories: {overall_counts['by_risk_category']}")
    print(
        f"Multi-turn vs single-turn: "
        f"{overall_counts['multi_turn_prompts']} multi-turn, "
        f"{overall_counts['single_turn_prompts']} single-turn"
    )

    if overall_lengths:
        print("\nPrompt length in characters (all domains combined):")
        for k, v in overall_lengths.items():
            print(f"  {k}: {v:.1f}")

    print("\n=== PER-DOMAIN SUMMARY (counts only) ===")
    for domain, stats in per_domain_stats.items():
        c = stats["counts"]
        print(f"\nDomain: {domain}")
        print(f"  Total: {c['total_prompts']}")
        print(f"  Difficulty tiers: {c['by_difficulty_tier']}")
        print(f"  Variants: {c['by_variant']}")
        print(f"  Expected behaviors: {c['by_expected_behavior']}")
        print(f"  Attack methods: {c['by_attack_method']}")
        print(f"  Risk categories: {c['by_risk_category']}")

    # Save machine-readable outputs
    os.makedirs(args.output_dir, exist_ok=True)
    summary_path = os.path.join(args.output_dir, "prompts_stats_summary.json")
    csv_path = os.path.join(args.output_dir, "prompts_detailed_stats.csv")

    summary_data = {
        "overall_counts": overall_counts,
        "overall_length_stats": overall_lengths,
        "per_domain_stats": per_domain_stats,
        "input_files": args.files,
    }

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2, ensure_ascii=False)

    save_detailed_csv(prompts, csv_path)

    print(f"\nSaved JSON summary to: {summary_path}")
    print(f"Saved per-prompt CSV to: {csv_path}")


if __name__ == "__main__":
    main()


