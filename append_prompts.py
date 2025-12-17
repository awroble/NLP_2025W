#!/usr/bin/env python3
"""
Append prompts from a CSV file to an existing JSON benchmark file.
Updates metadata counts after appending.
"""

import argparse
import ast
import csv
import json
import re
from pathlib import Path


DOMAIN_PREFIX_MAP = {
    "health": "health",
    "misinformation": "misinfo",
    "disinformation": "disinfo",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Append prompts from CSV to JSON benchmark file and update metadata."
    )
    parser.add_argument(
        "--json",
        required=True,
        help="Path to the JSON benchmark file (e.g., prompts_health.json)",
    )
    parser.add_argument(
        "--csv",
        required=True,
        help="Path to the CSV file with new prompts (e.g., prompts_csv/health_prompts_batch_2.csv)",
    )
    return parser.parse_args()


def load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: str, data: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.write("\n")


def load_csv(path: str) -> list[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def get_max_sequence_per_tier(prompts: list[dict], prefix: str) -> dict[int, int]:
    """
    Extract the maximum sequence number for each tier from existing prompts.
    Returns a dict like {1: 12, 2: 15, 3: 13, 4: 10}
    """
    max_seq = {1: 0, 2: 0, 3: 0, 4: 0}
    pattern = re.compile(rf"^{re.escape(prefix)}_t(\d+)_(\d+)$")

    for prompt in prompts:
        prompt_id = prompt.get("id", "")
        match = pattern.match(prompt_id)
        if match:
            tier = int(match.group(1))
            seq = int(match.group(2))
            if tier in max_seq:
                max_seq[tier] = max(max_seq[tier], seq)

    return max_seq


def parse_prompt_field(prompt_value: str):
    """
    Parse the prompt field. If it looks like a stringified list (multi-turn conversation),
    convert it to an actual list. Otherwise, return as string.
    """
    stripped = prompt_value.strip()
    if stripped.startswith("[") and stripped.endswith("]"):
        try:
            parsed = ast.literal_eval(stripped)
            if isinstance(parsed, list):
                return parsed
        except (ValueError, SyntaxError):
            pass
    return prompt_value


def build_prompt_object(
    row: dict,
    prompt_id: str,
    domain: str,
) -> dict:
    """
    Build a prompt object from a CSV row.
    """
    prompt_obj = {
        "id": prompt_id,
        "domain": domain,
        "difficulty_tier": int(row["difficulty_tier"]),
        "variant": row["variant"],
        "data_type": "text",
        "prompt": parse_prompt_field(row["prompt"]),
        "expected_behavior": row["expected_behavior"],
        "risk_category": row["risk_category"],
        "attack_method": row["attack_method"],
        "ground_truth": row["ground_truth"],
        "evaluation_criteria": {
            "accuracy": row["accuracy"],
            "harm_potential": row["harm_potential"],
        },
    }
    return prompt_obj


def update_metadata(data: dict) -> None:
    """
    Recalculate and update metadata counts based on current prompts.
    """
    prompts = data["prompts"]
    metadata = data["metadata"]

    metadata["total_prompts"] = len(prompts)

    tier_counts = {1: 0, 2: 0, 3: 0, 4: 0}
    variant_counts = {"safe": 0, "unsafe": 0}

    for prompt in prompts:
        tier = prompt.get("difficulty_tier")
        if tier in tier_counts:
            tier_counts[tier] += 1

        variant = prompt.get("variant")
        if variant in variant_counts:
            variant_counts[variant] += 1

    metadata["difficulty_distribution"] = {
        f"tier_{k}": v for k, v in tier_counts.items()
    }
    metadata["variant_distribution"] = variant_counts


def main():
    args = parse_args()

    json_path = Path(args.json)
    csv_path = Path(args.csv)

    if not json_path.exists():
        print(f"Error: JSON file not found: {json_path}")
        return 1

    if not csv_path.exists():
        print(f"Error: CSV file not found: {csv_path}")
        return 1

    # Load data
    data = load_json(json_path)
    csv_rows = load_csv(csv_path)

    if not csv_rows:
        print("Warning: CSV file is empty, no prompts to append.")
        return 0

    # Get domain and prefix
    domain = data["metadata"]["domain"]
    prefix = DOMAIN_PREFIX_MAP.get(domain, domain)

    # Get current max sequence numbers per tier
    max_seq = get_max_sequence_per_tier(data["prompts"], prefix)

    print(f"Domain: {domain} (prefix: {prefix})")
    print(f"Existing prompts: {len(data['prompts'])}")
    print(f"Current max sequences per tier: {max_seq}")
    print(f"CSV rows to append: {len(csv_rows)}")

    # Append new prompts
    added_count = 0
    for row in csv_rows:
        tier = int(row["difficulty_tier"])
        max_seq[tier] += 1
        seq_num = max_seq[tier]

        prompt_id = f"{prefix}_t{tier}_{seq_num:03d}"
        prompt_obj = build_prompt_object(row, prompt_id, domain)

        data["prompts"].append(prompt_obj)
        added_count += 1

    print(f"Added {added_count} new prompts")

    # Update metadata
    update_metadata(data)

    print(f"New total prompts: {data['metadata']['total_prompts']}")
    print(f"New difficulty distribution: {data['metadata']['difficulty_distribution']}")
    print(f"New variant distribution: {data['metadata']['variant_distribution']}")

    # Save updated JSON
    save_json(json_path, data)
    print(f"Saved updated JSON to: {json_path}")

    return 0


if __name__ == "__main__":
    exit(main())
