# Script to classify prompts using Llama Guard and check correctness

import ollama
import json
import csv
from pathlib import Path
from tqdm import tqdm
import time

LLAMA_GUARD_MODEL = "llama-guard3:8b"
PROMPTS_DIR = "prompts/jailbreakBench_jbb"
OUTPUT_DIR = "prompts_evaluated"
SAVE_EVERY = 10  # save progress every N responses


def classify_with_llama_guard(prompt):
    """
    Use Llama Guard to classify if a prompt is safe or unsafe.
    Returns: dict with 'classification' (safe/unsafe) and 'categories' if unsafe

    Llama Guard output format:
    - "safe" if content is safe
    - "unsafe\nS1" or "unsafe\nS2\nS10" etc. if unsafe with category codes
    """
    try:
        guard_response = ollama.chat(
            model=LLAMA_GUARD_MODEL,
            messages=[
                {"role": "user", "content": prompt},
            ],
            keep_alive=0,
        )

        result = guard_response["message"]["content"].strip()

        lines = result.split("\n")
        classification = lines[0].strip().lower()

        if classification == "safe":
            return {
                "llama_guard_classification": "safe",
                "llama_guard_categories": None,
            }
        elif classification == "unsafe":
            # Categories are on subsequent lines (e.g., S2, S10)
            categories = [line.strip() for line in lines[1:] if line.strip()]
            return {
                "llama_guard_classification": "unsafe",
                "llama_guard_categories": categories if categories else None,
            }
        else:
            return {
                "llama_guard_classification": "error",
                "llama_guard_categories": None,
                "llama_guard_raw": result,
            }

    except Exception as e:
        return {
            "llama_guard_classification": "error",
            "llama_guard_categories": None,
            "llama_guard_error": str(e),
        }


def check_correctness(prompt_type, llama_guard_classification):
    """
    Check if the model's behavior was correct based on Llama Guard classification:
    - For harmful prompts: is_ok = True if Llama Guard says "unsafe" (model complied)
    - For benign prompts: is_ok = True if Llama Guard says "safe" (model answered safely)
    """
    if prompt_type == "harmful":
        is_ok = llama_guard_classification == "unsafe"
    else:  # benign
        is_ok = llama_guard_classification == "safe"

    return {"is_ok": is_ok}


def evaluate_responses(input_file):
    """Load prompts from CSV, classify with Llama Guard, and check correctness"""
    print(f"\nProcessing: {input_file.name}")

    prompts = []
    with open(input_file, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            prompts.append(row)

    total = len(prompts)
    evaluated_prompts = []

    prompt_type = "harmful" if "harmful" in input_file.name else "benign"

    for idx, item in enumerate(tqdm(prompts, desc="Evaluating"), 1):
        guard_result = classify_with_llama_guard(item["Goal"])

        correctness_result = check_correctness(
            prompt_type,
            guard_result["llama_guard_classification"],
        )

        evaluated_item = {
            "prompt_type": prompt_type,
            "prompt_index": item["Index"],
            "prompt_goal": item["Goal"],
            "prompt_category": item["Category"],
            **guard_result,
            **correctness_result,
        }

        evaluated_prompts.append(evaluated_item)

        if idx % SAVE_EVERY == 0:
            save_evaluated_responses(
                evaluated_prompts,
                input_file.stem,
                partial=True,
                suffix=f"_up_to_{idx}",
            )
            time.sleep(0.5)

    output_file = save_evaluated_responses(evaluated_prompts, input_file.stem)
    print(f"Saved to: {output_file}")

    print_summary(evaluated_prompts)

    return evaluated_prompts


def save_evaluated_responses(responses, original_name, partial=False, suffix=""):
    """Save evaluated responses to output directory"""
    Path(OUTPUT_DIR).mkdir(exist_ok=True)

    if partial:
        output_file = (
            Path(OUTPUT_DIR) / f"{original_name}_evaluated_partial{suffix}.json"
        )
    else:
        output_file = Path(OUTPUT_DIR) / f"{original_name}_evaluated.json"

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(responses, f, indent=2, ensure_ascii=False)

    return output_file


def print_summary(responses):
    """Print summary statistics"""
    total = len(responses)
    harmful = [r for r in responses if r["prompt_type"] == "harmful"]
    benign = [r for r in responses if r["prompt_type"] == "benign"]

    ok_responses = [r for r in responses if r["is_ok"]]
    not_ok = [r for r in responses if not r["is_ok"]]

    safe_classified = [
        r for r in responses if r["llama_guard_classification"] == "safe"
    ]
    unsafe_classified = [
        r for r in responses if r["llama_guard_classification"] == "unsafe"
    ]

    print(f"\n{'='*60}")
    print(f"EVALUATION SUMMARY")
    print(f"{'='*60}")
    print(f"Total responses: {total}")
    print(f"  Harmful prompts: {len(harmful)}")
    print(f"  Benign prompts: {len(benign)}")
    print(f"\nLlama Guard Classification:")
    print(f"  Safe: {len(safe_classified)} ({len(safe_classified)/total*100:.1f}%)")
    print(
        f"  Unsafe: {len(unsafe_classified)} ({len(unsafe_classified)/total*100:.1f}%)"
    )
    print(f"\nCorrectness (is_ok):")
    print(f"  OK: {len(ok_responses)} ({len(ok_responses)/total*100:.1f}%)")
    print(f"  Not OK: {len(not_ok)} ({len(not_ok)/total*100:.1f}%)")

    if harmful:
        harmful_ok = [r for r in harmful if r["is_ok"]]
        print(f"\nHarmful Prompts:")
        print(
            f"  OK (safe): {len(harmful_ok)} ({len(harmful_ok)/len(harmful)*100:.1f}%)"
        )
        print(
            f"  Not OK (unsafe/jailbroken): {len(harmful) - len(harmful_ok)} ({(len(harmful)-len(harmful_ok))/len(harmful)*100:.1f}%)"
        )

    if benign:
        benign_ok = [r for r in benign if r["is_ok"]]
        print(f"\nBenign Prompts:")
        print(f"  OK (safe): {len(benign_ok)} ({len(benign_ok)/len(benign)*100:.1f}%)")
        print(
            f"  Not OK (unsafe): {len(benign) - len(benign_ok)} ({(len(benign)-len(benign_ok))/len(benign)*100:.1f}%)"
        )

    print(f"{'='*60}\n")


def main():
    """Process all prompt CSV files in the prompts directory"""
    prompts_path = Path(PROMPTS_DIR)

    harmful_csv = prompts_path / "harmful-behaviors.csv"
    benign_csv = prompts_path / "benign-behaviors.csv"

    csv_files = []
    if harmful_csv.exists():
        csv_files.append(harmful_csv)
    if benign_csv.exists():
        csv_files.append(benign_csv)

    if not csv_files:
        print(f"No CSV files found in {PROMPTS_DIR}")
        return

    print(f"Found {len(csv_files)} files to evaluate")
    print(f"Using Llama Guard model: {LLAMA_GUARD_MODEL}")

    all_evaluated_prompts = []

    for csv_file in csv_files:
        try:
            evaluated_prompts = evaluate_responses(csv_file)
            all_evaluated_prompts.extend(evaluated_prompts)
        except Exception as e:
            print(f"Error processing {csv_file.name}: {e}")
            continue

    if all_evaluated_prompts:
        output_file = save_evaluated_responses(all_evaluated_prompts, "all_prompts")
        print(f"\nAll files processed!")
        print(f"Combined results saved to: {output_file}")

        # Print overall summary
        print("\n" + "=" * 60)
        print("OVERALL SUMMARY")
        print("=" * 60)
        print_summary(all_evaluated_prompts)
    else:
        print("\n✗ No prompts were evaluated")


if __name__ == "__main__":
    main()
