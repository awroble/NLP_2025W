import json
import math
import os
from typing import List, Tuple

from tqdm import tqdm

from data.schemas import Sample
from evaluation.schemas import EvaluationResultItem, EvaluationOutput
from judges import JudgeFactory

# --- Configuration ---
MODEL_NAME = "llama"
JUDGE_CONFIG = {"type": "llm", "provider": "gpt-5-mini"}
DATA_DIR = "final_llama"
OUTPUT_FILE = "llama_evaluation.json"
BATCH_SIZE = 96
# ---------------------


def load_samples_from_jsonl(file_path: str) -> List[Tuple[Sample, str]]:
    """Loads samples and model responses from a .jsonl file."""
    samples_with_responses = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                data = json.loads(line)
                sample = Sample(
                    id=data["id"],
                    category=data["category"],
                    prompt=data["prompt"],
                    expected_response=data["expected_response"],
                )
                model_response = data.get("model_response")
                samples_with_responses.append((sample, model_response))
            except (json.JSONDecodeError, KeyError) as e:
                print(f"Skipping invalid line in {file_path}: {line.strip()} - Error: {e}")
    return samples_with_responses


def main():
    """
    Evaluates pre-existing model responses from JSONL files using a judge.
    """
    judge = JudgeFactory.create(JUDGE_CONFIG["type"], provider=JUDGE_CONFIG["provider"])
    results = []

    jsonl_files = [
        os.path.join(DATA_DIR, f) for f in os.listdir(DATA_DIR) if f.endswith(".jsonl")
    ]

    if not jsonl_files:
        print(f"No .jsonl files found in {DATA_DIR}")
        return

    all_samples_with_responses = []
    for file_path in jsonl_files:
        all_samples_with_responses.extend(load_samples_from_jsonl(file_path))

    print(f"Loaded {len(all_samples_with_responses)} samples from {len(jsonl_files)} files.")

    num_batches = math.ceil(len(all_samples_with_responses) / BATCH_SIZE)
    for i in tqdm(range(num_batches), desc=f"Judging {MODEL_NAME} responses"):
        start_idx = i * BATCH_SIZE
        end_idx = start_idx + BATCH_SIZE
        batch_data = all_samples_with_responses[start_idx:end_idx]

        if not batch_data:
            continue

        batch_samples = [item[0] for item in batch_data]
        batch_responses = [item[1] for item in batch_data]
        
        batch_prompts = [s.prompt for s in batch_samples]
        batch_expected = [s.expected_response for s in batch_samples]

        try:
            judge_result = judge.compare_batch(
                answers_a=batch_expected,
                answers_b=batch_responses,
                prompts=batch_prompts,
            )
            batch_scores = judge_result.scores
        except Exception as e:
            print(f"Error processing batch for {MODEL_NAME}: {e}")
            batch_scores = [None] * len(batch_samples)

        for j, sample in enumerate(batch_samples):
            model_response = batch_responses[j]
            results.append(
                EvaluationResultItem(
                    model=MODEL_NAME,
                    sample_id=sample.id,
                    score=batch_scores[j] if j < len(batch_scores) else None,
                    prompt=sample.prompt,
                    expected_response=sample.expected_response,
                    model_response=model_response
                    if model_response
                    else "ERROR: Response missing",
                )
            )

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write(EvaluationOutput(results=results).model_dump_json(indent=2))
    print(f"Saved {len(results)} results to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
