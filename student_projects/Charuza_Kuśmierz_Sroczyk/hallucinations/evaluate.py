from tqdm import tqdm
from data import DataFactory
from models import LLMFactory
from judges import JudgeFactory
from evaluation.schemas import EvaluationResultItem, EvaluationOutput
import math

# --- Configuration ---
MODELS = ['gpt-5-mini']
JUDGE_CONFIG = {"type": "llm", "provider": "gpt-5-mini"}
DATA_CONFIG = {"type": "jsonl", "data_dir": "wikipedia_data2"}
# DATA_CONFIG = {"type": "jsonl", "data_dir": "dummy_data"}
OUTPUT_FILE = "gpt-5-mini-wikipedia_dataset2_evaluation.json"
# OUTPUT_FILE = "dummy_evaluation.json"
BATCH_SIZE = 96
# ---------------------

def main():
    samples = DataFactory.create(DATA_CONFIG["type"], data_dir=DATA_CONFIG["data_dir"]).load()
    judge = JudgeFactory.create(JUDGE_CONFIG["type"], provider=JUDGE_CONFIG["provider"])
    results = []

    for model_name in MODELS:
        try:
            model = LLMFactory.create(model_name)
        except Exception as e:
            print(f"Skipping {model_name}: {e}")
            continue

        num_batches = math.ceil(len(samples) / BATCH_SIZE)
        for i in tqdm(range(num_batches), desc=f"Evaluating {model_name}"):
            start_idx = i * BATCH_SIZE
            end_idx = start_idx + BATCH_SIZE
            batch_samples = samples[start_idx:end_idx]

            if not batch_samples:
                continue

            batch_prompts = [s.prompt for s in batch_samples]
            batch_expected = [s.expected_response for s in batch_samples]

            try:
                batch_responses = model.generate_batch(batch_prompts)
                judge_result = judge.compare_batch(
                    answers_a=batch_expected,
                    answers_b=batch_responses,
                    prompts=batch_prompts
                )
                batch_scores = judge_result.scores
            except Exception as e:
                print(f"Error processing batch for {model_name}: {e}")
                batch_responses = [f"ERROR: {e}"] * len(batch_samples)
                batch_scores = [None] * len(batch_samples)

            for j, sample in enumerate(batch_samples):
                results.append(EvaluationResultItem(
                    model=model_name,
                    sample_id=sample.id,
                    score=batch_scores[j] if j < len(batch_scores) else None,
                    prompt=sample.prompt,
                    expected_response=sample.expected_response,
                    model_response=batch_responses[j] if j < len(batch_responses) else "ERROR: Response missing"
                ))

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write(EvaluationOutput(results=results).model_dump_json(indent=2))
    print(f"Saved results to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()