from tqdm import tqdm
from data import DataFactory
from models import LLMFactory
from judges import JudgeFactory
from evaluation.schemas import EvaluationResultItem, EvaluationOutput

# --- Configuration ---
MODELS = ['gpt-5-nano']
JUDGE_CONFIG = {"type": "llm", "provider": "gpt-5-mini"}
DATA_CONFIG = {"type": "jsonl", "data_dir": "dummy_data"}
OUTPUT_FILE = "evaluation_results.json"
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

        for sample in tqdm(samples, desc=model_name):
            try:
                resp = model.generate(sample.prompt)
                score = judge.compare([resp], [sample.expected_response], prompts=[sample.prompt]).scores[0]
            except Exception as e:
                resp, score = f"ERROR: {e}", None

            results.append(EvaluationResultItem(
                model=model_name, sample_id=sample.id, score=score,
                prompt=sample.prompt, expected_response=sample.expected_response, model_response=resp
            ))

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write(EvaluationOutput(results=results).model_dump_json(indent=2))
    print(f"Saved results to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()