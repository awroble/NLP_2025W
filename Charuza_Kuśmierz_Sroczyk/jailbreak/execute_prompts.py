import ollama
import json
import csv
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
import time

# Configuration
MODELS = ["qwen3:8b", "llama3.1:8b", "mistral:latest"]
PROMPTS_DIR = "prompts/jailbreakBench_jbb"
OUTPUT_DIR = "model_responses"
SEED = 123  # to make results reproducible
SAVE_EVERY = 10  # save progress every N prompts


def safe_name(name: str) -> str:
    """sanitize model name for filenames (windows-safe)."""
    return name.replace(":", "_").replace("/", "_")


def load_prompts_from_csv(file_path):
    """load prompts from a csv file"""
    prompts = []
    with open(file_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            prompts.append(
                {
                    "index": row["Index"],
                    "goal": row["Goal"],
                    "category": row["Category"],
                }
            )
    return prompts


def get_model_response(model_name, prompt):
    """get response from a specific model"""
    try:
        response = ollama.chat(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            keep_alive=0,
            options={"seed": SEED},
        )
        return response["message"]["content"]
    except Exception as e:
        return f"Error: {str(e)}"


def save_responses(responses, model_name, partial=False, suffix=""):
    """save responses to file (partial indicates if it's a checkpoint save)"""
    Path(OUTPUT_DIR).mkdir(exist_ok=True)
    fname_model = safe_name(model_name)
    if partial:
        output_file = Path(OUTPUT_DIR) / f"responses_{fname_model}_partial{suffix}.json"
    else:
        output_file = (
            Path(OUTPUT_DIR)
            / f"responses_{fname_model}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(responses, f, indent=2, ensure_ascii=False)
    return output_file


def test_models():
    """test all models with prompts and save responses"""
    Path(OUTPUT_DIR).mkdir(exist_ok=True)

    harmful_csv = Path(PROMPTS_DIR) / "harmful-behaviors.csv"
    benign_csv = Path(PROMPTS_DIR) / "benign-behaviors.csv"

    all_prompts = []

    if harmful_csv.exists():
        harmful_prompts = load_prompts_from_csv(harmful_csv)
        for p in harmful_prompts:
            p["type"] = "harmful"
            all_prompts.append(p)
        print(f"Loaded {len(harmful_prompts)} harmful prompts")

    if benign_csv.exists():
        benign_prompts = load_prompts_from_csv(benign_csv)
        for p in benign_prompts:
            p["type"] = "benign"
            all_prompts.append(p)
        print(f"Loaded {len(benign_prompts)} benign prompts")

    if not all_prompts:
        print(f"No CSV files found in {PROMPTS_DIR}")
        return

    total_prompts = len(all_prompts)

    for model in tqdm(MODELS, desc="Models", leave=True):
        model_responses = []

        for idx, prompt_info in tqdm(
            enumerate(all_prompts, 1), total=total_prompts, desc="Prompts", leave=True
        ):
            response = get_model_response(model, prompt_info["goal"])

            model_responses.append(
                {
                    "prompt_type": prompt_info["type"],
                    "prompt_index": prompt_info["index"],
                    "prompt_goal": prompt_info["goal"],
                    "prompt_category": prompt_info["category"],
                    "model": model,
                    "response": response,
                }
            )

            # checkpoint
            if idx % SAVE_EVERY == 0:
                save_responses(
                    model_responses, model, partial=True, suffix=f"_up_to_{idx}"
                )
                # to make my GPU happier :3
                time.sleep(1)

        output_file = save_responses(model_responses, model, partial=False)

    print(f"\n{'='*60}")
    print(f"All models tested!")
    print(f"Models: {', '.join(MODELS)}")
    print(f"{'='*60}")


if __name__ == "__main__":
    test_models()
