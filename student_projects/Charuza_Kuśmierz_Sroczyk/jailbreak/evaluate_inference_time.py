# Script to evaluate inference time for all models used in jailbreak experiments

import ollama
import json
import csv
import time
from pathlib import Path
from datetime import datetime
from statistics import mean, median, stdev

TEXT_MODELS = ["qwen3:8b", "llama3.1:8b", "mistral:latest"]
MULTIMODAL_MODELS = ["llava:7b"]
GUARD_MODELS = ["llama-guard3:8b"]
ALL_MODELS = TEXT_MODELS + MULTIMODAL_MODELS + GUARD_MODELS

PROMPTS_DIR = "prompts/jailbreakBench_jbb"
IMAGES_DIR = "prompts/vscbench/vscbench_image_centric_images"
OUTPUT_DIR = "inference_time_results"
SEED = 123
NUM_TEST_PROMPTS = 50  # number of prompts to test per model


def safe_name(name: str) -> str:
    return name.replace(":", "_").replace("/", "_")


def load_sample_prompts(num_prompts=10):
    benign_csv = Path(PROMPTS_DIR) / "benign-behaviors.csv"
    prompts = []

    with open(benign_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if i >= num_prompts:
                break
            prompts.append(
                {
                    "index": row["Index"],
                    "goal": row["Goal"],
                    "category": row["Category"],
                }
            )

    return prompts


def load_sample_multimodal_prompts(num_prompts=10):
    csv_path = Path(
        "prompts/vscbench/vscbench_image_centric_images/vscbench_subset.csv"
    )

    if not csv_path.exists():
        return []

    prompts = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if i >= num_prompts:
                break

            image_filename = row["safe_img_0"]
            image_path = Path(IMAGES_DIR) / row["Category"] / image_filename
            if image_path.exists():
                prompts.append(
                    {
                        "prompt": row["Prompt"],
                        "image_path": str(image_path),
                        "category": row["Category"],
                    }
                )

    return prompts


def time_text_model(model_name, prompts):
    print(f"\nTesting {model_name}...")

    times = []
    successful = 0
    failed = 0

    for i, prompt_data in enumerate(prompts):
        prompt = prompt_data["goal"]

        try:
            messages = [{"role": "user", "content": prompt}]

            start_time = time.time()
            response = ollama.chat(
                model=model_name,
                messages=messages,
                keep_alive=0,
                options={"seed": SEED},
            )
            end_time = time.time()

            inference_time = end_time - start_time
            times.append(inference_time)
            successful += 1

            print(f"  [{i+1}/{len(prompts)}] {inference_time:.2f}s")

        except Exception as e:
            print(f"  [{i+1}/{len(prompts)}] Error: {str(e)}")
            failed += 1

    return {
        "times": times,
        "successful": successful,
        "failed": failed,
    }


def time_multimodal_model(model_name, prompts):
    """Measure inference time for multimodal models"""
    print(f"\nTesting {model_name} (multimodal)...")

    times = []
    successful = 0
    failed = 0

    for i, prompt_data in enumerate(prompts):
        prompt = prompt_data["prompt"]
        image_path = prompt_data["image_path"]

        try:
            messages = [
                {
                    "role": "user",
                    "content": prompt,
                    "images": [image_path],
                }
            ]
            start_time = time.time()
            response = ollama.chat(
                model=model_name,
                messages=messages,
                keep_alive=0,
                options={"seed": SEED},
            )
            end_time = time.time()

            inference_time = end_time - start_time
            times.append(inference_time)
            successful += 1

            print(f"  [{i+1}/{len(prompts)}] {inference_time:.2f}s")

        except Exception as e:
            print(f"  [{i+1}/{len(prompts)}] Error: {str(e)}")
            failed += 1

    return {
        "times": times,
        "successful": successful,
        "failed": failed,
    }


def time_guard_model(model_name, prompts):
    """Measure inference time for Llama Guard models"""
    print(f"\nTesting {model_name} (guard)...")

    times = []
    successful = 0
    failed = 0

    for i, prompt_data in enumerate(prompts):
        prompt = prompt_data["goal"]

        try:
            messages = [{"role": "user", "content": prompt}]

            start_time = time.time()
            response = ollama.chat(
                model=model_name,
                messages=messages,
                keep_alive=0,
            )
            end_time = time.time()

            inference_time = end_time - start_time
            times.append(inference_time)
            successful += 1

            print(f"  [{i+1}/{len(prompts)}] {inference_time:.2f}s")

        except Exception as e:
            print(f"  [{i+1}/{len(prompts)}] Error: {str(e)}")
            failed += 1

    return {
        "times": times,
        "successful": successful,
        "failed": failed,
    }


def calculate_statistics(times):
    """Calculate statistics from timing data"""
    if not times:
        return {
            "count": 0,
            "total_time": 0,
            "mean": 0,
            "median": 0,
            "std": 0,
            "min": 0,
            "max": 0,
        }

    return {
        "count": len(times),
        "total_time": sum(times),
        "mean": mean(times),
        "median": median(times),
        "std": stdev(times) if len(times) > 1 else 0,
        "min": min(times),
        "max": max(times),
    }


def evaluate_all_models():
    print("=" * 60)
    print("INFERENCE TIME EVALUATION")
    print("=" * 60)

    Path(OUTPUT_DIR).mkdir(exist_ok=True)

    print("\nLoading test prompts...")
    text_prompts = load_sample_prompts(NUM_TEST_PROMPTS)
    multimodal_prompts = load_sample_multimodal_prompts(NUM_TEST_PROMPTS)

    print(f"Loaded {len(text_prompts)} text prompts")
    print(f"Loaded {len(multimodal_prompts)} multimodal prompts")

    results = {
        "timestamp": datetime.now().isoformat(),
        "num_test_prompts": NUM_TEST_PROMPTS,
        "models": {},
    }

    for model in TEXT_MODELS:
        model_results = time_text_model(model, text_prompts)
        results["models"][model] = {
            **calculate_statistics(model_results["times"]),
            "successful": model_results["successful"],
            "failed": model_results["failed"],
        }

    if multimodal_prompts:
        for model in MULTIMODAL_MODELS:
            model_results = time_multimodal_model(model, multimodal_prompts)
            results["models"][model] = {
                **calculate_statistics(model_results["times"]),
                "successful": model_results["successful"],
                "failed": model_results["failed"],
            }
    for model in GUARD_MODELS:
        guard_results = time_guard_model(model, text_prompts)
        results["models"][model] = {
            **calculate_statistics(guard_results["times"]),
            "successful": guard_results["successful"],
            "failed": guard_results["failed"],
        }

    output_file = (
        Path(OUTPUT_DIR)
        / f"inference_times_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)

    for model_name, stats in results["models"].items():
        print(f"\n{model_name}:")
        print(f"  Mean: {stats['mean']:.2f}s")
        print(f"  Median: {stats['median']:.2f}s")
        print(f"  Std: {stats['std']:.2f}s")
        print(f"  Min/Max: {stats['min']:.2f}s / {stats['max']:.2f}s")
        print(
            f"  Success rate: {stats['successful']}/{stats['successful'] + stats['failed']}"
        )

    print(f"\n\nResults saved to: {output_file}")

    return results


if __name__ == "__main__":
    evaluate_all_models()
