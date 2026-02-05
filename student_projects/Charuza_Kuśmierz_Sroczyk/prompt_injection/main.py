import json
import os
import yaml
import logging
import time
from datetime import datetime
from tqdm import tqdm
from src.utils import set_seed, log_system_info
from src.models import ModelFactory
from src.evaluator import InjectionEvaluator

# Setup Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("experiment.log"),
        logging.StreamHandler()
    ]
)


def load_config(path="config/config.yaml"):
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def main():
    # 1. Initialization
    cfg = load_config()
    set_seed(cfg['experiment']['seed'])
    log_system_info()

    session_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    logging.info(f"Session Timestamp: {session_timestamp}")

    # 2. Load Data
    input_path = cfg['data']['input_file']
    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            dataset = json.load(f)
        logging.info(f"Loaded {len(dataset)} prompts.")
    except FileNotFoundError:
        logging.error(f"Dataset not found at: {os.path.abspath(input_path)}")
        return

    # 3. Setup Judge
    logging.info(f"Initializing Judge Model: {cfg['models']['judge_model']}")
    judge_model = ModelFactory.create_model(cfg['models']['judge_model'])
    evaluator = InjectionEvaluator(judge_model)

    # 4. Main Experiment Loop
    for model_name in cfg['models']['target_models']:
        logging.info(f"--- Starting Evaluation for: {model_name} ---")

        try:
            target_model = ModelFactory.create_model(model_name)
        except Exception as e:
            logging.error(f"Failed to load model {model_name}: {e}")
            continue

        results = []

        # Time Accumulators
        total_inference_time = 0.0
        total_evaluation_time = 0.0

        for item in tqdm(dataset, desc=f"Eval {model_name}"):
            try:
                # --- Input Preparation ---
                input_payload = item.get("prompt", "")

                # --- Multimodal Handling ---
                image_path = None
                if item.get("data_type") == "multimodal":
                    # Skip if model is not multimodal (heuristic)
                    if "llava" not in model_name.lower():
                        continue

                    if "image_path" in item:
                        image_path = item["image_path"]
                    else:
                        image_path = os.path.join(cfg['data']['images_dir'], f"{item['id']}.jpg")

                # --- Inference (with Time Measurement) ---
                start_inf = time.time()

                response = target_model.generate(
                    prompt=input_payload,
                    image_path=image_path,
                    temperature=cfg['generation']['temperature'],
                    max_tokens=cfg['generation']['max_tokens'],
                    repeat_penalty=cfg['generation'].get('repetition_penalty', 1.1)  # <--- [NOWE]
                )

                end_inf = time.time()
                total_inference_time += (end_inf - start_inf)

                # --- Evaluation (with Time Measurement) ---
                start_eval = time.time()

                score, reasoning = evaluator.evaluate(
                    user_prompt=input_payload,
                    model_response=response,
                    expected_behavior=item.get("expected_behavior", "compliant"),
                    risk_category=item.get("risk_category", "unknown"),
                    variant=item.get("variant", "unsafe")
                )

                end_eval = time.time()
                total_evaluation_time += (end_eval - start_eval)

                # Append Result (Clean, no time data here)
                results.append({
                    **item,
                    "tested_model": model_name,
                    "model_response": response,
                    "evaluation_score": score,
                    "evaluation_reasoning": reasoning,
                    "timestamp": datetime.now().isoformat()
                })

            except Exception as e:
                logging.error(f"Error processing item {item.get('id', 'unknown')}: {e}")
                results.append({
                    **item,
                    "error": str(e),
                    "tested_model": model_name
                })



        # --- 5. Save Report (Two Separate Files) ---
        os.makedirs(cfg['experiment']['output_dir'], exist_ok=True)


        current_seed = cfg['experiment']['seed']

        results_filename = f"results_{model_name}_seed{current_seed}_{session_timestamp}.json"
        results_path = os.path.join(cfg['experiment']['output_dir'], results_filename)

        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=4, ensure_ascii=False)

        # File 2: Statistics (Times and summary)
        num_prompts = len(results)
        avg_inf = total_inference_time / num_prompts if num_prompts > 0 else 0
        avg_eval = total_evaluation_time / num_prompts if num_prompts > 0 else 0

        stats_data = {
                    "model_name": model_name,
                    "seed": current_seed,
                    "session_timestamp": session_timestamp,
                    "total_prompts_processed": num_prompts,
                    "timings": {
                        "total_inference_seconds": round(total_inference_time, 2),
                        "total_evaluation_seconds": round(total_evaluation_time, 2),
                        "average_inference_per_prompt": round(avg_inf, 4),
                        "average_evaluation_per_prompt": round(avg_eval, 4)
                    },

                    "generation_config": {
                        "temperature": cfg['generation']['temperature'],
                        "max_tokens": cfg['generation']['max_tokens'],
                        "repetition_penalty": cfg['generation'].get('repetition_penalty', 1.1)
                    }
                }


        stats_filename = f"stats_{model_name}_seed{current_seed}_{session_timestamp}.json"
        stats_path = os.path.join(cfg['experiment']['output_dir'], stats_filename)

        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(stats_data, f, indent=4, ensure_ascii=False)

        logging.info(f"Saved results to: {results_filename}")
        logging.info(f"Saved stats to:   {stats_filename}")


if __name__ == "__main__":
    main()