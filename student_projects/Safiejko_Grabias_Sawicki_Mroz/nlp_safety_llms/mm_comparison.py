import json
import os
import gc
import logging
import sys
import time
import datetime
import contextlib
import glob
import random
from typing import Dict, Any
from tqdm import tqdm

# --- IMPORT LLAMA ---
from llama_cpp import Llama
from llama_cpp.llama_chat_format import Llava15ChatHandler

# --- LOGGING CONFIG ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# --- CONSTANTS ---
BASE_PATH = "./"
MODEL_DIR = os.path.join(BASE_PATH, "models")
PROCESSED_QUESTIONS_DIR = os.path.join(BASE_PATH, "processed_questions")
IMAGES_DIR = os.path.join(BASE_PATH, "imgs")
OUTPUT_DIR = os.path.join(BASE_PATH, "questions_with_answers_bakllava")

# Model Paths
# PATH_VISION_MODEL = os.path.join(MODEL_DIR, "ggml-model-q4_k.gguf")
# PATH_VISION_CLIP = os.path.join(MODEL_DIR, "mmproj-model-f16.gguf")
PATH_VISION_MODEL = os.path.join(MODEL_DIR, "ggml-model-q4_k.gguf") 
PATH_VISION_CLIP = os.path.join(MODEL_DIR, "bakllava-mmproj-model-f16.gguf") 
# MODEL_NAME_BASE = "Llava-v1.5-7B-Quant"
MODEL_NAME_BASE = "BakLLaVA-1-Mistral-7B-Quant"
# --- CONFIGURATION ---
# Attack type: "SD", "TYPO", or "SD_TYPO"
SELECTED_ATTACK_TYPE = "SD_TYPO" 

SAMPLE_SIZE = 10 

ATTACK_CONFIG = {
    "SD": {"prompt_key": "Rephrased Question(SD)", "subfolder": "SD"},
    "TYPO": {"prompt_key": "Rephrased Question", "subfolder": "TYPO"},
    "SD_TYPO": {"prompt_key": "Rephrased Question", "subfolder": "SD_TYPO"}
}

# --- UTILITIES ---

@contextlib.contextmanager
def suppress_c_output():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        try:
            sys.stdout = devnull
            sys.stderr = devnull
            yield
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr

def unload_model(model_instance):
    if model_instance is not None:
        del model_instance
        gc.collect()

def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

# --- INFERENCE LOGIC ---

def run_multimodal_inference():
    # Initialize Model
    if not os.path.exists(PATH_VISION_MODEL) or not os.path.exists(PATH_VISION_CLIP):
        logger.critical(f"Model files not found in {MODEL_DIR}")
        return

    logger.info(f"Initializing Vision Model: {MODEL_NAME_BASE}...")
    try:
        with suppress_c_output():
            chat_handler = Llava15ChatHandler(clip_model_path=PATH_VISION_CLIP)
            llm_vision = Llama(
                model_path=PATH_VISION_MODEL,
                chat_handler=chat_handler,
                n_ctx=4096,
                n_gpu_layers=-1,
                seed=42,
                verbose=False
            )
    except Exception as e:
        logger.critical(f"Failed to load model: {e}")
        return

    ensure_dir(OUTPUT_DIR)
    scenario_files = glob.glob(os.path.join(PROCESSED_QUESTIONS_DIR, "*.json"))
    
    if not scenario_files:
        logger.warning(f"No JSON files found in {PROCESSED_QUESTIONS_DIR}")
        return

    logger.info(f"Found {len(scenario_files)} scenario files. Sampling {SAMPLE_SIZE} questions each.")

    # Iterate over each scenario file
    for file_path in scenario_files:
        file_name = os.path.basename(file_path)
        scenario_name = os.path.splitext(file_name)[0]
        
        logger.info(f"Processing Scenario: {scenario_name}")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        all_ids = list(data.keys())
        sampled_ids = random.sample(all_ids, min(len(all_ids), SAMPLE_SIZE))

        sampled_ids.sort(key=lambda x: int(x) if x.isdigit() else x)

        attack_cfg = ATTACK_CONFIG[SELECTED_ATTACK_TYPE]

        for q_id in tqdm(sampled_ids, desc=f"Scenario {scenario_name}", unit="q"):
            entry = data[q_id]
            
            if "ans" not in entry:
                entry["ans"] = {}

            prompt_text = entry.get(attack_cfg["prompt_key"])

            img_subfolder = attack_cfg["subfolder"]
            img_path = os.path.join(IMAGES_DIR, scenario_name, img_subfolder, f"{q_id}.jpg")

            if not os.path.exists(img_path):
                img_path_png = img_path.replace(".jpg", ".png")
                if os.path.exists(img_path_png):
                    img_path = img_path_png
                else:
                    logger.warning(f"Skipping ID {q_id}: Image missing at {img_path}")
                    continue
            
            try:
                abs_img_path = os.path.abspath(img_path)
                image_uri = f"file://{abs_img_path}"
                
                response = llm_vision.create_chat_completion(
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt_text},
                                {"type": "image_url", "image_url": {"url": image_uri}}
                            ]
                        }
                    ],
                    temperature=0.0,
                    max_tokens=1024
                )
                
                response_text = response["choices"][0]["message"]["content"]

                key_name = f"{MODEL_NAME_BASE}_{SELECTED_ATTACK_TYPE}"
                entry["ans"][key_name] = {
                    "text": response_text
                }

            except Exception as e:
                logger.error(f"Error on {scenario_name} ID {q_id}: {e}")

        output_file = os.path.join(OUTPUT_DIR, file_name)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
        
        logger.info(f"Saved sampled results to {output_file}")

    unload_model(llm_vision)
    logger.info("Inference complete.")

def main():
    start_time = time.time()
    run_multimodal_inference()
    end_time = time.time()
    elapsed = end_time - start_time
    logger.info(f"Total execution time: {elapsed:.2f}s")

if __name__ == "__main__":
    main()