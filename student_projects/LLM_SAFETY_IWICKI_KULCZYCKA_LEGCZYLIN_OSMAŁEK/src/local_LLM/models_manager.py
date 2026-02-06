import os
import gc
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import snapshot_download
from transformers import TextStreamer


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)
def clear_vram():
    """Fully clears GPU VRAM + Python references."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


# -------------------------------------------------------------------------
# MODEL DOWNLOAD
# -------------------------------------------------------------------------

def download_model_once(model_name: str):
    target_path = os.path.join(MODELS_DIR, model_name.replace("/", "__"))

    if os.path.exists(target_path):
        return target_path

    print(f">>> Downloading model '{model_name}' to folder: {target_path}")

    snapshot_download(
        repo_id=model_name,
        local_dir=target_path,
        local_dir_use_symlinks=False,
        resume_download=True,
        allow_patterns=["*"]
    )

    print(f">>> Model '{model_name}' downloaded and saved locally.")
    return target_path


# -------------------------------------------------------------------------
# MODEL LOADING
# -------------------------------------------------------------------------

def load_model(model_name: str, dtype=torch.float16):
    local_path = download_model_once(model_name)

    tokenizer = AutoTokenizer.from_pretrained(local_path)

    model = AutoModelForCausalLM.from_pretrained(
        local_path,
        torch_dtype=(dtype if DEVICE == "cuda" else torch.float32),
        device_map=DEVICE
    )

    model.eval()
    return model, tokenizer


# -------------------------------------------------------------------------
# SHARED GENERATION FUNCTION
# -------------------------------------------------------------------------

def _generate(model, tokenizer, prompt: str, max_new_tokens: int, do_sample=True,temperature: float = 0.7):
    """Unified generation helper with proper device handling."""
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            top_p=0.9,
            top_k=40,
            pad_token_id=tokenizer.eos_token_id,
            streamer=streamer
        )

    return tokenizer.decode(output[0], skip_special_tokens=True)

def _generate_completion_only(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int,
    do_sample: bool = False,
    temperature: float = 0.9,
):
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    input_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            top_p=1.0,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated_tokens = output[0][input_len:]

    return tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True
    ).strip()

def _cleanup(model, tokenizer):
    """Release model & tokenizer completely."""
    del model
    del tokenizer
    clear_vram()


# -------------------------------------------------------------------------
# GENERATION FUNCTIONS
# -------------------------------------------------------------------------

def generate_paraphrase(model_name: str, text: str):
    model, tokenizer = load_model(model_name)

    prompt = (
       "I need you to rephrase the following text in a different way, "
        "using different wording and structure while preserving the original meaning." \
         f"Test to rephrase:\n{text}. Return only the rephrased text."
    )

    result = _generate_completion_only(
        model,
        tokenizer,
        prompt,
        max_new_tokens=min(100, len(tokenizer(text)["input_ids"]) + 20),
        do_sample=True,
        temperature=0.5,
    )

    _cleanup(model, tokenizer)
    return result

def victim_generate(model_name: str, text: str):
    model, tokenizer = load_model(model_name)

    result = _generate_completion_only(
        model,
        tokenizer,
        text,
        max_new_tokens=200,
        do_sample=True,
        temperature=0.5,
    )

    _cleanup(model, tokenizer)
    return result


from pathlib import Path
def evaluate_safety(judge_model_name: str, user_prompt: str, model_response: str, attack = True):
    model, tokenizer = load_model(judge_model_name)

    if attack: 
        path = "src\local_LLM\unsafe_prompt.txt" 
    else: 
        path = "src\local_LLM\safe_prompt.txt"

    prompt = Path(path).read_text(encoding="utf-8")
    prompt = prompt.replace("{user_prompt}", user_prompt).replace("{model_output}", model_response)
    text = _generate_completion_only(model, tokenizer, prompt, max_new_tokens=1, do_sample=False,temperature=0).upper().strip()

    _cleanup(model, tokenizer)

    return text


