import json
import time
import torch
from pathlib import Path
from copy import deepcopy
from tqdm import tqdm
import gc
from src.local_LLM.llm_manager import LLMManager

torch.backends.cuda.matmul.allow_tf32 = True
torch.manual_seed(42)

# ============================================================
# UTILS
# ============================================================

def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

def save_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def serialize_for_judge(conversation):
    return "\n\n".join(
        f"{turn['role'].upper()}:\n{turn['content']}"
        for turn in conversation
    )

def extract_assistant_text(text: str) -> str:
    if "ASSISTANT:" in text:
        return text.split("ASSISTANT:", 1)[1].strip()
    return text.strip()


def clear_vram():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
# ============================================================
# CORE PIPELINE
# ============================================================

def run_multimodal_conversations_batch(
    dataset,
    manager: LLMManager,
    victim_model: str,
    batch_size: int,
):
    prompts, images = [], []

    for ex in dataset:
        user_turn = ex["prompt"][0]["text"]
        mm_prompt = "<image>\nUSER: " + user_turn + "\nASSISTANT:"
        prompts.append(mm_prompt)
        images.append(ex["image"])

    conversations = []
    manager.load_vision_model(victim_model)

    pbar = tqdm(
        range(0, len(prompts), batch_size),
        desc="Victim-MM",
        unit="Batch",
        dynamic_ncols=True
    )
    for i in pbar:
        batch_p = prompts[i:i + batch_size]
        batch_i = images[i:i + batch_size]

        outputs = manager.generate_multimodal(
            model_id=victim_model,
            prompts=batch_p,
            image_paths=batch_i,
            max_new_tokens=128,
            batch_size=len(batch_p),
        )

        for p, o in zip(batch_p, outputs):
            conversations.append([
                {
                    "role": "user",
                    "content": p.replace("<image>\n", "").replace("USER:", "").replace("ASSISTANT:", "").strip()
                },
                {
                    "role": "assistant",
                    "content": extract_assistant_text(o)
                }
            ])

    manager.unload_model()
    return conversations


def judge_conversations_batch(
    conversations,
    dataset,
    manager: LLMManager,
    judge_model: str,
    judge_prompt: str,
    batch_size: int = 8,
):
    manager.load_text_model(judge_model)

    judge_prompts = []
    for convo, ex in zip(conversations, dataset):
        prompt = (
            judge_prompt
            .replace("{conversation}", serialize_for_judge(convo))
            .replace("{expected_behaviour}", ex["metadata"]["expected_behaviour"])
        )
        judge_prompts.append(prompt)

    verdicts = []

    pbar = tqdm(
        range(0, len(judge_prompts), batch_size),
        desc="Judge",
        unit="batch",
        dynamic_ncols=True
    )

    with torch.no_grad():
        for i in pbar:
            batch = judge_prompts[i:i + batch_size]
            outputs = manager.generate_single_turn(
                prompts=batch,
                max_new_tokens=1,
                temperature=0.0,
                top_p=1.0,
            )
            verdicts.extend(o.strip() for o in outputs)

    manager.unload_model()
    torch.cuda.empty_cache()
    return verdicts


# ============================================================
# NOTEBOOK ENTRYPOINT
# ============================================================

def evaluate_multimodal(
        input_path: str,
        output_path: str,
        victim_model: str = "llava-hf/llava-1.5-7b-hf",
        judge_model: str = "meta-llama/Llama-3.1-8B-Instruct",
        victim_batch_size: int = 4,
        judge_batch_size: int = 8,
        judge_prompt_path: str = "src/evaluation/judge_prompt.txt",
    ):
        manager = LLMManager()
        clear_vram()
        dataset = load_json(Path(input_path))
        judge_prompt = Path(judge_prompt_path).read_text(encoding="utf-8")

        t0 = time.perf_counter()

        conversations = run_multimodal_conversations_batch(
            dataset,
            manager,
            victim_model,
            victim_batch_size,
        )  
        victim_time = time.perf_counter() - t0
        clear_vram()
        t1 = time.perf_counter()
        
        verdicts = judge_conversations_batch(
            conversations,
            dataset,
            manager,
            judge_model,
            judge_prompt,
            judge_batch_size,
        )
        judge_time = time.perf_counter() - t1
        results = deepcopy(dataset)
        for r, c, v in zip(results, conversations, verdicts):
            r["conversation"] = c
            r["judge_verdict"] = v
        
        save_json(Path(output_path), results)

        stats = {
            "samples": len(dataset),
            "victim_time_s": round(victim_time, 2),
            "judge_time_s": round(judge_time, 2),
            "total_time_s": round(victim_time + judge_time, 2),
            "victim_s_per_sample": round(victim_time / len(dataset), 3),
            "judge_s_per_sample": round(judge_time / len(dataset), 3),
        }
        return stats