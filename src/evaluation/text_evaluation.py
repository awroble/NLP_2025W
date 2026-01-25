import json
import time
from pathlib import Path
from copy import deepcopy
from tqdm import tqdm
from src.local_LLM.llm_manager import LLMManager
import gc
import torch

torch.backends.cuda.matmul.allow_tf32 = True
torch.manual_seed(42)
VICTIM_BATCH_SIZE = 10
JUDGE_BATCH_SIZE  = 10
MAX_NEW_TOKENS_VICTIM = 256
JUDGE_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
JUDGE_PROMPT = Path("src/evaluation/judge_prompt.txt").read_text("utf-8")


def clear_vram():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False) 


def serialize_for_judge(conversation):
    return "\n\n".join(
        f"{t['role'].upper()}:\n{t['content']}"
        for t in conversation
    )


def build_single_turn_histories(dataset):
    histories = []
    for ex in dataset:
        h = []
        for t in ex["prompt"]:
            h.append({"role": t["role"], "content": t["text"]})
        histories.append(h)
    return histories


def evaluate_single_turn(input_path, output_path, VICTIM_MODEL):
    dataset = load_json(Path(input_path))
    manager = LLMManager()
    results = deepcopy(dataset)

    histories = build_single_turn_histories(dataset)

    # ---------------- VICTIM ----------------
    t0 = time.perf_counter()
    victim_outputs = []

    pbar = tqdm(
        range(0, len(histories), VICTIM_BATCH_SIZE),
        desc="Victim",
        unit="batch",
        dynamic_ncols=True,
    )
    clear_vram()
    for i in pbar:
        batch = histories[i:i + VICTIM_BATCH_SIZE]
        out = manager.generate_chat_once(
            VICTIM_MODEL,
            batch,
            batch_size=VICTIM_BATCH_SIZE,
            max_new_tokens=MAX_NEW_TOKENS_VICTIM,
        )
        victim_outputs.extend(out)
        elapsed = time.perf_counter() - t0
        pbar.set_postfix(s_per_sample=f"{elapsed / len(victim_outputs):.3f}")

    victim_time = time.perf_counter() - t0
    pbar.close()

    conversations = [
        h + [{"role": "assistant", "content": r}]
        for h, r in zip(histories, victim_outputs)
    ]

    # ---------------- JUDGE ----------------
    judge_inputs = [serialize_for_judge(c) for c in conversations]
    behaviours = [ex["metadata"]["expected_behaviour"] for ex in dataset]

    judge_prompts = [
        JUDGE_PROMPT
        .replace("{conversation}", c)
        .replace("{expected_behaviour}", b)
        for c, b in zip(judge_inputs, behaviours)
    ]

    verdicts = []
    t1 = time.perf_counter()

    pbar = tqdm(
        range(0, len(judge_prompts), JUDGE_BATCH_SIZE),
        desc="Judge",
        unit="batch",
        dynamic_ncols=True,
    )
    clear_vram()
    manager.load_text_model(JUDGE_MODEL)
    for i in pbar:
        out = manager.generate_single_turn(
            judge_prompts[i:i + JUDGE_BATCH_SIZE],
            batch_size=JUDGE_BATCH_SIZE,
            max_new_tokens=1,
            temperature=0.0,   
            top_p=1.0,
        )

        verdicts.extend(v.strip() for v in out)
        elapsed = time.perf_counter() - t1
        pbar.set_postfix(s_per_sample=f"{elapsed / len(verdicts):.3f}")

    judge_time = time.perf_counter() - t1
    pbar.close()

    manager.unload_model()

    for rec, convo, verdict in zip(results, conversations, verdicts):
        rec["conversation"] = convo
        rec["judge_verdict"] = verdict

    stats = {
        "samples": len(dataset),
        "victim_time_s": round(victim_time, 2),
        "judge_time_s": round(judge_time, 2),
        "total_time_s": round(victim_time + judge_time, 2),
        "victim_s_per_sample": round(victim_time / len(dataset), 3),
        "judge_s_per_sample": round(judge_time / len(dataset), 3),
    }
    save_json(Path(output_path), results)
    return stats



def evaluate_multiturn(input_path, output_path, VICTIM_MODEL):
    dataset = load_json(Path(input_path))
    manager = LLMManager()
    results = deepcopy(dataset)
    clear_vram()
    manager.load_text_model(VICTIM_MODEL)
    all_conversations = []

    t0 = time.perf_counter()
    pbar = tqdm(dataset, desc="Victim (multiturn)", unit="sample", dynamic_ncols=True)

    for ex in pbar:
        history = []

        for turn in ex["prompt"]:
            history.append({
                "role": turn["role"],
                "content": turn["text"]
            })

            if turn["role"] == "user":
                out = manager.generate_chat_once(
                    VICTIM_MODEL,
                    [history],          
                    batch_size=1,
                    max_new_tokens=MAX_NEW_TOKENS_VICTIM,
                )

                assistant_msg = out[0]
                history.append({
                    "role": "assistant",
                    "content": assistant_msg
                })

        all_conversations.append(history)

        elapsed = time.perf_counter() - t0
        pbar.set_postfix(s_per_sample=f"{elapsed / len(all_conversations):.3f}")

    victim_time = time.perf_counter() - t0
    pbar.close()

    # ---------------- JUDGE ----------------
    judge_inputs = [serialize_for_judge(c) for c in all_conversations]
    behaviours = [ex["metadata"]["expected_behaviour"] for ex in dataset]

    judge_prompts = [
        JUDGE_PROMPT
        .replace("{conversation}", c)
        .replace("{expected_behaviour}", b)
        for c, b in zip(judge_inputs, behaviours)
    ]

    verdicts = []
    t1 = time.perf_counter()

    pbar = tqdm(
        range(0, len(judge_prompts), JUDGE_BATCH_SIZE),
        desc="Judge",
        unit="batch",
        dynamic_ncols=True,
    )
    clear_vram()
    manager.load_text_model(JUDGE_MODEL)
    for i in pbar:
        out = manager.generate_single_turn(
            judge_prompts[i:i + JUDGE_BATCH_SIZE],
            batch_size=JUDGE_BATCH_SIZE,
            max_new_tokens=1,
            temperature=0.0,
            top_p=1.0,
        )
        verdicts.extend(v.strip() for v in out)

        elapsed = time.perf_counter() - t1
        pbar.set_postfix(s_per_sample=f"{elapsed / len(verdicts):.3f}")

    judge_time = time.perf_counter() - t1
    pbar.close()

    manager.unload_model()

    for rec, convo, verdict in zip(results, all_conversations, verdicts):
        rec["conversation"] = convo
        rec["judge_verdict"] = verdict

    stats = {
        "samples": len(dataset),
        "victim_time_s": round(victim_time, 2),
        "judge_time_s": round(judge_time, 2),
        "total_time_s": round(victim_time + judge_time, 2),
        "victim_s_per_sample": round(victim_time / len(dataset), 3),
        "judge_s_per_sample": round(judge_time / len(dataset), 3),
    }

    save_json(Path(output_path), results)
    return stats