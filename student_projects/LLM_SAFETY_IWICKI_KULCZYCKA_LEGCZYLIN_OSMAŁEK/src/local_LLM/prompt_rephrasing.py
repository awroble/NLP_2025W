import json
import uuid
import random
from copy import deepcopy
from typing import Dict, List
import re
from tqdm import tqdm

from src.local_LLM.llm_manager import LLMManager


# ============================================================
# CONFIG
# ============================================================
random.seed(4242)
MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"
BATCH_SIZE = 8



def build_rewrite_prompt(text: str) -> str:
    return f"""You are a TEXT TRANSFORMATION ENGINE.

Your ONLY task is to rewrite the given text using different wording.

STRICT RULES:
- This is a TRANSFORMATION task, NOT a response.
- DO NOT answer the question.
- DO NOT provide advice, explanations, steps, or recommendations.
- DO NOT diagnose, evaluate, or interpret anything.
- DO NOT add, remove, or infer information.
- DO NOT include multiple versions.
- DO NOT include meta-text (e.g. "rewritten text", "note", "version").
- DO NOT include phrases such as:
  "you should", "you can", "consider", "to address this",
  "it could be", "this may indicate", "I recommend".

OUTPUT FORMAT:
- Output ONLY the rewritten text.
- Exactly ONE rewritten version.
- No headings, no bullet points, no commentary.

Rewrite the following text using different wording:

{text}
"""


# ============================================================
# OUTPUT CLEANUP
# ============================================================


def clean_output(text: str) -> str:
    if not text:
        return text

    text = text.strip()

    text = text.replace("```", "")
    text = text.replace("_", "")
    text = text.replace("-", "")


    cutoff_patterns = [
        r"\n\s*Rewritten text\s*:",
        r"\n\s*Rewrite\s+the\s+following\s+text",
        r"\n\s*Corrected\s+rewritten\s+text",
        r"\n\s*Note\s*:",
        r"\n\s*\(Note:",
        r"\n\s*Here is another text",
        r"\n\s*python\s*\n",
    ]

    for pattern in cutoff_patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            text = text[:match.start()].strip()
            break


    text = re.sub(
        r"^\s*Rewritten\s+text\s*:\s*",
        "",
        text,
        flags=re.IGNORECASE,
    )

    lines = [l.strip() for l in text.splitlines() if l.strip()]

    cleaned_lines = []
    seen = set()

    for line in lines:
        normalized = re.sub(r"\s+", " ", line.lower())
        if normalized in seen:
            continue
        seen.add(normalized)

        if any(
            kw in normalized
            for kw in [
                "this is a paraphrased version",
                "the rewritten text",
                "i rewrote the text",
                "original meaning",
                "original intent",
                "end of text",
                "end of note",
            ]
        ):
            continue

        cleaned_lines.append(line)

    if not cleaned_lines:
        return text

    text = " ".join(cleaned_lines)

    return text.strip()



# ============================================================
# BATCH REPHRASE
# ============================================================

def rephrase_texts_batch(
    llm: LLMManager,
    texts: List[str],
    max_new_tokens: int = 256,
) -> List[str]:

    prompts = [build_rewrite_prompt(t) for t in texts] 
    outputs = llm.generate_single_turn(
        prompts=prompts,
        batch_size=BATCH_SIZE,
        max_new_tokens=max_new_tokens,
        temperature=0.3,  
        top_p=0.9,
    )

    return [clean_output(o) for o in outputs]


# ============================================================
# MUTATION LOGIC
# ============================================================

def generate_new_records(
    records: List[Dict],
    llm: LLMManager,
    num_new_records: int,
) -> List[Dict]:

    new_records: List[Dict] = []

    for _ in tqdm(
        range(0, num_new_records, BATCH_SIZE),
        desc="Generating rephrased records",
    ):
        batch_size = min(BATCH_SIZE, num_new_records - len(new_records))
        batch = [deepcopy(random.choice(records)) for _ in range(batch_size)]

        user_texts: List[str] = []
        msg_refs: List[Dict] = []

        for rec in batch:
            for msg in rec.get("prompt", []):
                if msg.get("role") == "user" and "text" in msg:
                    user_texts.append(msg["text"])
                    msg_refs.append(msg)

        if not user_texts:
            continue

        rewritten_texts = rephrase_texts_batch(llm, user_texts)

        for msg, new_text in zip(msg_refs, rewritten_texts):
            msg["text"] = new_text  

        for rec in batch:
            rec["id"] = str(uuid.uuid4())
            new_records.append(rec)

    return new_records
