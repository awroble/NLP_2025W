import os
import time
import hashlib
import re
import argparse
from typing import List, Iterator

import openai

openai.api_key = os.getenv("OPENAI_API_KEY")

SYSTEM_PROMPT_FILE_PARSING = "pipeline_prompts/parsing_system_prompt.txt"
SYSTEM_PROMPT_FILE_CLASSIFICATION = "pipeline_prompts/classification_system_prompt.txt"

MODEL = "gpt-5-mini"
MAX_CHARS_PER_CHUNK = 18000
SLEEP_SECONDS_BETWEEN_CALLS = 0.5

def iter_files(root: str) -> Iterator[str]:
    for dirpath, _, filenames in os.walk(root):
        if dirpath == root:
            continue
        for fn in filenames:
            full = os.path.join(dirpath, fn)
            if os.path.isfile(full):
                yield full

def chunk_text(text: str, max_chars: int) -> List[str]:
    """
    Split text into size-limited chunks without breaking sentences or list items.

    Heuristics:
    - Prefer splitting at a double newline (paragraph boundary).
    - Otherwise split after sentence-ending punctuation (.!?), followed by whitespace and a capital/number.
    - Never cut inside a line: fallback boundary will be the last newline before limit (if reasonably sized).
    - If no suitable backward boundary is found, look forward up to +10% for a boundary to avoid mid‑sentence cuts.
    """
    if len(text) <= max_chars:
        return [text]

    boundaries = []
    start = 0
    text_length = len(text)
    while start < text_length:  
        end = min(start + max_chars, text_length)

        double_newline_pos = text.rfind("\n\n", start, end)
        sentence_end_pos = None
        for match in re.finditer(r'([.!?])(\s+)([A-Z0-9])', text[start:end]):
            sentence_end_pos = match.end(1) + start

        chosen_boundary = None
        if double_newline_pos != -1:
            chosen_boundary = double_newline_pos + 2
        elif sentence_end_pos is not None:
            chosen_boundary = sentence_end_pos

        if chosen_boundary is None:
            last_newline_pos = text.rfind("\n", start, end)
            if last_newline_pos != -1 and last_newline_pos - start > max_chars // 4:
                chosen_boundary = last_newline_pos + 1

        if chosen_boundary is None:
            look_ahead_limit = min(end + max_chars // 10, text_length)
            forward_double_newline = text.find("\n\n", end, look_ahead_limit)
            forward_sentence_end = None
            for match in re.finditer(r'([.!?])(\s+)([A-Z0-9])', text[end:look_ahead_limit]):
                forward_sentence_end = match.end(1) + end

            if forward_double_newline != -1:
                chosen_boundary = forward_double_newline + 2
            elif forward_sentence_end is not None:
                chosen_boundary = forward_sentence_end

        if chosen_boundary is None:
            chosen_boundary = end

        boundaries.append((start, chosen_boundary))
        start = chosen_boundary

def call_llm(content: str, system_prompt: str) -> str:
    resp = openai.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": content}
        ]
    )
    return resp.choices[0].message.content.strip()

def ensure_output_dir(output_dir: str):
    os.makedirs(output_dir, exist_ok=True)

def output_path(root_dir: str, output_dir: str, input_path: str, chunk_index: int = None) -> str:
    """
    Reproduce original subdirectory structure inside output_dir.
    """
    rel = os.path.relpath(input_path, root_dir)
    subdir, filename = os.path.split(rel)
    base_hash = hashlib.sha1(rel.encode()).hexdigest()[:8]
    stem, _ext = os.path.splitext(filename)

    if chunk_index is None:
        out_filename = f"{stem}.json"
    else:
        out_filename = f"{stem}__chunk{chunk_index}.json"

    target_dir = os.path.join(output_dir, subdir) if subdir else output_dir
    os.makedirs(target_dir, exist_ok=True)
    return os.path.join(target_dir, out_filename)

def load_system_prompt(system_prompt_type: str) -> str:
    if system_prompt_type == "parsing":
        fname = SYSTEM_PROMPT_FILE_PARSING
    elif system_prompt_type == "classification":
        fname = SYSTEM_PROMPT_FILE_CLASSIFICATION
    else:
        raise ValueError("system_prompt_type must be 'parsing' or 'classification'")

    if not os.path.isfile(fname):
        raise FileNotFoundError(f"System prompt file not found: {fname}")

    with open(fname, "r", encoding="utf-8") as f:
        return f.read().strip()

def process_file(path: str, root_dir: str, output_dir: str, system_prompt: str):
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        raw = f.read()
    chunks = chunk_text(raw, MAX_CHARS_PER_CHUNK)

    extracted_chunks = []
    for i, chunk in enumerate(chunks):
        user_content = f"FILE_PATH: {path}\nCHUNK_INDEX: {i}\nCONTENT START\n{chunk}\nCONTENT END"
        extracted = call_llm(user_content, system_prompt)
        extracted_chunks.append(extracted.strip())

        # Optional: small pause to respect rate limits
        time.sleep(SLEEP_SECONDS_BETWEEN_CALLS)

    merged = "\n".join(
        line for part in extracted_chunks
        for line in part.splitlines()
        if line.strip() != ""
    )

    out_path = output_path(root_dir, output_dir, path, None)
    with open(out_path, "w", encoding="utf-8") as out:
        out.write(merged)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract institutional statements using an LLM."
    )
    parser.add_argument(
        "--system-prompt-type",
        choices=["classification", "parsing"],
        required=True,
        help="Which system prompt to use: 'classification' or 'parsing'.",
    )
    parser.add_argument(
        "--root-dir",
        required=True,
        help="Root directory containing input files.",
    )
    parser.add_argument(
        "--output-dir-suffix",
        required=True,
        help="Suffix appended to root-dir name to form output directory name.",
    )
    return parser.parse_args()

def main():
    args = parse_args()

    root_dir = args.root_dir
    output_dir = f"{os.path.basename(root_dir)}-{args.output_dir_suffix}"
    output_dir = os.path.join(os.path.dirname(root_dir), output_dir)

    system_prompt = load_system_prompt(args.system_prompt_type)

    ensure_output_dir(output_dir)
    files = list(iter_files(root_dir))
    print(f"Found {len(files)} files to process.")
    for idx, fp in enumerate(files, 1):
        print(f"[{idx}/{len(files)}] Processing {fp}")
        try:
            process_file(fp, root_dir, output_dir, system_prompt)
        except Exception as e:
            print(f"Error processing {fp}: {e}")

if __name__ == "__main__":
    main()