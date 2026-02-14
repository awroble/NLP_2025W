import os
import time
import re
import math
import json
import argparse
from dataclasses import dataclass
from typing import List, Iterator, Optional, Literal, Tuple, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed

import openai
from google import genai
from google.genai import types

# -----------------------------
# Configuration
# -----------------------------
SYSTEM_PROMPT_FILE_PARSING = "pipeline_prompts/parsing_system_prompt.txt"
SYSTEM_PROMPT_FILE_CLASSIFICATION = "pipeline_prompts/classification_system_prompt.txt"

DEFAULT_MODEL = "gemini-3-flash-preview"

# Chunking for model calls (packing budget)
DEFAULT_MAX_CHARS_PER_CHUNK = 1500

# Semantic segmentation (paragraph building)
DEFAULT_EMBED_MODEL = "text-embedding-004"
DEFAULT_SEMANTIC_SIM_THRESHOLD = 0.6
DEFAULT_MIN_SEGMENT_CHARS = 250
DEFAULT_MAX_SEGMENT_CHARS = 1200

DEFAULT_WORKERS = 8
DEFAULT_SLEEP_SECONDS = 0.0  # keep 0; sleep kills throughput


# -----------------------------
# IO helpers
# -----------------------------
def iter_files(root: str) -> Iterator[str]:
    for dirpath, _, filenames in os.walk(root):
        if dirpath == root:
            continue
        for fn in filenames:
            full = os.path.join(dirpath, fn)
            if os.path.isfile(full):
                yield full


def ensure_output_dir(output_dir: str):
    os.makedirs(output_dir, exist_ok=True)


def output_path(
    root_dir: str,
    output_dir: str,
    input_path: str,
    *,
    stage_name: str,
) -> str:
    """
    Mirrors directory structure under output_dir.
    Writes one file per stage: <stem>__<stage>.json
    """
    rel = os.path.relpath(input_path, root_dir)
    subdir, filename = os.path.split(rel)
    stem, _ext = os.path.splitext(filename)

    out_filename = f"{stem}.json"
    target_dir = os.path.join(output_dir, stage_name, subdir) if subdir else output_dir
    os.makedirs(target_dir, exist_ok=True)
    return os.path.join(target_dir, out_filename)


def load_text_file(path: str) -> str:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"File not found: {path}")
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def save_text_file(path: str, content: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content or "")


def load_system_prompt(path: str) -> str:
    txt = load_text_file(path).strip()
    if not txt:
        raise ValueError(f"System prompt is empty: {path}")
    return txt


# -----------------------------
# Provider abstraction
# -----------------------------
Provider = Literal["openai", "google"]


def infer_provider(model: str) -> Provider:
    return "google" if (model or "").lower().strip().startswith("gemini-") else "openai"


@dataclass(frozen=True)
class LLMConfig:
    provider: Provider
    model: str
    temperature: float = 0.0
    max_output_tokens: Optional[int] = None
    response_mime_type: Optional[str] = "application/json"
    sleep_seconds_between_calls: float = DEFAULT_SLEEP_SECONDS


class LLMClient:
    def generate(self, *, system_prompt: str, user_content: str) -> str:
        raise NotImplementedError


class EmbeddingClient:
    def embed_texts(self, texts: List[str], *, embed_model: str) -> List[List[float]]:
        raise NotImplementedError


class OpenAIClient(LLMClient):
    def __init__(self, config: LLMConfig):
        self.config = config
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise EnvironmentError("Set OPENAI_API_KEY in the environment.")
        openai.api_key = api_key

    def generate(self, *, system_prompt: str, user_content: str) -> str:
        resp = openai.chat.completions.create(
            model=self.config.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ]
        )
        text = (resp.choices[0].message.content or "").strip()
        return strip_markdown_fences(text)


class GoogleGeminiClient(LLMClient, EmbeddingClient):
    def __init__(self, config: LLMConfig):
        self.config = config
        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise EnvironmentError("Set GEMINI_API_KEY (or GOOGLE_API_KEY) in the environment.")
        self.client = genai.Client(api_key=api_key)

    def generate(self, *, system_prompt: str, user_content: str) -> str:
        resp = self.client.models.generate_content(
            model=self.config.model,
            contents=[types.Content(role="user", parts=[types.Part(text=user_content)])],
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                temperature=self.config.temperature,
                max_output_tokens=self.config.max_output_tokens,
                response_mime_type=self.config.response_mime_type,
            ),
        )
        text = strip_markdown_fences((resp.text or "").strip())
        return text

    def embed_texts(self, texts: List[str], *, embed_model: str) -> List[List[float]]:
        # Batch embedding (SDK supports list inputs)
        resp = self.client.models.embed_content(
            model=embed_model,
            contents=texts,
        )

        # SDK response shape varies; handle common cases
        if hasattr(resp, "embeddings") and resp.embeddings:
            out: List[List[float]] = []
            for e in resp.embeddings:
                out.append(list(getattr(e, "values", []) or []))
            return out

        if hasattr(resp, "embedding") and resp.embedding:
            return [list(getattr(resp.embedding, "values", []) or [])]

        raise RuntimeError("Unexpected embeddings response shape from Google SDK.")


def build_llm_client(cfg: LLMConfig) -> LLMClient:
    if cfg.provider == "openai":
        return OpenAIClient(cfg)
    if cfg.provider == "google":
        return GoogleGeminiClient(cfg)
    raise ValueError(f"Unsupported provider: {cfg.provider}")


# -----------------------------
# Output cleanup
# -----------------------------
def strip_markdown_fences(text: str) -> str:
    """
    Remove leading ```json (or ```) and trailing ``` if present.
    Keeps the interior exactly as-is.
    """
    t = (text or "").strip()
    if t.lower().startswith("```json"):
        nl = t.find("\n")
        t = t[nl + 1 :] if nl != -1 else ""
    elif t.startswith("```"):
        nl = t.find("\n")
        t = t[nl + 1 :] if nl != -1 else ""
    t = t.strip()
    if t.endswith("```"):
        t = t[: -3].rstrip()
    return t

def strip_json_array(text: str, place: Literal["start", "body", "end"]) -> str:
    """
    Remove leading ```json (or ```) and trailing ``` if present.
    Keeps the interior exactly as-is.
    """
    t = (text or "").strip()
    if place in ["start", "body"]:
        if t.endswith("}\n]"):
            t = t[:-2] + ",\n"
    
    if place in ["end", "body"]:
        if t.startswith("["):
            t = t[1:]
    
    return t


# -----------------------------
# Baseline block splitting (structure-aware)
# -----------------------------
_BULLET_RE = re.compile(r"^\s*(?:[-*•]|(?:\d+|[a-zA-Z])[\.\)])\s+")
_LISTHEAD_RE = re.compile(r"^.*:$")
_HEADING_RE = re.compile(r"^\s{0,3}#{1,6}\s+")
_HRULE_RE = re.compile(r"^\s*[-*_]{3,}\s*$")


def _is_list_line(line: str) -> bool:
    return bool(_BULLET_RE.match(line)) or bool(_LISTHEAD_RE.match(line))


def _is_heading_line(line: str) -> bool:
    return bool(_HEADING_RE.match(line))


def _is_separator_line(line: str) -> bool:
    return bool(_HRULE_RE.match(line))


def split_into_blocks(text: str) -> List[str]:
    """
    Creates candidate blocks from messy text:
      - contiguous list items (with indented continuation lines) become list blocks
      - everything else becomes paragraph blocks
      - blank lines and horizontal rules split blocks
    """
    lines = (text or "").splitlines()
    blocks: List[List[str]] = []

    current: List[str] = []
    current_kind: Optional[str] = None  # "para" | "list"

    def flush():
        nonlocal current, current_kind
        if current:
            blocks.append(current)
        current = []
        current_kind = None

    for ln in lines:
        raw = ln.rstrip("\n")
        stripped = raw.strip()

        if stripped == "" or _is_separator_line(raw):
            flush()
            continue

        if _is_list_line(raw):
            current_kind = "list"
            current.append(raw)
            continue

        # hanging indent continuation of an active list block
        if current_kind == "list" and (len(raw) - len(raw.lstrip(" ")) >= 2):
            current.append(raw)
            continue

        # paragraph line
        if current_kind in (None, "para"):
            current_kind = "para"
            current.append(raw)
        else:
            flush()
            current_kind = "para"
            current.append(raw)

    flush()
    return ["\n".join(b).strip() for b in blocks if "\n".join(b).strip()]


# -----------------------------
# Semantic paragraph segmentation
# -----------------------------
def _cosine(a: List[float], b: List[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    denom = na * nb
    return dot / denom if denom else 0.0


def _looks_like_continuation(prev_text: str, next_text: str) -> bool:
    p = (prev_text or "").rstrip()
    n = (next_text or "").lstrip()
    if not p or not n:
        return False

    # If previous doesn't end strongly and next starts lower-case -> likely wrapped line
    if (not re.search(r"[.!?:]\s*$", p)) and re.match(r"^[a-z]", n):
        return True

    # connector words
    if re.match(r"^(and|or|but|because|however|therefore|thus|also|then|which|that)\b", n, flags=re.I):
        return True

    return False


def semantic_segments(
    *,
    text: str,
    embedder: Optional[EmbeddingClient],
    embed_model: str,
    similarity_threshold: float,
    min_segment_chars: int,
    max_segment_chars: int,
) -> List[str]:
    """
    Build semantically-coherent paragraphs by merging adjacent candidate blocks.

    If embeddings are unavailable, falls back to candidate blocks (still better than raw lines).
    """
    blocks = split_into_blocks(text)
    if not blocks:
        return [text] if text else []

    if embedder is None:
        return blocks

    try:
        vecs = embedder.embed_texts(blocks, embed_model=embed_model)
        if len(vecs) != len(blocks):
            raise RuntimeError("Embedding count mismatch.")
    except Exception:
        print("Embedding error occured, no semantic blocks")
        return blocks

    out: List[str] = []
    buf = blocks[0].strip()
    buf_vec = vecs[0]

    def avg_vec(a: List[float], b: List[float]) -> List[float]:
        return [(x + y) / 2.0 for x, y in zip(a, b)]

    for i in range(1, len(blocks)):
        nxt = blocks[i].strip()
        if not nxt:
            continue

        sim = _cosine(buf_vec, vecs[i])

        should_merge = False
        if len(buf) < min_segment_chars:
            should_merge = True
        elif _looks_like_continuation(buf, nxt):
            should_merge = True
        elif sim >= similarity_threshold:
            should_merge = True

        if should_merge and (len(buf) + 2 + len(nxt) <= max_segment_chars):
            buf = f"{buf}\n\n{nxt}"
            buf_vec = avg_vec(buf_vec, vecs[i])
        else:
            out.append(buf.strip())
            buf = nxt
            buf_vec = vecs[i]

    if buf.strip():
        out.append(buf.strip())

    return out


# -----------------------------
# Packing paragraphs into request-sized chunks
# -----------------------------
def pack_segments_into_chunks(segments: List[str], max_chars: int) -> List[str]:
    chunks: List[str] = []
    buf: List[str] = []
    buf_len = 0

    def flush():
        nonlocal buf, buf_len
        if buf:
            chunks.append("\n\n".join(buf).strip())
        buf = []
        buf_len = 0

    for seg in segments:
        s = (seg or "").strip()
        if not s:
            continue

        if len(s) > max_chars:
            flush()
            # for very long segments, overflow chunk limit
            buf.append(s)
            flush()
            continue

        add_len = len(s) + (2 if buf else 0)
        if buf_len + add_len <= max_chars:
            buf.append(s)
            buf_len += add_len
        else:
            flush()
            buf.append(s)
            buf_len = len(s)

    flush()
    return chunks


# -----------------------------
# Parallel per-chunk LLM stage
# -----------------------------
def _run_one_chunk(
    *,
    llm: LLMClient,
    system_prompt: str,
    input_path: str,
    stage_name: str,
    chunk_index: int,
    chunk_text: str,
    sleep_seconds: float,
) -> Tuple[int, str]:
    user_content = (
        f"FILE_PATH: {input_path}\n"
        f"STAGE: {stage_name}\n"
        f"CHUNK_INDEX: {chunk_index}\n"
        f"CONTENT START\n{chunk_text}\nCONTENT END"
    )
    out = llm.generate(system_prompt=system_prompt, user_content=user_content)
    if sleep_seconds:
        time.sleep(sleep_seconds)
    return chunk_index, out

def run_classification(
    *,
    llm: LLMClient,
    system_prompt: str,
    input_path: str,
    text: str,
    workers: int,
    max_statements_per_chunk: int = 10,
    sleep_seconds_between_calls: float,
) -> str:
    input_array = json.loads(text)
    chunks = []
    i = 0
    while i < len(input_array):
        if i+max_statements_per_chunk < len(input_array):
            chunks.append(input_array[i:i+max_statements_per_chunk])
        else:
            chunks.append(input_array[i:-1])
        i += max_statements_per_chunk
    results: List[Optional[str]] = [None] * len(chunks)

    with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        futures = [
            ex.submit(
                _run_one_chunk,
                llm=llm,
                system_prompt=system_prompt,
                input_path=input_path,
                stage_name="classification",
                chunk_index=i,
                chunk_text=json.dumps(ch),
                sleep_seconds=sleep_seconds_between_calls,
            )
            for i, ch in enumerate(chunks)
        ]
        for fut in as_completed(futures):
            i, out = fut.result()
            results[i] = out

    output_array = []
    for i, t in enumerate(results):
        t = strip_markdown_fences(t)
        output_array.extend(json.loads(t))

    return json.dumps(output_array)


def run_parsing(
    *,
    llm: LLMClient,
    system_prompt: str,
    input_path: str,
    text: str,
    workers: int,
    max_chars_per_chunk: int,
    semantic: bool,
    embed_model: str,
    similarity_threshold: float,
    min_segment_chars: int,
    max_segment_chars: int,
    sleep_seconds_between_calls: float,
) -> str:
    embedder = llm if isinstance(llm, EmbeddingClient) else None

    segments = (
        semantic_segments(
            text=text,
            embedder=embedder,
            embed_model=embed_model,
            similarity_threshold=similarity_threshold,
            min_segment_chars=min_segment_chars,
            max_segment_chars=max_segment_chars,
        )
        if semantic
        else split_into_blocks(text)
    )

    # save chunks to check if it works well
    basename = os.path.basename(input_path)
    filename = basename.split(".")[0]
    dirname = os.path.dirname(input_path)
    segdir = os.path.join(dirname, "../../segmented-prompts")
    os.makedirs(segdir, exist_ok=True)
    save_text_file(os.path.join(segdir, f"{filename}_segments.txt"), "\n\nDIFF SEGMENT\n\n".join(segments))    

    chunks = pack_segments_into_chunks(segments, max_chars_per_chunk)
    results: List[Optional[str]] = [None] * len(chunks)

    with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        futures = [
            ex.submit(
                _run_one_chunk,
                llm=llm,
                system_prompt=system_prompt,
                input_path=input_path,
                stage_name="parsing",
                chunk_index=i,
                chunk_text=ch,
                sleep_seconds=sleep_seconds_between_calls,
            )
            for i, ch in enumerate(chunks)
        ]
        for fut in as_completed(futures):
            i, out = fut.result()
            results[i] = out

    for i, t in enumerate(results):
        if i == 0:
            place = "start"
        elif i == len(results) - 1:
            place = "end"
        else:
            place = "body"
        t = strip_json_array(t, place)
        results[i] = t

    merged = "\n".join(
        line
        for part in results
        if part
        for line in part.splitlines()
        if line.strip()
    )
    return merged


# -----------------------------
# Two-stage pipeline (sequential prompts, save in between)
# -----------------------------
def process_file_two_stage(
    *,
    path: str,
    root_dir: str,
    output_dir: str,
    llm: LLMClient,
    parsing_prompt: str,
    classification_prompt: str,
    stages: Tuple[str, ...],
    workers: int,
    max_chars_per_chunk: int,
    semantic: bool,
    embed_model: str,
    similarity_threshold: float,
    min_segment_chars: int,
    max_segment_chars: int,
    sleep_seconds_between_calls: float,
):
    ensure_output_dir(output_dir)

    parsing_out = output_path(root_dir, output_dir, path, stage_name="parsing")
    classification_out = output_path(root_dir, output_dir, path, stage_name="classification")

    if "parsing" in stages:
        raw = load_text_file(path)
        parsed = run_parsing(
            llm=llm,
            system_prompt=parsing_prompt,
            input_path=path,
            text=raw,
            workers=workers,
            max_chars_per_chunk=max_chars_per_chunk,
            semantic=semantic,
            embed_model=embed_model,
            similarity_threshold=similarity_threshold,
            min_segment_chars=min_segment_chars,
            max_segment_chars=max_segment_chars,
            sleep_seconds_between_calls=sleep_seconds_between_calls,
        )
        save_text_file(parsing_out, parsed)

    if "classification" in stages:
        if not os.path.isfile(parsing_out):
            raise FileNotFoundError(
                f"Parsing output not found: {parsing_out}. Run parsing first or include 'parsing' in --stages."
            )
        parsed_text = load_text_file(parsing_out)
        classified = run_classification(
            llm=llm,
            system_prompt=classification_prompt,
            input_path=parsing_out,
            text=parsed_text,
            workers=workers,
            sleep_seconds_between_calls=sleep_seconds_between_calls,
        )
        save_text_file(classification_out, classified)


# -----------------------------
# CLI
# -----------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Two-stage (parsing -> classification) LLM pipeline.")
    p.add_argument("--root-dir", required=True, help="Root directory containing input files.")
    p.add_argument("--output-dir-suffix", required=True, help="Output dir suffix.")
    p.add_argument("--model", default=DEFAULT_MODEL, help="Model name (gemini-* or OpenAI).")
    p.add_argument("--provider", choices=["auto", "openai", "google"], default="auto")

    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--max-output-tokens", type=int, default=None)

    p.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    p.add_argument("--max-chars-per-chunk", type=int, default=DEFAULT_MAX_CHARS_PER_CHUNK)
    p.add_argument("--sleep", type=float, default=DEFAULT_SLEEP_SECONDS)

    p.add_argument(
        "--stages",
        nargs="+",
        choices=["parsing", "classification"],
        default=["parsing", "classification"],
    )

    # semantic segmentation controls
    p.add_argument(
        "--semantic",
        action="store_true",
        default=False,
        help="Enable semantic paragraph segmentation (uses embeddings when provider is google).",
    )
    p.add_argument("--embed-model", default=DEFAULT_EMBED_MODEL)
    p.add_argument("--semantic-threshold", type=float, default=DEFAULT_SEMANTIC_SIM_THRESHOLD)
    p.add_argument("--min-segment-chars", type=int, default=DEFAULT_MIN_SEGMENT_CHARS)
    p.add_argument("--max-segment-chars", type=int, default=DEFAULT_MAX_SEGMENT_CHARS)

    return p.parse_args()


def main():
    args = parse_args()

    provider = infer_provider(args.model) if args.provider == "auto" else args.provider

    cfg = LLMConfig(
        provider=provider,
        model=args.model,
        temperature=args.temperature,
        max_output_tokens=args.max_output_tokens,
        response_mime_type="application/json",
        sleep_seconds_between_calls=args.sleep,
    )
    llm = build_llm_client(cfg)

    parsing_prompt = load_system_prompt(SYSTEM_PROMPT_FILE_PARSING)
    classification_prompt = load_system_prompt(SYSTEM_PROMPT_FILE_CLASSIFICATION)

    root_dir = args.root_dir
    output_dir = os.path.join(
        os.path.dirname(root_dir),
        f"{os.path.basename(root_dir)}-{args.output_dir_suffix}",
    )
    ensure_output_dir(output_dir)

    files = list(iter_files(root_dir))
    print(
        f"Found {len(files)} files. provider={provider} model={args.model} "
        f"workers={args.workers} stages={args.stages} semantic={args.semantic}"
    )

    for idx, fp in enumerate(files, 1):
        print(f"[{idx}/{len(files)}] {fp}")
        try:
            process_file_two_stage(
                path=fp,
                root_dir=root_dir,
                output_dir=output_dir,
                llm=llm,
                parsing_prompt=parsing_prompt,
                classification_prompt=classification_prompt,
                stages=tuple(args.stages),
                workers=args.workers,
                max_chars_per_chunk=args.max_chars_per_chunk,
                semantic=args.semantic,
                embed_model=args.embed_model,
                similarity_threshold=args.semantic_threshold,
                min_segment_chars=args.min_segment_chars,
                max_segment_chars=args.max_segment_chars,
                sleep_seconds_between_calls=args.sleep,
            )
        except Exception as e:
            print(f"ERROR: {fp}: {e}")


if __name__ == "__main__":
    main()