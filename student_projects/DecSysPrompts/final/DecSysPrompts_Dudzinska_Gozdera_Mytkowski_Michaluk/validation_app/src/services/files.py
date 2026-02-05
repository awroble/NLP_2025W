from pathlib import Path
from typing import Optional

from services.loader import load_json, load_text


STATEMENTS_ROOT = "../CL4R1T4S-no-code-processed-gemini-flash-3_5/classification"
TXT_ROOT = "../CL4R1T4S"


def list_statement_files(base_dir: str = STATEMENTS_ROOT) -> list[Path]:
    """Recursively list all .json files under base_dir."""
    base = Path(base_dir)
    return sorted(base.rglob("*.json"))


def get_candidate_text_path(rel: str, txt_root: str = TXT_ROOT) -> Optional[Path]:
    """Given a relative stem (without .json), find a matching text/markdown file."""
    candidate_txt = Path(txt_root) / f"{rel}.txt"
    if not candidate_txt.exists():
        candidate_txt = Path(txt_root) / f"{rel}.md"
    if not candidate_txt.exists():
        candidate_txt = Path(txt_root) / f"{rel}.mkd"
    if not candidate_txt.exists():
        candidate_txt = Path(txt_root) / f"{rel}"
    return candidate_txt if candidate_txt.exists() else None


def load_statements_and_context(json_file_path: str) -> tuple[list[dict], str]:
    """Load statements JSON and optional raw text context."""
    json_path = Path(json_file_path)
    rel = str(json_path).replace(".json", "").replace(f"{STATEMENTS_ROOT}/", "")

    json_data = load_json(str(json_path))

    candidate_txt = get_candidate_text_path(rel)
    raw_context = load_text(str(candidate_txt)) if candidate_txt else ""

    return json_data, raw_context