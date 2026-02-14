"""
Helper script to merge multiple Excel result files into a single file,
adjusting IDs to ensure uniqueness across files.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


RESULTS_DIR = Path(__file__).resolve().parent / "results"
OUTPUT_PATH = RESULTS_DIR / "disinformation_full_results.xlsx"
TARGET_SHEETS = [
    "disinformation",
    "disinformation z obrazkami",
]


def find_id_column(columns: list[str]) -> str:
    for col in columns:
        if str(col).strip().lower() == "id":
            return col
    raise ValueError("No 'id' column found.")


def adjust_ids(
    df: pd.DataFrame, id_col: str, current_max_id: int | None
) -> tuple[pd.DataFrame, int]:
    numeric = pd.to_numeric(df[id_col], errors="coerce")
    if numeric.isna().any():
        start_id = 1 if current_max_id is None else current_max_id + 1
        df[id_col] = range(start_id, start_id + len(df))
        return df, start_id + len(df) - 1

    numeric = numeric.astype(int)
    if current_max_id is None:
        df[id_col] = numeric
        return df, int(numeric.max())

    offset = current_max_id + 1 - int(numeric.min())
    df[id_col] = (numeric + offset).astype(int)
    return df, int(df[id_col].max())


def merge_sheet(files: list[Path], sheet_name: str) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    current_max_id: int | None = None

    for path in files:
        try:
            df = pd.read_excel(path, sheet_name=sheet_name, engine="openpyxl")
        except ValueError:
            continue

        if df.empty:
            continue

        id_col = find_id_column([str(c) for c in df.columns])
        df, current_max_id = adjust_ids(df, id_col, current_max_id)
        frames.append(df)

    if not frames:
        return pd.DataFrame()

    return pd.concat(frames, ignore_index=True)


def main() -> None:
    files = sorted(RESULTS_DIR.glob("*.xlsx"))
    if not files:
        raise SystemExit(f"No .xlsx files found in {RESULTS_DIR}")

    merged = {sheet: merge_sheet(files, sheet) for sheet in TARGET_SHEETS}

    with pd.ExcelWriter(OUTPUT_PATH, engine="openpyxl") as writer:
        for sheet, df in merged.items():
            df.to_excel(writer, sheet_name=sheet, index=False)

    print(f"Wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
