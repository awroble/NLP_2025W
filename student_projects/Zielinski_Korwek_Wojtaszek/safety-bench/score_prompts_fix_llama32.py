"""
Helper script to fix rows with ollama_error in llama3.2:11b disinformation results.
Good idea is to increase timeout to even 1440s for those problematic rows.
It can be set back later to default 180s for text-only prompts.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from score_prompts import (
    Config,
    build_openai_client,
    logger,
    process_disinfo_or_offensive_sheet,
)


RESULTS_DIR = Path(__file__).resolve().parent  # / "results"
DEFAULT_INPUT = RESULTS_DIR / "scored_offensive_obrazki.xlsx"
DEFAULT_OUTPUT = RESULTS_DIR / "fix_scored_offensive_obrazki.xlsx"
SHEET_NAME = "offensive z obrazkami"


def pick_reason_column(df: pd.DataFrame) -> str:
    if "judge_reason" in df.columns:
        return "judge_reason"
    if "judge_reson" in df.columns:
        return "judge_reson"
    raise SystemExit("Missing judge_reason column in input sheet.")


def main() -> int:
    load_dotenv()
    parser = argparse.ArgumentParser(
        description="Rerun rows with ollama_error for llama32 disinformation z obrazkami."
    )
    parser.add_argument(
        "--input", default=str(DEFAULT_INPUT), help="Path to input .xlsx"
    )
    parser.add_argument(
        "--output", default=str(DEFAULT_OUTPUT), help="Path to output .xlsx"
    )
    parser.add_argument(
        "--ollama-host", default=os.getenv("OLLAMA_HOST", "http://localhost:11434")
    )
    parser.add_argument(
        "--ollama-model", default=os.getenv("OLLAMA_MODEL", "qwen3:0.6b")
    )
    parser.add_argument(
        "--judge-model", default=os.getenv("OPENAI_JUDGE_MODEL", "gpt-5-nano")
    )
    parser.add_argument(
        "--image-base-dir",
        default=os.getenv(
            "IMAGE_BASE_DIR",
            str(Path(__file__).resolve().parent / "images"),
        ),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process only first N filtered rows (debug).",
    )
    parser.add_argument(
        "--skip-openai", action="store_true", help="Skip OpenAI scoring (debug)."
    )
    parser.add_argument(
        "--reason-substring",
        default="ollama_error",
        help="Substring to match in judge_reason.",
    )
    args = parser.parse_args()

    cfg = Config(
        input_xlsx=args.input,
        output_xlsx=args.output,
        ollama_host=args.ollama_host,
        ollama_model=args.ollama_model,
        openai_judge_model=args.judge_model,
        image_base_dir=args.image_base_dir,
        limit=args.limit,
        skip_openai=args.skip_openai,
        subset=None,
    )

    client = None
    if not cfg.skip_openai:
        api_key = os.getenv("OPENAI_API_KEY", "").strip()
        if not api_key:
            logger.warning(
                "OPENAI_API_KEY not found in environment/.env; OpenAI scoring will be skipped."
            )
            cfg.skip_openai = True
        else:
            try:
                client = build_openai_client()
            except Exception as e:
                logger.exception("Failed to init OpenAI client: %s", e)
                cfg.skip_openai = True
                client = None

    df = pd.read_excel(cfg.input_xlsx, sheet_name=SHEET_NAME, engine="openpyxl")
    # reason_col contains reason_substring or actual_response is empty
    reason_col = pick_reason_column(df)
    mask = (
        df[reason_col]
        .astype(str)
        .str.contains(args.reason_substring, case=False, na=False)
        | df["actual_response"].isna()
        | df["actual_response"].astype(str).str.strip().eq("")
    )
    df_filtered = df.loc[mask].copy()
    logger.info(
        "Filtered %d rows from %d where %s contains '%s'.",
        len(df_filtered),
        len(df),
        reason_col,
        args.reason_substring,
    )

    if df_filtered.empty:
        logger.warning("No rows matched; writing empty sheet to %s", cfg.output_xlsx)
        out_df = df_filtered
    else:
        out_df = process_disinfo_or_offensive_sheet(
            df_filtered, cfg, client, SHEET_NAME
        )

    with pd.ExcelWriter(cfg.output_xlsx, engine="openpyxl") as writer:
        out_df.to_excel(writer, sheet_name=SHEET_NAME, index=False)
    logger.info("Done. Wrote: %s", cfg.output_xlsx)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
