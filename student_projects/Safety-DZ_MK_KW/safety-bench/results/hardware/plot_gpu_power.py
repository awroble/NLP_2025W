from __future__ import annotations

from pathlib import Path
import re

import matplotlib.pyplot as plt

DATA_DIR = Path(__file__).resolve().parent

# Values read from the PNG headers (POW xx / 160 W).
POWER_W = {
    "gemma3-1b": 98,
    "gemma3-4b": 116,
    "llama3.2-11b": 36,
    "mistral-7b": 126,
    "qwen2.5-7b": 125,
    "qwen3-vl-2b": 115,
}


def model_id_from_stem(stem: str) -> str:
    if stem.endswith("-gpu"):
        return stem[:-4]
    return stem


def params_in_billion(model_id: str) -> float:
    matches = re.findall(r"(\d+(?:\.\d+)?)b", model_id)
    if not matches:
        raise ValueError(f"Could not parse params from '{model_id}'")
    return float(matches[-1])


def main() -> None:
    rows = []
    for path in sorted(DATA_DIR.glob("*-gpu.png")):
        model_id = model_id_from_stem(path.stem)
        if model_id not in POWER_W:
            raise KeyError(f"Missing power value for '{model_id}'")
        rows.append(
            {
                "model": model_id,
                "params_b": params_in_billion(model_id),
                "power_w": POWER_W[model_id],
            }
        )

    rows.sort(key=lambda r: (r["params_b"], r["model"]))

    labels = [f"{r['model']}\n({r['params_b']}B)" for r in rows]
    values = [r["power_w"] for r in rows]

    plt.figure(figsize=(7, 5))
    bars = plt.bar(labels, values, color="#4C78A8")
    plt.ylabel("GPU Power (W)")
    plt.title("GPU Power by Model Size")
    plt.xticks(rotation=20, ha="right")
    plt.grid(axis="y", linestyle="--", alpha=0.4)

    for bar, val in zip(bars, values, strict=True):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1,
            f"{val}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    plt.tight_layout()
    out_path = DATA_DIR / "gpu_power_by_params.png"
    plt.savefig(out_path, dpi=150)
    plt.show()


if __name__ == "__main__":
    main()
