import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

BASE = Path(__file__).resolve().parent.parent.parent


########### Loading ######################################################################

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def flatten_entry(entry):
    prompt_text = " ".join([p["text"] for p in entry.get("prompt", [])])
    assistant_response = next((t["content"] for t in entry.get("conversation", []) if t["role"] == "assistant"), "")
    meta = entry.get("metadata", {})
    slots = meta.get("slots_filled", {})

    result = {
        "id": entry.get("id"),
        "prompt_text": prompt_text,
        "assistant_response": assistant_response,
        "verdict": int(entry.get("judge_verdict", 0)),
        **slots,
        **meta
    }
    return result


def load_category_df(category):
    path = os.path.join(BASE, "data", "eval", f"{category}-all-eval.json")
    data = load_json(path)
    return pd.DataFrame([flatten_entry(e) for e in data])


############## Combine results ########################################################

def combine_modalities():
    for category in ["sensitive-data-extraction", "social-engineering", "mental-physical-health"]:
        data = []
        for type in ["multiturn", "multimodal", "single"]:
            data.extend(load_json(os.path.join(BASE, "data", "eval", f"{category}-{type}-eval.json")))
        with open(os.path.join(BASE, "data", "eval", f"{category}-all-eval.json"), "w", encoding="utf-8") as f:
            f.write(json.dumps(data, indent=2, ensure_ascii=False))


def combine_all():
    data = []
    for category in ["sensitive-data-extraction", "social-engineering", "mental-physical-health"]:
        data.extend(load_json(os.path.join(BASE, "data", "eval", f"{category}-all-eval.json")))
    with open(os.path.join(BASE, "data", "eval", f"all-eval.json"), "w", encoding="utf-8") as f:
        f.write(json.dumps(data, indent=2, ensure_ascii=False))


############## Plots ###################################################################

def set_plot_style():
    sns.set_theme(style="whitegrid", context="talk", font_scale=1.2)
    plt.rcParams.update({
        "axes.titlesize": 18,
        "axes.labelsize": 14,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "figure.figsize": (10, 6)
    })


def show_boxplot(df, x, y, title):
    unique_vals = df[x].dropna().nunique()
    plt.figure()
    sns.boxplot(data=df, x=x, y=y, palette=sns.color_palette("rocket", n_colors=unique_vals), hue=x, linewidth=1.5,
                fliersize=3)
    plt.title(title, weight="bold", pad=20)
    sns.despine()
    plt.tight_layout()
    plt.show()


def show_barplot(series, title):
    df_plot = series.reset_index()
    df_plot.columns = ["category", "value"]
    unique_vals = df_plot["category"].nunique()

    plt.figure()
    sns.barplot(
        data=df_plot,
        x="category",
        y="value",
        hue="category",
        palette=sns.color_palette("rocket", n_colors=unique_vals),
        linewidth=1.5
    )
    plt.legend([], [], frameon=False)
    plt.xticks(rotation=45)
    plt.title(title, weight="bold", pad=20)
    plt.tight_layout()
    plt.show()


########### Metrics ############################################################

def pass_rate(df, col):
    return df.groupby(col)["verdict"].mean().sort_values(ascending=False)


def compute_linguistic_features(df):
    df["prompt_length_chars"] = df["prompt_text"].str.len()
    df["prompt_length_tokens"] = df["prompt_text"].str.split().str.len()
    df["contains_please"] = df["prompt_text"].str.contains(r"\bplease\b", case=False)
    df["contains_urgent"] = df["prompt_text"].str.contains(r"\burgent|immediately|asap|now\b", case=False)
    df["contains_emotional"] = df["prompt_text"].str.contains(r"smartest|trust|only you|help me|helpful", case=False)
    df["response_length_tokens"] = df["assistant_response"].str.split().str.len()
    df["explicit_refusal"] = df["assistant_response"].str.contains(
        r"\b(cannot assist|cannot help|not able|sorry|cannot provide)\b", case=False)
    df["gives_example"] = df["assistant_response"].str.contains(r"(example|script|template|scenario)", case=False)
    return df


######### Analysis ######################################################

def analyze_category(category):
    print(f"\n\n########## {category.upper()} ##########")
    df = load_category_df(category)
    df = compute_linguistic_features(df)

    print("Total entries:", len(df))
    print("Verdict distribution:\n", df["verdict"].value_counts())
    print("Feature coverage:\n", (len(df) - df.isna().sum()) / len(df))

    sns.countplot(data=df, x="verdict", palette=sns.color_palette("rocket", n_colors=2), hue='verdict')
    plt.title(f"Verdict Distribution: {category}")
    plt.show()

    modality_rate = pass_rate(df, "modality")
    print("Pass rate by modality:\n", modality_rate)
    show_barplot(modality_rate, f"Pass Rate by Modality: {category}")

    show_boxplot(df, "verdict", "prompt_length_tokens", f"Prompt Length (tokens) vs Verdict: {category}")
    show_boxplot(df, "verdict", "prompt_length_chars", f"Prompt Length (chars) vs Verdict: {category}")
    show_boxplot(df, "verdict", "response_length_tokens", f"Response Length vs Verdict: {category}")

    for marker in ["contains_please", "contains_urgent", "contains_emotional", "explicit_refusal", "gives_example"]:
        print(f"\nPass rate by {marker}:\n", df.groupby(marker)["verdict"].mean())

    template_rate = pass_rate(df, "template_id")
    print(template_rate)
    best = template_rate.head(10)
    worst = template_rate.tail(10)
    show_barplot(best, f"Top Templates by Pass Rate: {category}")
    show_barplot(worst, f"Lowest Templates by Pass Rate: {category}")


def analyze_all():
    print(f"\n\n########## ALL ##########")
    path = os.path.join(BASE, "data", "eval", "all-eval.json")
    data = load_json(path)
    df = pd.DataFrame([flatten_entry(e) for e in data])
    df = compute_linguistic_features(df)

    print("Total entries:", len(df))
    print("Verdict distribution:\n", df["verdict"].value_counts())
    print("Feature coverage:\n", (len(df) - df.isna().sum()) / len(df))

    sns.countplot(data=df, x="verdict", palette=sns.color_palette("rocket", n_colors=2), hue='verdict')
    plt.title(f"Verdict Distribution: all-eval")
    plt.show()

    modality_rate = pass_rate(df, "modality")
    print("Pass rate by modality:\n", modality_rate)
    show_barplot(modality_rate, f"Pass Rate by Modality: all-eval")

    category_rate = pass_rate(df, "category")
    print("Pass rate by category:\n", category_rate)
    show_barplot(category_rate, f"Pass Rate by Category: all-eval")

    show_boxplot(df, "verdict", "prompt_length_tokens", f"Prompt Length (tokens) vs Verdict: all-eval")
    show_boxplot(df, "verdict", "prompt_length_chars", f"Prompt Length (chars) vs Verdict: all-eval")
    show_boxplot(df, "verdict", "response_length_tokens", f"Response Length vs Verdict: all-eval")

    for marker in ["contains_please", "contains_urgent", "contains_emotional", "explicit_refusal", "gives_example"]:
        print(f"\nPass rate by {marker}:\n", df.groupby(marker)["verdict"].mean())

    template_rate = pass_rate(df, "template_id")
    print(template_rate)
    best = template_rate.head(10)
    worst = template_rate.tail(10)
    show_barplot(best, f"Top Templates by Pass Rate: all-eval")
    show_barplot(worst, f"Lowest Templates by Pass Rate: all-eval")


########## MAIN #################################################################

def main():
    set_plot_style()

    for category in ["sensitive-data-extraction", "social-engineering", "mental-physical-health"]:
        analyze_category(category)

    analyze_all()


if __name__ == "__main__":
    # Only once after new results:
    # combine_modalities()
    # combine_all()

    # Always:
    main()
