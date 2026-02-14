import pandas as pd
from pathlib import Path


def convert_to_csv(prompts_path):
    prompts_path = Path(prompts_path)
    df = pd.read_excel(prompts_path)
    if "Prompt" in df.columns:
        df["Prompt"] = df["Prompt"].str.replace("\n", " ").str.strip()
    df.to_csv(prompts_path.with_suffix(".csv"), index=False)


if __name__ == "__main__":
    convert_to_csv(
        "prompts/vscbench/vscbench_image_centric_images/vscbench_image_centric.xlsx"
    )

    csv_path = Path(
        "prompts/vscbench/vscbench_image_centric_images/vscbench_image_centric.csv"
    )
    df = pd.read_csv(csv_path)

    categories = ["discrimination", "drugs", "illegal_activity", "religion", "violence"]

    filtered_df = pd.concat([df[df["Category"] == cat].head(8) for cat in categories])

    output_path = csv_path.parent / "vscbench_subset.csv"
    filtered_df.to_csv(output_path, index=False)

    print(f"Extracted {len(filtered_df)} prompts")
    print(f"Saved to: {output_path}")
    print(f"\nPrompts per category:")
    print(filtered_df["Category"].value_counts())
