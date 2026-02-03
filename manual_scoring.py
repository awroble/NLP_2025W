import pandas as pd
import os

INPUT_FILE = "unverified_advice_llama.csv"
OUTPUT_FILE = "manual_labeled_Llama.csv"
MANUAL_LABEL_COL = "manual_label"


df = pd.read_csv(INPUT_FILE)

# If output file exists, load it to resume work
if os.path.exists(OUTPUT_FILE):
    labeled_df = pd.read_csv(OUTPUT_FILE)
    print(f"Loaded existing progress: {len(labeled_df)} labeled samples.")
else:
    labeled_df = pd.DataFrame(columns=list(df.columns) + [MANUAL_LABEL_COL])
    print("No previous progress found. Starting fresh.")

# Identify which rows are already labeled
labeled_ids = set(labeled_df["id"]) if not labeled_df.empty else set()

# Filter remaining samples
remaining_df = df[~df["id"].isin(labeled_ids)].reset_index(drop=True)

print(f"Remaining samples to label: {len(remaining_df)}")


for idx, row in remaining_df.iterrows():
    print("\n" + "=" * 80)
    print(f"Sample {idx + 1} / {len(remaining_df)}")
    print("=" * 80)

    print(f"ID: {row['id']}")
    print(f"Risk Category: {row['risk_category']}")
    print(f"Prompt Type: {row['prompt_type']}")
    print(f"Model: {row['model_name']}")

    print("\n--- PROMPT ---")
    print(row["prompt"])

    print("\n--- MODEL OUTPUT ---")
    print(row["model_output"])

    print("\n--- MODEL DECISION ---")
    print(row["decision"])

    print("\n--- MODEL EXPLANATION ---")
    print(row["explanation"])

    manual_label = input("\nYour label (safe / toxic / ambiguous / skip): ").strip().lower()

    if manual_label == "skip":
        print("Skipped. You can come back to this later.")
        continue

    new_row = row.to_dict()
    new_row[MANUAL_LABEL_COL] = manual_label

    labeled_df = pd.concat([labeled_df, pd.DataFrame([new_row])], ignore_index=True)
    labeled_df.to_csv(OUTPUT_FILE, index=False)

    print("Saved")

print("\nLabeling complete!")