import pandas as pd
import glob
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sympy import plot_implicit
import csv
from io import StringIO
from typing import List

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

def load_mixed_csv_to_df(
    path: str,
    columns: List[str],
    encoding: str = "utf-8"
) -> pd.DataFrame:
    """
    Load a CSV file where some rows are normal CSV
    and some rows are a full CSV row wrapped in quotes.

    Args:
        path: path to csv file
        columns: list of column names
        encoding: file encoding

    Returns:
        pandas DataFrame
    """

    rows = []

    with open(path, newline="", encoding=encoding) as f:
        reader = csv.reader(f)
        next(reader)

        for i, row in enumerate(reader, start=1):
            try:

                if len(row) == len(columns):
                    rows.append(row)
                    continue


                if len(row) == 1:
                    parsed = next(csv.reader([row[0]]))
                    if len(parsed) == len(columns):
                        rows.append(parsed)
                        continue

                # Fallback: try pandas parser
                parsed = pd.read_csv(StringIO(",".join(row)), header=None).iloc[0]
                if len(parsed) == len(columns):
                    rows.append(parsed.tolist())
                    continue

                raise ValueError(f"Invalid column count: {len(row)}")

            except Exception as e:
                print(f"[WARN] Skipping row {i}: {e}")

    return pd.DataFrame(rows, columns=columns)


def generate_plots(df):
    '''
    Function to generate plots for prompt data
    :param df: with columns [id;data_type;risk_category;model_name;
                            prompt;model_output;label;score;decision;
                            manual_label;prompt_type]
    :return: 4 Plots
    '''
    (df
     .groupby(['model_name', 'manual_label'])
     .size()
     .unstack(fill_value=0)
     .plot(kind='bar')
    )

    plt.title("Model prompt safety by model type")
    plt.xlabel("model type")
    plt.ylabel("Number of prompts")
    plt.tight_layout()
    plt.show()

    (df
     .groupby(['prompt_type', 'manual_label'])
     .size()
     .unstack(fill_value=0)
     .plot(kind='bar')
    )

    plt.title("Model prompt safety by prompt type")
    plt.xlabel("prompt type")
    plt.ylabel("Number of prompts")
    plt.tight_layout()
    plt.show()


    for model in df['model_name'].unique():
        (
            df[df['model_name'] == model]
            .groupby(['prompt_type', 'manual_label'])
            .size()
            .unstack(fill_value=0)
            .plot(kind='bar')
        )

        plt.title(f"Model prompt safety by prompt type | {model}")
        plt.xlabel("prompt type")
        plt.ylabel("Number of prompts")
        plt.tight_layout()
        plt.show()

def plot_model_diff(df):

    df['match'] = df['manual_label'] == df['model_label']
    summary = df['match'].value_counts(normalize=True).rename({True: 'Consistent', False: 'Different'})
    sns.barplot(x=summary.index, y=summary.values, palette='Set2')

    for i, value in enumerate(summary.values):
        plt.text(i, value + 0.01, f'{value*100:.1f}%', ha='center', va='bottom', fontsize=12)


    plt.yticks([])

    plt.ylabel('')
    plt.title('Comparison of manual vs model label')
    plt.show()

columns = [
    "idx",
    "id",
    "data_type",
    "risk_category",
    "model_name",
    "prompt",
    "model_output",
    "label",
    "score",
    "decision",
    "manual_label",
    "prompt_type",
    "model_label"
]
df = load_mixed_csv_to_df('Data/Mental_physical_health_prompts.csv', columns)
plot_model_diff(df)
print(df.head())