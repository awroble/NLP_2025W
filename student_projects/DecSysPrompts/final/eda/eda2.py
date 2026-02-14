import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer

DATASET_ROOT = "../CL4R1T4S"
SNS_THEME = "whitegrid"
OUTPUT_DIR = "eda/eda_plots"

os.makedirs(OUTPUT_DIR, exist_ok=True)

sns.set_theme(style=SNS_THEME)
plt.rcParams["figure.figsize"] = (6, 6)


def load_data(root_path):
    data = []

    for current_path, folders, files in os.walk(root_path):
        folders[:] = [d for d in folders if not d.startswith(".")]
        rel_path = os.path.relpath(current_path, root_path)

        if rel_path == ".":
            continue

        # Organization name extraction: CL4R1T4S/OpenAI/GPT-4 -> Org: OpenAI
        organization = rel_path.split(os.sep)[0]

        for file in files:
            if file.startswith("."):
                continue

            file_path = os.path.join(current_path, file)

            try:
                with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                    content = f.read()

                file_stats = os.stat(file_path)

                data.append(
                    {
                        "organization": organization,
                        "file_path": file_path,
                        "file_name": file,
                        "file_size_kb": file_stats.st_size / 1024,
                        "content": content,
                        "char_count": len(content),
                        "word_count": len(content.split()),
                    }
                )
            except Exception as e:
                print(f"Error reading {file_path}: {e}")

    df = pd.DataFrame(data)
    print(
        f"Successfully loaded {len(df)} files from {df['organization'].nunique()} organizations."
    )
    return df


def plot_file_distribution(df):
    plt.figure(figsize=(6, 6))
    order = df["organization"].value_counts().index
    sns.countplot(y="organization", data=df, order=order, palette="viridis")
    plt.xlabel("Count")
    plt.ylabel("Organization")
    plt.tight_layout()
    plt.savefig(
        os.path.join(OUTPUT_DIR, "file_distribution.png"), dpi=300, bbox_inches="tight"
    )
    plt.show()


def plot_length_distribution(df):
    top_orgs = df["organization"].value_counts().nlargest(15).index
    subset = df[df["organization"].isin(top_orgs)]

    plt.figure(figsize=(6, 6))
    sns.boxplot(x="word_count", y="organization", data=subset, palette="coolwarm")
    plt.xlabel("Word Count")
    plt.ylabel("Organization")
    plt.xscale("log")
    plt.tight_layout()
    plt.savefig(
        os.path.join(OUTPUT_DIR, "length_distribution.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.show()


def plot_file_sizes(df):
    plt.figure(figsize=(6, 6))
    sns.histplot(df["file_size_kb"], bins=30, kde=True, color="teal")
    plt.title("Distribution of File Sizes (KB)", fontsize=16)
    plt.xlabel("Size (KB)")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(
        os.path.join(OUTPUT_DIR, "file_sizes.png"), dpi=300, bbox_inches="tight"
    )
    plt.show()


def plot_wordcloud(df):
    text = " ".join(df["content"].astype(str))

    stopwords = set(["system", "prompt", "user", "assistant", "model", "response"])

    wc = WordCloud(
        width=600, height=600, background_color="white", stopwords=stopwords
    ).generate(text)

    plt.figure(figsize=(6, 6))
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.title("Most Frequent Words in System Prompts (Global)", fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "wordcloud.png"), dpi=300, bbox_inches="tight")
    plt.show()


def plot_top_ngrams(df, n=2, top_k=15):
    vec = CountVectorizer(ngram_range=(n, n), stop_words="english")
    bag_of_words = vec.fit_transform(df["content"])
    sum_words = bag_of_words.sum(axis=0)
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)[:top_k]

    common_words_df = pd.DataFrame(words_freq, columns=["ngram", "count"])

    plt.figure(figsize=(6, 6))
    sns.barplot(x="count", y="ngram", data=common_words_df, palette="magma")
    plt.title(f"Top {top_k} Most Common {n}-grams", fontsize=16)
    plt.xlabel("Frequency")
    plt.ylabel(f"{n}-gram")
    plt.tight_layout()
    plt.savefig(
        os.path.join(OUTPUT_DIR, f"top_{n}grams.png"), dpi=300, bbox_inches="tight"
    )
    plt.show()


if __name__ == "__main__":
    if not os.path.exists(DATASET_ROOT):
        print(f"Error: Dataset root '{DATASET_ROOT}' not found.")
    else:
        df = load_data(DATASET_ROOT)

        if not df.empty:
            print("\nDataset Info:")
            print(df.info())
            print(f"\nTotal Dataset Size: {df['file_size_kb'].sum() / 1024:.2f} MB")

            plot_file_distribution(df)
            plot_file_sizes(df)
            plot_length_distribution(df)
            plot_wordcloud(df)
            plot_top_ngrams(df, n=2, top_k=20)  # Bi-grams
        else:
            print("No valid files found. Check directory structure.")
