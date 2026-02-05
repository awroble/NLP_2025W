from pathlib import Path
import re
from collections import Counter

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

from wordcloud import WordCloud

sns.set(style="whitegrid")

TOP_N_TERMS = 20
TFIDF_MAX_FEATURES = 5000
RANDOM_STATE = 42
PLOT_DIR = Path("eda/eda_plots")
SUMMARY_DIR = PLOT_DIR / "summaries"
PLOT_DIR.mkdir(parents=True, exist_ok=True)
SUMMARY_DIR.mkdir(parents=True, exist_ok=True)


def is_hidden(path: Path) -> bool:
    return any(part.startswith(".") for part in path.parts)


def load_files(root: Path, exts=None):
    files = []
    for p in root.rglob("*"):
        if p.is_dir():
            continue
        if p.parent.resolve() == root.resolve():
            continue
        if is_hidden(p.relative_to(root)):
            continue
        if exts and p.suffix.lower() not in exts:
            continue
        files.append(p)
    return sorted(files)


def read_text_file(path: Path) -> str:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        text = path.read_bytes().decode("latin1", errors="replace")
    return text


def simple_tokenize(text: str) -> list:
    text = text.lower()
    tokens = re.findall(r"[\w@#']+", text)
    return tokens


def build_metadata(files):
    rows = []
    for p in files:
        text = read_text_file(p)
        size = p.stat().st_size
        n_lines = text.count("\n") + 1
        tokens = simple_tokenize(text)
        n_tokens = len(tokens)
        parts = p.relative_to(ROOT).parts
        top_org = parts[0] if len(parts) >= 1 else ""
        model_path = "/".join(parts[1:]) if len(parts) >= 2 else ""
        rows.append(
            {
                "path": str(p),
                "relative_path": "/".join(parts),
                "top_org": top_org,
                "model_path": model_path,
                "size_bytes": size,
                "n_lines": n_lines,
                "n_tokens": n_tokens,
                "text": text,
            }
        )
    df = pd.DataFrame(rows)
    return df


def plot_basic_stats(df: pd.DataFrame):
    plt.figure(figsize=(6, 6))
    sns.histplot(df["size_bytes"] / 1024.0, bins=30)
    plt.xlabel("File size (KiB)")
    plt.title("Distribution of file sizes")
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "file_size_distribution.png")
    plt.show()

    plt.figure(figsize=(6, 6))
    sns.histplot(df["n_tokens"], bins=40)
    plt.xlabel("Number of tokens")
    plt.title("Distribution of token counts")
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "token_count_distribution.png")
    plt.show()

    # files per top org
    plt.figure(figsize=(6, 6))
    order = df["top_org"].value_counts().index
    sns.countplot(data=df, y="top_org", order=order)
    plt.title("Number of files per organization")
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "files_per_top_org.png")
    plt.show()

    counts = df["top_org"].value_counts()
    big_orgs = counts[counts >= 3].index.tolist()[
        :12
    ]  # show up to 12 orgs with >=3 files
    plt.figure(figsize=(6, 6))
    sns.boxplot(data=df[df["top_org"].isin(big_orgs)], x="n_tokens", y="top_org")
    plt.xlabel("Tokens")
    plt.title("Token distribution by top organizations")
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "tokens_by_org_boxplot.png")
    plt.show()


def compute_top_ngrams(texts, n=1, top_k=30, stop_words=ENGLISH_STOP_WORDS):
    counter = Counter()
    for t in texts:
        toks = [tok for tok in simple_tokenize(t) if tok not in stop_words]
        if n == 1:
            counter.update(toks)
        else:
            ngrams = zip(*[toks[i:] for i in range(n)])
            counter.update([" ".join(ng) for ng in ngrams])
    return counter.most_common(top_k)


def plot_top_ngrams(df):
    texts = df["text"].tolist()
    uni = compute_top_ngrams(texts, n=1, top_k=30)
    bi = compute_top_ngrams(texts, n=2, top_k=30)

    # unigrams
    uni_df = pd.DataFrame(uni, columns=["term", "count"]).iloc[:20]
    plt.figure(figsize=(6, 6))
    sns.barplot(data=uni_df, x="count", y="term")
    plt.title("Top unigrams in corpus")
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "top_unigrams.png")
    plt.show()

    # bigrams
    bi_df = pd.DataFrame(bi, columns=["term", "count"]).iloc[:20]
    plt.figure(figsize=(6, 6))
    sns.barplot(data=bi_df, x="count", y="term")
    plt.title("Top bigrams in corpus")
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "top_bigrams.png")
    plt.show()


def make_wordcloud(text, outpath, max_words=200):
    wc = WordCloud(
        width=1200, height=1200, background_color="white", max_words=max_words
    )
    wc.generate(text)
    plt.figure(figsize=(6, 6))
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(outpath)
    plt.show()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="EDA for CL4R1T4S dataset")
    ap.add_argument("--root", default="../CL4R1T4S", help="Path to CL4R1T4S root folder")
    args = ap.parse_args()

    ROOT = Path(args.root)
    if not ROOT.exists():
        raise SystemExit(f"Root path {ROOT} does not exist")

    files = load_files(ROOT, exts=None)  # accept all files
    print(f"Found {len(files)} files (after filtering)")

    df = build_metadata(files)
    df.to_csv(SUMMARY_DIR / "file_metadata.csv", index=False)

    print(df[["relative_path", "top_org", "size_bytes", "n_tokens"]].head())

    plot_basic_stats(df)
    plot_top_ngrams(df)

    corpus_text = "\n".join(df["text"].tolist())
    make_wordcloud(corpus_text, PLOT_DIR / "wordcloud_corpus.png")

    for org, group in df.groupby("top_org"):
        if len(group) < 2:
            continue
        make_wordcloud(
            "\n".join(group["text"].tolist()), PLOT_DIR / f"wordcloud_{org}.png"
        )

    print(f"Saved to {PLOT_DIR.resolve()}")
