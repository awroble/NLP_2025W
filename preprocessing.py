from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer


def train_test_split_data(
    df,
    text_col,
    label_col,
    test_size=0.2,
    random_state=42
):
    X = df[text_col]
    y = df[label_col]

    return train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )


def get_vectorizer(method="tfidf", **kwargs):
    if method == "tfidf":
        return TfidfVectorizer(
            max_features=kwargs.get("max_features", 10000),
            ngram_range=kwargs.get("ngram_range", (1, 2)),
            min_df=kwargs.get("min_df", 2),
            stop_words="english"
        )
    else:
        raise ValueError(f"Unknown vectorizer method: {method}")
