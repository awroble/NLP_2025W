from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer


def get_model(model_name="logreg", **kwargs):
    if model_name == "logreg":
        return LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=42
        )

    elif model_name == "svm":
        return LinearSVC(
            class_weight="balanced",
            random_state=42
        )

    else:
        raise ValueError(f"Unknown model: {model_name}")


def get_vectorizer(method="tfidf", **kwargs):
    if method == "tfidf":
        return TfidfVectorizer(
            max_features=kwargs.get("max_features", 10000),
            ngram_range=kwargs.get("ngram_range", (1, 2)),
            min_df=kwargs.get("min_df", 2),
            stop_words="english"
        )
    if method == "bow":
        return CountVectorizer(
            max_features=kwargs.get("max_features", 10000),
            ngram_range=kwargs.get("ngram_range", (1, 2)),
            min_df=kwargs.get("min_df", 2),
            stop_words="english"
        )
    else:
        raise ValueError(f"Unknown vectorizer method: {method}")
