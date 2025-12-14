from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC


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
