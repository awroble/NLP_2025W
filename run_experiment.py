import numpy as np
from sklearn.metrics import accuracy_score, f1_score


def run_experiment(
    X_train,
    X_test,
    y_train,
    y_test,
    vectorizer,
    model,
    n_runs=1
):
    acc_scores = []
    f1_scores = []

    for run in range(n_runs):
        print(f"--- Run {run + 1}/{n_runs} ---")
        # TF-IDF
        X_train_vec = vectorizer.fit_transform(X_train)
        X_test_vec = vectorizer.transform(X_test)

        # Train
        model.fit(X_train_vec, y_train)
        print("Model trained.")

        # Predict
        y_pred = model.predict(X_test_vec)
        print("Prediction done.")

        # Metrics
        acc_scores.append(accuracy_score(y_test, y_pred))
        f1_scores.append(f1_score(y_test, y_pred))

    return {
        "accuracy_mean": np.mean(acc_scores),
        "accuracy_std": np.std(acc_scores),
        "f1_mean": np.mean(f1_scores),
        "f1_std": np.std(f1_scores)
    }
