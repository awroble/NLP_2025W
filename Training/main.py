import pandas as pd


# Load data

# Choose components
vectorizer = get_vectorizer(
    method="tfidf",
    max_features=15000,
    ngram_range=(1, 2)
)

model = get_model("logreg")   # albo "svm"

# Run PoC
results = run_experiment(
    X_train,
    X_test,
    y_train,
    y_test,
    vectorizer,
    model,
    n_runs=3
)

print("PoC results:")
for k, v in results.items():
    print(f"{k}: {v:.4f}")
