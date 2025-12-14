import pandas as pd

from preprocessing import train_test_split_data, get_vectorizer
from models import get_model
from run_experiment import run_experiment

# Load data
df = pd.read_pickle("./data/combined_df.pkl")
df["content_text_joined"] = df["content_words"].apply(lambda x: " ".join(x))


TEXT_COL = "content_text_joined"    
LABEL_COL = "is_spoiler"      

# Split
X_train, X_test, y_train, y_test = train_test_split_data(
    df,
    TEXT_COL,
    LABEL_COL
)

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
