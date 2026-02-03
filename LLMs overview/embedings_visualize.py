import os
import pickle
import warnings

import faiss
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

# ---------------------------------------------------------
# WARNINGS
# ---------------------------------------------------------
warnings.filterwarnings("ignore")

# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------
FAISS_INDEX_FILE = "faiss_index.bin"
CHUNKS_FILE = "chunks.pkl"

OUTPUT_IMAGE = "embeddings_cloud_visualization.png"

# ---- VISUAL & COMPUTE SETTINGS ----
MAX_POINTS_TO_PLOT = 80000        # try to render almost everything
PCA_COMPONENTS = 25
TSNE_PERPLEXITY = 35
TSNE_ITER = 750

POINT_SIZE = 18                  # visible but not too large
POINT_ALPHA = 0.35               # overlap creates cloud effect
POINT_COLOR = "#1f77b4"          # classic matplotlib blue

# ---------------------------------------------------------
# FUNCTIONS
# ---------------------------------------------------------

def safe_tsne_execution(data: np.ndarray) -> np.ndarray:
    """
    PCA + t-SNE with compatibility for old and new sklearn versions.
    """
    print(f"Running PCA reduction to {PCA_COMPONENTS} dimensions...")
    pca = PCA(n_components=PCA_COMPONENTS, random_state=42)
    data_pca = pca.fit_transform(data)

    print(f"Running t-SNE on {len(data_pca)} points...")

    try:
        tsne = TSNE(
            n_components=2,
            perplexity=TSNE_PERPLEXITY,
            max_iter=TSNE_ITER,
            init="pca",
            learning_rate="auto",
            random_state=42
        )
        return tsne.fit_transform(data_pca)

    except TypeError:
        print("[INFO] Old sklearn detected. Using legacy TSNE.")
        tsne = TSNE(
            n_components=2,
            perplexity=30,
            random_state=42
        )
        return tsne.fit_transform(data_pca)

    except MemoryError:
        print("[WARNING] t-SNE out of memory. Falling back to PCA-only.")
        return data_pca[:, :2]


def load_data_and_embeddings():
    """
    Load FAISS index and reconstruct embeddings.
    """
    print("Loading FAISS index and metadata...")

    if not (os.path.exists(FAISS_INDEX_FILE) and os.path.exists(CHUNKS_FILE)):
        raise FileNotFoundError("FAISS index or chunks file not found.")

    index = faiss.read_index(FAISS_INDEX_FILE)

    with open(CHUNKS_FILE, "rb") as f:
        doc_df = pickle.load(f)

    total_vectors = index.ntotal
    print(f"FAISS index contains {total_vectors} vectors.")

    print("Reconstructing vectors from FAISS...")
    embeddings = index.reconstruct_n(0, total_vectors)

    min_len = min(len(doc_df), len(embeddings))
    doc_df = doc_df.iloc[:min_len].reset_index(drop=True)
    embeddings = embeddings[:min_len].astype("float32")

    return embeddings, doc_df


def visualize_cloud(embeddings: np.ndarray):
    """
    Render a single-color semantic embedding cloud.
    """
    total_points = len(embeddings)
    print(f"Total available points: {total_points}")

    # ---- Sampling ----
    if total_points > MAX_POINTS_TO_PLOT:
        print(f"Sampling {MAX_POINTS_TO_PLOT} points...")
        idx = np.random.choice(total_points, MAX_POINTS_TO_PLOT, replace=False)
        embeddings = embeddings[idx]
    else:
        print("Using all available points.")

    # ---- Dimensionality reduction ----
    vis = safe_tsne_execution(embeddings)

    print(f"Rendering plot with {len(vis)} points...")

    # ---- Plot ----
    plt.figure(figsize=(26, 22))

    plt.scatter(
        vis[:, 0],
        vis[:, 1],
        s=POINT_SIZE,
        alpha=POINT_ALPHA,
        c=POINT_COLOR,
        linewidth=0
    )

    plt.title(
        f"Semantic Embedding Cloud ({len(vis)} points)",
        fontsize=28
    )

    plt.axis("off")
    plt.tight_layout()
    plt.savefig(OUTPUT_IMAGE, dpi=300, bbox_inches="tight")
    plt.close()

    print("Visualization completed successfully.")


# ---------------------------------------------------------
# ENTRY POINT
# ---------------------------------------------------------
if __name__ == "__main__":
    embeddings, _ = load_data_and_embeddings()
    visualize_cloud(embeddings)
