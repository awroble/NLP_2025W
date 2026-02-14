import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import json
import numpy as np
import os

BASE_PATH = "datasets/corpus_data/data/institution/ru-putin-speeches"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
TARGET_SPEAKERS = {"PRESIDENT PUTIN", "VLADIMIR PUTIN"}
FAISS_INDEX_FILE = "faiss_index.bin"
CHUNKS_FILE = "chunks.pkl"
import pickle

# ---------------------------------------------------------
# KNOWLEDGE BASE
# ---------------------------------------------------------

def load_or_create_knowledge_base():
    """Load existing FAISS index or build it from the corpus if missing."""
    global index, doc_df, embedder

    # Load embedding model on CPU to keep GPU free for the LLM
    print("Loading embedding model on CPU...")
    embedder = SentenceTransformer(EMBEDDING_MODEL_NAME, device="cpu")

    # Load cached FAISS index and chunks if available
    if os.path.exists(FAISS_INDEX_FILE) and os.path.exists(CHUNKS_FILE):
        print("Loading existing FAISS index...")
        index = faiss.read_index(FAISS_INDEX_FILE)
        with open(CHUNKS_FILE, "rb") as f:
            doc_df = pickle.load(f)
    else:
        print("Building knowledge base from scratch...")
        _build_knowledge_base_from_scratch()


def _build_knowledge_base_from_scratch():
    """Create FAISS index from the Putin speech corpus."""
    global index, doc_df
    corpus_data = []

    print(f"Scanning {BASE_PATH}...")
    for root, _, files in os.walk(BASE_PATH):
        for file in files:
            if not file.endswith(".json"):
                continue
            try:
                with open(os.path.join(root, file), encoding="utf-8") as f:
                    data = json.load(f)

                speaker = data.get("speaker")
                structured = data.get("structured_content", [])

                # Handle unstructured speeches
                if speaker in TARGET_SPEAKERS and not structured:
                    text = data.get("text") or data.get("content")
                    if text:
                        corpus_data.append({"text": text, "source": file})

                # Handle structured speech blocks
                for block in structured:
                    if block.get("speaker", speaker) in TARGET_SPEAKERS:
                        text = block.get("text") or block.get("content")
                        if text:
                            corpus_data.append({"text": text, "source": file})
            except Exception:
                continue

    if not corpus_data:
        print("Warning: No data found. Check BASE_PATH.")
        return

    df = pd.DataFrame(corpus_data)
    df = df[df.text.str.len() > 0]

    chunks = []
    print(f"Chunking {len(df)} documents...")
    for _, row in df.iterrows():
        text = row.text
        # Smaller chunks improve FAISS retrieval quality
        for i in range(0, len(text), 700):
            chunks.append({"text": text[i:i+800], "source": row.source})

    doc_df = pd.DataFrame(chunks)

    print(f"Embedding {len(doc_df)} chunks...")
    embeddings = embedder.encode(
        doc_df.text.tolist(),
        batch_size=32,
        show_progress_bar=True
    )

    # Build FAISS index
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings).astype("float32"))

    # Persist index and chunk metadata
    faiss.write_index(index, FAISS_INDEX_FILE)
    with open(CHUNKS_FILE, "wb") as f:
        pickle.dump(doc_df, f)
