# https://huggingface.co/google/gemma-3n-E2B-it
import os
import torch
import numpy as np
import pandas as pd
import gc
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
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

QUESTIONS_FILE = "questions.txt"
OUTPUT_CSV = "answers.csv"

USE_CPU_FOR_LLM = False
embedder = None
index = None
doc_df = None
tokenizer = None
model = None

def answer_question(question, k=3):
    query_vec = embedder.encode([question])
    _, indices = index.search(np.array(query_vec).astype("float32"), k)

    context = "\n\n".join(doc_df.iloc[i].text for i in indices[0])

    # Content for LLM
    messages = [
        {"role": "system", "content": f"You are a helpful assistant. Answer based only on the context below. Be concise.\n\nContext:\n{context}"},
        {"role": "user", "content": question}
    ]
    
    # apply_chat_template adds correct tags <|im_start|>
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=200,
            temperature=0.7,
            repetition_penalty=1.1
        )

    # Save answer getter
    input_len = inputs["input_ids"].shape[1]
    generated_ids = output[0][input_len:]
    answer = tokenizer.decode(generated_ids, skip_special_tokens=True)
    
    return answer.strip()

# ---------------------------------------------------------
# MAIN BATCH PROCESS
# ---------------------------------------------------------

def answer_questions_from_file():
    results = []
    
    # questions proessing
    if not os.path.exists(QUESTIONS_FILE):
        print(f"File {QUESTIONS_FILE} not found.")

    with open(QUESTIONS_FILE, encoding="utf-8") as f:
        questions = [q.strip() for q in f if q.strip()]

    for q in questions:
        print(f"Answering: {q}")
        try:
            a = answer_question(q)
        except Exception as e:
            a = f"ERROR: {e}"
            print(a)
        results.append({"question": q, "answer": a})

    pd.DataFrame(results).to_csv(OUTPUT_CSV, index=False)


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

# Model: Google Gemma 3n E2B Instruct (Next Gen architecture)
LLM_MODEL_NAME = "google/gemma-3n-E2B-it"

# ---------------------------------------------------------
# LLM LOAD
# ---------------------------------------------------------

def load_llm_4b():
    global tokenizer, model

    # Agresywne czyszczenie pamięci GPU przed załadowaniem
    if 'model' in globals() and model is not None: del model
    if 'tokenizer' in globals() and tokenizer is not None: del tokenizer
    torch.cuda.empty_cache()
    gc.collect()
    
    print(f"Loading {LLM_MODEL_NAME} with 4-bit quantization...")

    # Konfiguracja NF4 (optymalna dla kart 4-8GB VRAM)
    nf4_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16
    )

    try:
        # 1. Tokenizer
        # trust_remote_code=True jest KLUCZOWE dla Gemma 3n (nowa architektura)
        tokenizer = AutoTokenizer.from_pretrained(
            LLM_MODEL_NAME, 
            trust_remote_code=True
        )
        
        # 2. Model
        model = AutoModelForCausalLM.from_pretrained(
            LLM_MODEL_NAME,
            quantization_config=nf4_config,
            device_map="auto",
            low_cpu_mem_usage=True,
            trust_remote_code=True,  # Wymagane dla Gemma 3n
            attn_implementation="eager" # Stabilność ponad szybkość Flash Attn
        )
        
        model.generation_config.max_new_tokens = 256  # Limit dla zwięzłych odpowiedzi
        model.generation_config.temperature = 0.1     # Precyzja (fakty RAG)
        model.generation_config.do_sample = True
        model.generation_config.repetition_penalty = 1.15 # Zapobieganie zapętleniom
        
        # Gemma 3n ma duże okno kontekstowe (32k), ale RAG zazwyczaj używa mniej.
        # Tokenizer automatycznie obsłuży chat template.
        
        print(f"[SUCCESS] Model loaded on {model.device}")
        
    except Exception as e:
        print(f"\n[CRITICAL ERROR] Failed to load model: {e}")
        print("Wskazówka: Upewnij się, że masz najnowszą wersję transformers:")
        print("pip install -U transformers")
        exit(1)


# ---------------------------------------------------------
# RUN
# ---------------------------------------------------------

if __name__ == "__main__":
    load_or_create_knowledge_base()
    load_llm_4b()
    answer_questions_from_file()
    
    print(f"Done. Answers saved to answers.csv")