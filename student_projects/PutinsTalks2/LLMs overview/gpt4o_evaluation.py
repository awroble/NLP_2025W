import os
import json
import pickle
import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from openai import OpenAI
from FAISS_RAG_creation import load_or_create_knowledge_base

# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

os.environ["OPENAI_API_KEY"] = "___" 
OPENAI_MODEL_NAME = "gpt-4o" 

# parameters
TEMPERATURE = 0.1
MAX_TOKENS = 300

client = None
embedder = None
index = None
doc_df = None
QUESTIONS_FILE = "questions.txt"
OUTPUT_CSV = "answers.csv"

# ---------------------------------------------------------
# SETUP CLIENT
# ---------------------------------------------------------

def setup_openai_client():
    global client
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key or "___" in api_key:
        print("ERROR: no OPENAI_API_KEY!")
        return
    client = OpenAI(api_key=api_key)
    print(f"OpenAI Client initialized. Using model: {OPENAI_MODEL_NAME}")

# ---------------------------------------------------------
# QA LOGIC (OpenAI API)
# ---------------------------------------------------------

def answer_question(question, k=3):
    # 1. Looking for context frames
    query_vec = embedder.encode([question])
    _, indices = index.search(np.array(query_vec).astype("float32"), k)

    context_list = [doc_df.iloc[i].text for i in indices[0]]
    context = "\n\n".join(context_list)
    
    # 2. OpenAI prompt
    messages = [
        {"role": "system", "content": f"You are a helpful assistant. Answer based ONLY on the context below. If the answer is not in the context, say 'I don't know'.\n\nContext:\n{context}"},
        {"role": "user", "content": question}
    ]
    
    try:
        response = client.chat.completions.create(
            model=OPENAI_MODEL_NAME,
            messages=messages,
            temperature=TEMPERATURE, 
            max_tokens=MAX_TOKENS
        )
        return response.choices[0].message.content.strip()
    
    except Exception as e:
        return f"OPENAI ERROR: {str(e)}"

# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------

def answer_questions_from_file():
    results = []
    
    if not os.path.exists(QUESTIONS_FILE):
        print(f"File {QUESTIONS_FILE} not found. Creating dummy.")
        with open(QUESTIONS_FILE, "w", encoding="utf-8") as f:
            f.write("What is the main threat to Russia?\n")

    with open(QUESTIONS_FILE, encoding="utf-8") as f:
        questions = [q.strip() for q in f if q.strip()]

    print(f"Processing {len(questions)} questions using {OPENAI_MODEL_NAME}...")

    for q in questions:
        print(f"Answering: {q}")
        a = answer_question(q)
        results.append({"question": q, "answer": a})

    pd.DataFrame(results).to_csv(OUTPUT_CSV, index=False)

# ---------------------------------------------------------
# RUN
# ---------------------------------------------------------

if __name__ == "__main__":
    setup_openai_client()
    load_or_create_knowledge_base()
    if client:
        answer_questions_from_file()
        print(f"Done. Answers saved to {OUTPUT_CSV}")