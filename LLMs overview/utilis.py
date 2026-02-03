import numpy as np
import torch
import os
import pandas as pd

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
