import os
import json
import pickle
import threading
import torch
import numpy as np
import pandas as pd
import gc
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer
from FAISS_RAG_creation import load_or_create_knowledge_base

# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

# ---------------------------------------------------------
# KONFIGURACJA
# ---------------------------------------------------------
BASE_PATH = "datasets/corpus_data/data/institution/ru-putin-speeches"

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
USE_CPU_FOR_LLM = False 

app = FastAPI(title="Putin RAG API")

# Zmienne globalne
embedder = None
index = None
doc_df = None
tokenizer = None
model = None

## LLM

def load_llm():
    global tokenizer, model
    
    # Memory clear for space on GPU
    torch.cuda.empty_cache()
    gc.collect()

    print(f"\n[2/3] Loading TinyLlama (CPU Mode: {USE_CPU_FOR_LLM})...")
    tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_NAME)
    
    device_map = "cpu" if USE_CPU_FOR_LLM else "auto"
    dtype = torch.float32 if USE_CPU_FOR_LLM else torch.float16

    try:
        model = AutoModelForCausalLM.from_pretrained(
            LLM_MODEL_NAME,
            torch_dtype=dtype,
            device_map=device_map,
            low_cpu_mem_usage=True
        )
        print(f"[SUCCESS] LLM Loaded on: {model.device}")
    except Exception as e:
        print(f"[ERROR] Failed to load model: {e}")
        print("Try setting USE_CPU_FOR_LLM = True at the top of the script.")
        exit(1)

@app.on_event("startup")
async def startup_event():
    load_or_create_knowledge_base()
    load_llm()
    print("\n[3/3] API Server Ready! Access at http://localhost:8000")

# ---------------------------------------------------------
# API
# ---------------------------------------------------------

class QueryRequest(BaseModel):
    question: str
    k: int = 3

@app.post("/ask")
async def stream_answer(request: QueryRequest):
    print(f"Received question: {request.question}")
    
    # 1. RETRIEVE
    query_vector = embedder.encode([request.question])
    distances, indices = index.search(np.array(query_vector).astype('float32'), request.k)
    
    context_texts = [doc_df.iloc[idx]['text'] for idx in indices[0]]
    context_joined = "\n\n".join(context_texts)
    
    # 2. PROMPT
    prompt = f"""<|system|>
You are a helpful assistant. Use the provided context to answer the user's question.
Context:
{context_joined}
</s>
<|user|>
{request.question}
</s>
<|assistant|>
"""
    
    # 3. GENERATION
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    generation_kwargs = dict(
        inputs, 
        streamer=streamer, 
        max_new_tokens=256, 
        temperature=0.7, 
        repetition_penalty=1.1
    )
    
    def generate_with_error_logging():
        try:
            model.generate(**generation_kwargs)
        except Exception as e:
            print(f"\n[CRITICAL ERROR IN THREAD] {e}")

    thread = threading.Thread(target=generate_with_error_logging)
    thread.start()
    
    def token_generator():
        try:
            for new_text in streamer:
                yield new_text
        except Exception as e:
            yield f"\n[SERVER ERROR]: {str(e)}"

    return StreamingResponse(token_generator(), media_type="text/plain")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)