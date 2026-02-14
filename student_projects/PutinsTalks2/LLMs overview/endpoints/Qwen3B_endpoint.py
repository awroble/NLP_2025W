import os
import torch
import numpy as np
import pandas as pd
import gc
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from FAISS_RAG_creation import load_or_create_knowledge_base

# ---------------------------------------------------------
# KONFIGURACJA
# ---------------------------------------------------------
BASE_PATH = "datasets/corpus_data/data/institution/ru-putin-speeches"
LLM_MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"

app = FastAPI(title="Putin RAG API (Qwen Edition)")

# Zmienne globalne
embedder = None
index = None
doc_df = None
tokenizer = None
model = None

def load_llm():
    global tokenizer, model
    
    # clear memory
    torch.cuda.empty_cache()
    gc.collect()

    print(f"\n[2/3] Loading {LLM_MODEL_NAME} (4-bit quantization)...")
    
    # 4 bit configuration for memory
    nf4_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16
    )

    tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        LLM_MODEL_NAME,
        quantization_config=nf4_config,
        device_map="auto",
        low_cpu_mem_usage=True
    )
    print(f"[SUCCESS] LLM Loaded.")

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
    
    # 2. PROMPT (ZMIANA NA CHAT TEMPLATE DLA QWEN)
    # Qwen uses ChatML (<|im_start|>system...), so we use tokenizer.apply_chat_template
    messages = [
        {"role": "system", "content": f"You are a helpful assistant. Use the provided context to answer the user's question accurately.\n\nContext:\n{context_joined}"},
        {"role": "user", "content": request.question}
    ]
    prompt_text = tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True
    )
    
    # 3. GENERATION
    inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
    
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    
    generation_kwargs = dict(
        inputs, 
        streamer=streamer, 
        max_new_tokens=512,  
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