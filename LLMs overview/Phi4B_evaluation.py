# LLM_MODEL_NAME = "microsoft/Phi-4-mini-instruct"
# https://huggingface.co/microsoft/Phi-3.5-mini-instruct
import os
import torch
import numpy as np
import pandas as pd
import gc
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from FAISS_RAG_creation import load_or_create_knowledge_base
from utilis import answer_questions_from_file

# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

# Core model
LLM_MODEL_NAME = "microsoft/Phi-3.5-mini-instruct"

# ---------------------------------------------------------
# LLM LOAD
# ---------------------------------------------------------

def load_llm_4b():
    global tokenizer, model

    # Clear CUDA cache before loading model for memory
    if model is not None: del model
    if tokenizer is not None: del tokenizer
    torch.cuda.empty_cache()
    gc.collect()
    
    print(f"Loading {LLM_MODEL_NAME} with 4-bit quantization...")

    # Configuration 4B for GPU compute
    nf4_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16
    )

    try:
        # 1. Tokenizer
        tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_NAME)
        
        # 2. Model
        # attn_implementation="eager" -> Disabled Flash Attention (Protects from errors)
        model = AutoModelForCausalLM.from_pretrained(
            LLM_MODEL_NAME,
            quantization_config=nf4_config,
            device_map="auto",
            low_cpu_mem_usage=True,
            trust_remote_code=False, 
            attn_implementation="eager" 
        )
        print(f"[SUCCESS] Model loaded on {model.device}")
        
    except Exception as e:
        print(f"\n[CRITICAL ERROR] Failed to load model: {e}")
        exit(1)


# ---------------------------------------------------------
# RUN
# ---------------------------------------------------------

if __name__ == "__main__":
    load_or_create_knowledge_base()
    load_llm_4b()
    answer_questions_from_file()
    print(f"Done. Answers saved to answers.csv")