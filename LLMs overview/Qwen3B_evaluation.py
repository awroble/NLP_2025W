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
LLM_MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"

# ---------------------------------------------------------
# LLM
# ---------------------------------------------------------

def load_llm():
    global tokenizer, model

    torch.cuda.empty_cache()
    gc.collect()
    
    print(f"Loading {LLM_MODEL_NAME} with 4-bit quantization...")

    # quantization
    nf4_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16
    )

    tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_NAME)
    
    # quantization_config
    model = AutoModelForCausalLM.from_pretrained(
        LLM_MODEL_NAME,
        quantization_config=nf4_config,
        device_map="auto",
        low_cpu_mem_usage=True
    )

# ---------------------------------------------------------
# RUN
# ---------------------------------------------------------

if __name__ == "__main__":
    load_or_create_knowledge_base()
    load_llm()
    answer_questions_from_file()
    print("Done. Answers saved to answers.csv")