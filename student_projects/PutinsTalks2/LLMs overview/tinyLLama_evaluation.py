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
LLM_MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
USE_CPU_FOR_LLM = False

# ---------------------------------------------------------
# LLM
# ---------------------------------------------------------

def load_llm():
    global tokenizer, model

    torch.cuda.empty_cache()
    gc.collect()

    tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        LLM_MODEL_NAME,
        device_map="cpu" if USE_CPU_FOR_LLM else "auto",
        torch_dtype=torch.float32 if USE_CPU_FOR_LLM else torch.float16,
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
