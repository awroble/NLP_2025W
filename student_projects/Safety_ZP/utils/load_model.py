import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM


def load_hf_model(model_name: str, dtype=torch.float16, device_map="auto"):
    """
    Loads a HuggingFace model and tokenizer.

    Tries CausalLM first, then Seq2SeqLM.

    Returns:
        model (PreTrainedModel)
        tokenizer (PreTrainedTokenizer)
        model_type (str): "causal" or "seq2seq"

    Raises:
        RuntimeError if model cannot be loaded
    """
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    except Exception as e:
        raise RuntimeError(f"Failed to load tokenizer for {model_name}: {e}")

    # Try causal LM
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            device_map=device_map
        )
        return model, tokenizer, "causal"

    except Exception as causal_error:
        try:
            model = AutoModelForSeq2SeqLM.from_pretrained(
                model_name,
                torch_dtype=dtype,
                device_map=device_map
            )
            return model, tokenizer, "seq2seq"

        except Exception as seq2seq_error:
            raise RuntimeError(
                f"Failed to load model {model_name}.\n"
                f"CausalLM error: {causal_error}\n"
                f"Seq2SeqLM error: {seq2seq_error}"
            )