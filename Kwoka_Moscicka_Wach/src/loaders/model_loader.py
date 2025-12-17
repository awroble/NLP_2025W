import os
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'

import torch
from transformers import (
    AutoTokenizer,
    AutoProcessor,
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    AutoModel,
    BitsAndBytesConfig
)
from PIL import Image
import yaml
from pathlib import Path

class HFModel:

    def __init__(self, checkpoint, model_type, max_history=5):

        self.model_type = model_type
        self.max_history = max_history
        self.messages = [] #store history
        self.bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype="bfloat16"
            )

        # device selection
        if torch.backends.mps.is_available() and torch.backends.mps.is_built():
            self.device = torch.device("mps")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        # text
        if model_type == "text":
            self.tokenizer = AutoTokenizer.from_pretrained(checkpoint, use_fast=True)

            self.model = AutoModelForCausalLM.from_pretrained(checkpoint, dtype='auto',
                quantization_config=self.bnb_config
            ).to(self.device)

            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            self.processor = None # not needed

        elif model_type == "multimodal":
            self.processor = AutoProcessor.from_pretrained(checkpoint)

            try:
                self.model = AutoModelForImageTextToText.from_pretrained(checkpoint, dtype='auto',
                                                                         quantization_config=self.bnb_config
                                                                        ).to(self.device)
            except:
                self.model = AutoModel.from_pretrained(checkpoint, dtype='auto',
                                                       quantization_config=self.bnb_config
                                                       ).to(self.device)
            self.tokenizer = getattr(self.processor, "tokenizer", None)
        else:
            raise ValueError("model_type must be either 'text' or 'vision'")

    def add_message(self, role, content):
        self.messages.append({"role": role, "content": content})

        if len(self.messages) > self.max_history * 2:  # each turn has user+assistant
            self.messages = self.messages[-self.max_history*2:]

    def generate(self, prompt_dict, **generate_kwargs):
        data_type = prompt_dict.get('data_type', None)
        prompt_content = prompt_dict.get('prompt', None)

        if data_type == 'text':
            self.add_message('user', prompt_content)

        elif data_type == 'image':
            text = prompt_content.get("text", "")
            image_path = prompt_content.get("path")
            content = [
                {"type": "image", "path": image_path},
                {"type": "text", "text": f"<image> {text}"}
            ]
            self.add_message("user", content)
        else:
            raise ValueError("data_type must be either 'text' or 'image'")


        ## text mode
        if self.model_type == "text":
            prompt = "\n".join(f"{m['role'].upper()}: {m['content']}" for m in self.messages)
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            outputs = self.model.generate(**inputs, **generate_kwargs)

            return self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

        elif self.model_type == "multimodal":
            texts = []
            images = []

            for m in self.messages:
                if isinstance(m["content"], str):
                    texts.append(m["content"])
                elif isinstance(m["content"], list):
                    for item in m["content"]:
                        if item["type"] == "text":
                            texts.append(item["text"])
                        elif item["type"] == "image":
                            # load image once per turn
                            images.append(Image.open(item["path"]).convert("RGB"))

            text_input = "\n".join(texts) if texts else None

            inputs = self.processor(text=text_input, images=images if images else None, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            outputs = self.model.generate(**inputs, **generate_kwargs)
            if hasattr(self.processor, "tokenizer"):
                out_text = self.processor.tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
            else:
                out_text = "[Cannot decode: no tokenizer]"

        else:
            raise ValueError("model_type must be either 'text' or 'vision'")

        self.add_message("assistant", out_text)
        return out_text

    def clear_history(self):
        self.messages = []
