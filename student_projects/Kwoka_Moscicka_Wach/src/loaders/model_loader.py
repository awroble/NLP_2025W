import os
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'

import torch
from transformers import (
    AutoProcessor,
    AutoModelForImageTextToText,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig
)
from PIL import Image

class HFModel:
    """
    Class to load a model from Hugging Face Hub.
    """
    def __init__(self, checkpoint, model_type, max_history, multimodal_format=None, image_token=None):
        """
        Initialize a Hugging Face model.
        :param checkpoint: Checkpoint name from Hugging Face Hub, e.g. 'llava-hf/llava-1.5-7b-hf'
        :param model_type: Model type: 'text' or 'multimodal'.
        :param max_history: Max history size for multimodal conversations.
        :param multimodal_format: Optional. If model_type is 'multimodal' specifies how to pass multimodal prompts to the model.
        :param image_token: Optional. If multimodal_format is 'image_token' specifies which image token to use.
        """
        self.model_type = model_type
        self.multimodal_format = multimodal_format
        self.image_token = image_token
        self.max_history = max_history
        self.messages = []

        # quantization configuration
        self.bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16
        )

        # setting the device
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        # read model from checkpoint for multimodal data
        if model_type == "multimodal":
            self.processor = AutoProcessor.from_pretrained(checkpoint)
            self.model = AutoModelForImageTextToText.from_pretrained(
                checkpoint,
                quantization_config=self.bnb_config,
                dtype="auto",
                low_cpu_mem_usage=True,
            ).to(self.device)

            self.tokenizer = getattr(self.processor, "tokenizer", None)

        # read model from checkpoint for text only data
        elif model_type == "text":
            self.tokenizer = AutoTokenizer.from_pretrained(checkpoint, use_fast=True)
            self.model = AutoModelForCausalLM.from_pretrained(
                checkpoint,
                quantization_config=self.bnb_config,
                dtype="auto",
            ).to(self.device)
            self.processor = None # no processor needed for text only

        else:
            raise ValueError("model_type must be 'text' or 'multimodal'")

        # set pad_token if not exists
        if self.tokenizer is not None:
            if self.tokenizer.pad_token_id is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            self.model.generation_config.pad_token_id = self.tokenizer.pad_token_id

        self.model.eval()

    def clear_history(self):
        """
        Clear all history of the model.
        """
        self.messages = []

    def add_message(self, role, content):
        """
        Add a message to the model. For each role keeps only number of messages specified by max_history.
        :param role: User or Assistant.
        :param content: Prompt content.
        """
        self.messages.append({"role": role, "content": content})
        if len(self.messages) > self.max_history * 2:
            self.messages = self.messages[-self.max_history * 2:]

    def generate(self, prompt_dict, **generate_kwargs):
        """
        Generate prompt response from the model.
        :param prompt_dict: Dictionary with prompt that should be passed to the model. Requires keys 'prompt' and 'data_type'.
        :param generate_kwargs: Generation params accepted by model.
        :return: Models response to the prompt.
        """

        if not isinstance(prompt_dict, dict):
            raise ValueError("prompt_dict must be a dictionary.")
        if 'data_type' not in prompt_dict.keys():
            raise ValueError("prompt_dict must contain 'data_type' key.")
        if 'prompt' not in prompt_dict.keys():
            raise ValueError("prompt_dict must contain 'prompt' key.")

        data_type = prompt_dict.get('data_type')
        prompt = prompt_dict.get('prompt')

        # Process text only prompts.
        if data_type == "text":
            self.add_message("user", prompt)
            full_prompt = ""

            # construct conversation for multiturn
            for m in self.messages:
                full_prompt += f"{m['role'].upper()}: {m['content']}\n"
            full_prompt += "ASSISTANT: "

            inputs = self.tokenizer(full_prompt, return_tensors="pt").to(self.device)

            # generate model response
            outputs = self.model.generate(**inputs, **generate_kwargs)

            # remove inputs (user prompt) from output
            gen = outputs[0][inputs["input_ids"].shape[-1]:]

            # decode output
            return self.tokenizer.decode(gen, skip_special_tokens=True).strip()

        # Process multimodal prompts
        elif data_type == "multimodal":
            text = prompt.get("text", "")
            image_path = prompt.get("path")

            # open and resize image
            image = Image.open(image_path).convert("RGB").resize((336, 336)) if image_path else None

            # Construct input for models that accept format 'messages'
            if self.multimodal_format == "messages":
                messages = self.messages + [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image"},
                            {"type": "text", "text": text},
                        ],
                    }
                ]

                prompt_text = self.processor.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )

                inputs = self.processor(
                    text=prompt_text,
                    images=image,
                    return_tensors="pt"
                ).to(self.device)

            elif self.multimodal_format == "image_token":
                # Build text with <image> placeholder
                full_text = f"{self.image_token} {text}"

                # Add previous messages if any
                if self.messages:
                    past_texts = []
                    for m in self.messages:
                        if isinstance(m["content"], str):
                            past_texts.append(m["content"])
                        elif isinstance(m["content"], list):
                            for item in m["content"]:
                                if item["type"] == "text":
                                    past_texts.append(item["text"])
                    full_text = "\n".join(past_texts + [full_text])

                inputs = self.processor(
                    text=full_text,
                    images=image,
                    return_tensors="pt"
                ).to(self.device)

            else:
                raise ValueError("Unknown multimodal_format")

            # generate outputs
            outputs = self.model.generate(
                **inputs,
                **generate_kwargs
            )

            # remove input prompt from output
            input_len = inputs["input_ids"].shape[-1]
            gen_ids = outputs[0][input_len:]

            return self.processor.tokenizer.decode(
                gen_ids,
                skip_special_tokens=True
            ).strip()
