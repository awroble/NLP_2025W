import gc
import torch
from typing import List, Dict
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoProcessor, 
    AutoModelForVision2Seq,
    LlavaForConditionalGeneration
)

from PIL import Image

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32
MAX_CONTEXT_TOKENS = 4096

torch.set_grad_enabled(False)
torch.manual_seed(42)

if DEVICE == "cuda":
    torch.backends.cuda.matmul.allow_tf32 = True


def soft_clear_vram():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


class LLMManager:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.processor = None
        self.model_id = None

    # ================= TEXT =================

    def load_text_model(self, model_id: str):
        if self.model_id == model_id:
            return

        self.unload_model()

        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"

        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=DTYPE,
            device_map=DEVICE,
        ).eval()

        self.model_id = model_id

    def serialize_chat(self, history):
        parts = []
        for t in history:
            parts.append(f"<|{t['role']}|>\n{t['content']}")
        parts.append("<|assistant|>\n")
        return "\n".join(parts)

    @torch.no_grad()
    def _generate_text_batch(self, prompts, max_new_tokens, temperature, top_p, do_sample):
        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=MAX_CONTEXT_TOKENS,
        ).to(DEVICE)

        output = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature if do_sample else 0.0,
            top_p=top_p if do_sample else 1.0,
            pad_token_id=self.tokenizer.eos_token_id,
            use_cache=True,
        )

        decoded = self.tokenizer.batch_decode(
            output[:, inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        )

        del inputs, output
        return decoded

    def generate_single_turn(
        self,
        prompts: List[str],
        batch_size=8,
        max_new_tokens=256,
        temperature=0.7,
        top_p=0.9,
    ):
        do_sample = temperature > 0.0
        out = []

        for i in range(0, len(prompts), batch_size):
            out.extend(
                self._generate_text_batch(
                    prompts[i:i + batch_size],
                    max_new_tokens,
                    temperature,
                    top_p,
                    do_sample,
                )
            )
        return out
    
    def generate_chat_once(
        self,
        model_id: str,
        histories: List[List[Dict[str, str]]],
        batch_size: int = 8,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ):
        """
        histories: [
            [
                {"role": "user", "content": "..."},
                {"role": "assistant", "content": "..."},
                ...
            ],
            ...
        ]
        """

        # 1️⃣ Load text model
        self.load_text_model(model_id)

        # 2️⃣ Serialize chats → flat prompts
        prompts = [self.serialize_chat(h) for h in histories]

        # 3️⃣ Generate
        outputs = self.generate_single_turn(
            prompts=prompts,
            batch_size=batch_size,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
        )

        return outputs

    # ================= MULTIMODAL =================

    def load_vision_model(self, model_id: str):
        if self.model_id == model_id:
            return

        self.unload_model()

        self.processor = AutoProcessor.from_pretrained(
            model_id,
            trust_remote_code=True,
        )

        if model_id == "llava-hf/llava-1.5-7b-hf":
            self.model = LlavaForConditionalGeneration.from_pretrained( 
                model_id,  
                torch_dtype=DTYPE, 
                device_map=DEVICE,
            ).eval()
        else:
            self.model = AutoModelForVision2Seq.from_pretrained(
                model_id,
                torch_dtype=DTYPE,
                device_map="auto",
                trust_remote_code=True,
            ).eval()
        self.model_id = model_id

    @torch.no_grad()
    def generate_multimodal(
        self,
        model_id: str,
        prompts: List[str],
        image_paths: List[str],
        max_new_tokens: int = 128,
        batch_size: int = 4,
    ):
        assert len(prompts) == len(image_paths)

        self.load_vision_model(model_id)

        outputs = []

        for i in range(0, len(prompts), batch_size):
            batch_prompts = prompts[i:i + batch_size]
            batch_images = image_paths[i:i + batch_size]

            images = [
                Image.open(p).convert("RGB")
                for p in batch_images
            ]

            inputs = self.processor(
                text=batch_prompts,
                images=images,
                return_tensors="pt",
                padding=True,
            ).to(DEVICE)

            generated = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.processor.tokenizer.eos_token_id,
                use_cache=True,
            )

            decoded = self.processor.batch_decode(
                generated,
                skip_special_tokens=True,
            )
            outputs.extend([d.strip() for d in decoded])

            del inputs, generated, images
            soft_clear_vram()

        return outputs


    # ================= CLEANUP =================

    def unload_model(self):
        if self.model is not None:
            del self.model
        if self.tokenizer is not None:
            del self.tokenizer
        if self.processor is not None:
            del self.processor

        self.model = None
        self.tokenizer = None
        self.processor = None
        self.model_id = None

        soft_clear_vram()
