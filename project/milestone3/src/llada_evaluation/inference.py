"""
LLaDA (Large Language Diffusion with mAsking) Inference Wrapper

Provides a clean interface for generating text using LLaDA-8B-Instruct model.
This wrapper handles model loading, prompt formatting, and generation with
configurable diffusion parameters.

Authors: Based on work by Kisiel, Kosakowski, Franczak, and Koniecko
Institution: Warsaw University of Technology, NLP Course Winter 2025
"""

import sys
import os
from pathlib import Path
from typing import Optional, Union, List
import time

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
import transformers

# Version compatibility check
TRANSFORMERS_REQUIRED_VERSION = "4.38.2"
if transformers.__version__ != TRANSFORMERS_REQUIRED_VERSION:
    import warnings
    warnings.warn(
        f"LLaDA requires transformers=={TRANSFORMERS_REQUIRED_VERSION} but found {transformers.__version__}. "
        f"This may cause compatibility issues. Install with: pip install -r src/requirements_llada.txt",
        UserWarning
    )

# Add LLaDA to Python path for importing generate function
LLADA_PATH = Path(__file__).parent.parent.parent / "LLaDA"
if LLADA_PATH.exists():
    sys.path.insert(0, str(LLADA_PATH))
    from generate import generate
else:
    raise ImportError(
        f"LLaDA repository not found at {LLADA_PATH}. "
        "Please ensure LLaDA is cloned in the project root."
    )

from utils import setup_logging, validate_device, format_chat_prompt, GenerationStats


class LLaDAInference:
    """
    Wrapper for LLaDA-8B-Instruct model inference.

    This class provides a simplified interface for text generation using
    the LLaDA diffusion language model with configurable parameters.
    """

    # LLaDA-specific constants
    MASK_ID = 126336  # Token ID for [MASK]
    EOS_ID = 126081   # Token ID for EOS
    EOT_ID = 126348   # Token ID for EoT (End of Turn)

    def __init__(
        self,
        model_path: str = "GSAI-ML/LLaDA-8B-Instruct",
        device: Optional[str] = None,
        steps: int = 128,
        gen_length: int = 128,
        block_length: int = 32,
        temperature: float = 0.0,
        cfg_scale: float = 0.0,
        remasking: str = "low_confidence",
        logits_eos_inf: bool = False,
        confidence_eos_eot_inf: bool = False,
        dtype: torch.dtype = torch.bfloat16,
        log_level: str = "INFO",
        log_file: Optional[str] = None
    ):
        """
        Initialize LLaDA inference wrapper.

        Args:
            model_path: HuggingFace model path or local path
            device: Device to run on ('cuda', 'mps', 'cpu', or None for auto)
            steps: Number of diffusion sampling steps (default: 128)
            gen_length: Maximum generation length (default: 128)
            block_length: Block size for semi-autoregressive remasking (default: 32)
                         Must divide gen_length evenly
            temperature: Sampling temperature for Gumbel noise (default: 0.0 = greedy)
            cfg_scale: Classifier-free guidance scale (default: 0.0 = disabled)
            remasking: Remasking strategy ('low_confidence' or 'random')
            logits_eos_inf: Set EOS logits to -inf (prevents early stopping)
            confidence_eos_eot_inf: Set EOS/EoT confidence to -inf
            dtype: Model data type (default: bfloat16)
            log_level: Logging level
            log_file: Optional log file path
        """
        self.logger = setup_logging(log_level, log_file, name="LLaDAInference")

        self.model_path = model_path
        self.device = validate_device(device)

        if steps <= 0:
            raise ValueError(f"steps must be positive, got {steps}")
        if gen_length <= 0:
            raise ValueError(f"gen_length must be positive, got {gen_length}")
        if block_length <= 0 or block_length > gen_length:
            raise ValueError(f"block_length must be in (0, {gen_length}], got {block_length}")
        if gen_length % block_length != 0:
            raise ValueError(f"gen_length ({gen_length}) must be divisible by block_length ({block_length})")
        if temperature < 0:
            raise ValueError(f"temperature must be non-negative, got {temperature}")
        if cfg_scale < 0:
            raise ValueError(f"cfg_scale must be non-negative, got {cfg_scale}")
        if remasking not in ["low_confidence", "random"]:
            raise ValueError(f"remasking must be 'low_confidence' or 'random', got '{remasking}'")

        self.steps = steps
        self.gen_length = gen_length
        self.block_length = block_length
        self.temperature = temperature
        self.cfg_scale = cfg_scale
        self.remasking = remasking
        self.logits_eos_inf = logits_eos_inf
        self.confidence_eos_eot_inf = confidence_eos_eot_inf
        self.dtype = dtype

        self.stats = GenerationStats()

        self._load_model()

    def _load_model(self):
        """Load LLaDA model and tokenizer."""
        self.logger.info(f"Loading LLaDA model: {self.model_path}")
        self.logger.info(f"Using device: {self.device}")
        self.logger.info(f"Model dtype: {self.dtype}")

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                trust_remote_code=True
            )
            self.logger.info("Tokenizer loaded successfully")

            self.model = AutoModel.from_pretrained(
                self.model_path,
                trust_remote_code=True,
                torch_dtype=self.dtype
            ).to(self.device)

            self.model.eval()
            self.logger.info("Model loaded successfully")

            if hasattr(self.tokenizer, 'mask_token_id'):
                if self.tokenizer.mask_token_id != self.MASK_ID:
                    self.logger.warning(
                        f"Expected MASK_ID={self.MASK_ID}, but tokenizer has "
                        f"mask_token_id={self.tokenizer.mask_token_id}"
                    )

            if self.device == "cuda" and torch.cuda.is_available():
                mem_allocated = torch.cuda.memory_allocated(self.device) / (1024**3)
                mem_reserved = torch.cuda.memory_reserved(self.device) / (1024**3)
                self.logger.info(
                    f"GPU memory: {mem_allocated:.2f}GB allocated, "
                    f"{mem_reserved:.2f}GB reserved"
                )

        except Exception as e:
            self.logger.error(f"Failed to load model: {type(e).__name__}: {str(e)}")
            raise

    def format_prompt(
        self,
        prompt: Union[str, List[dict]],
        add_chat_template: bool = True
    ) -> str:
        """
        Format prompt for LLaDA model.

        Args:
            prompt: Either a string or list of message dicts with 'role' and 'content'
            add_chat_template: Whether to apply chat template formatting

        Returns:
            Formatted prompt string
        """
        if isinstance(prompt, str):
            if not add_chat_template:
                return prompt
            messages = [{"role": "user", "content": prompt}]
        elif isinstance(prompt, list):
            messages = prompt
        else:
            raise TypeError(f"prompt must be str or list, got {type(prompt)}")

        if add_chat_template:
            return format_chat_prompt(messages, self.tokenizer, fallback=True)
        else:
            return "\n".join([f"{m['role']}: {m['content']}" for m in messages])

    def generate_response(
        self,
        prompt: Union[str, List[dict]],
        steps: Optional[int] = None,
        gen_length: Optional[int] = None,
        block_length: Optional[int] = None,
        temperature: Optional[float] = None,
        cfg_scale: Optional[float] = None,
        add_chat_template: bool = True,
        remove_eos: bool = True,
        return_tokens: bool = False
    ) -> Union[str, tuple]:
        """
        Generate a single response from a prompt.

        Args:
            prompt: Input prompt (string or messages list)
            steps: Override default sampling steps
            gen_length: Override default generation length
            block_length: Override default block length
            temperature: Override default temperature
            cfg_scale: Override default CFG scale
            add_chat_template: Apply chat template to prompt
            remove_eos: Remove EOS tokens from response
            return_tokens: If True, return (text, token_ids) tuple

        Returns:
            Generated text string, or (text, token_ids) if return_tokens=True
        """
        steps = steps if steps is not None else self.steps
        gen_length = gen_length if gen_length is not None else self.gen_length
        block_length = block_length if block_length is not None else self.block_length
        temperature = temperature if temperature is not None else self.temperature
        cfg_scale = cfg_scale if cfg_scale is not None else self.cfg_scale

        formatted_prompt = self.format_prompt(prompt, add_chat_template)

        try:
            input_ids = self.tokenizer(formatted_prompt, return_tensors="pt")['input_ids']
            input_ids = input_ids.to(self.device)
        except Exception as e:
            self.logger.error(f"Tokenization failed: {type(e).__name__}: {str(e)}")
            raise

        prompt_length = input_ids.shape[1]
        self.logger.debug(f"Prompt length: {prompt_length} tokens")

        start_time = time.time()
        try:
            with torch.no_grad():
                output_ids = generate(
                    model=self.model,
                    prompt=input_ids,
                    steps=steps,
                    gen_length=gen_length,
                    block_length=block_length,
                    temperature=temperature,
                    cfg_scale=cfg_scale,
                    remasking=self.remasking,
                    mask_id=self.MASK_ID,
                    logits_eos_inf=self.logits_eos_inf,
                    confidence_eos_eot_inf=self.confidence_eos_eot_inf
                )

            generation_time = time.time() - start_time

            generated_ids = output_ids[:, prompt_length:]

            response = self.tokenizer.decode(
                generated_ids[0],
                skip_special_tokens=True
            ).strip()

            if remove_eos:
                response = response.replace(self.tokenizer.eos_token, "").strip()

            tokens_generated = generated_ids.shape[1]
            self.stats.add_generation(tokens_generated, generation_time)

            self.logger.debug(
                f"Generated {tokens_generated} tokens in {generation_time:.2f}s "
                f"({tokens_generated/generation_time:.2f} tok/s)"
            )

            if return_tokens:
                return response, generated_ids[0].cpu().tolist()
            return response

        except Exception as e:
            self.logger.error(f"Generation failed: {type(e).__name__}: {str(e)}")
            self.stats.add_generation(0, time.time() - start_time, error=True)
            raise

    def generate_batch(
        self,
        prompts: List[Union[str, List[dict]]],
        **kwargs
    ) -> List[str]:
        """
        Generate responses for multiple prompts.

        Note: LLaDA's current implementation doesn't support true batch inference,
        so this processes prompts sequentially. Future versions may support batching.

        Args:
            prompts: List of prompts (strings or message lists)
            **kwargs: Arguments passed to generate_response

        Returns:
            List of generated response strings
        """
        self.logger.info(f"Generating responses for {len(prompts)} prompts (sequential)")

        responses = []
        for i, prompt in enumerate(prompts):
            try:
                response = self.generate_response(prompt, **kwargs)
                responses.append(response)

                if (i + 1) % 10 == 0:
                    self.logger.info(f"Completed {i + 1}/{len(prompts)} prompts")

            except Exception as e:
                self.logger.error(f"Failed to generate response {i}: {str(e)}")
                responses.append(f"[ERROR: {type(e).__name__}]")

        return responses

    def get_stats(self) -> dict:
        """Get generation statistics."""
        return self.stats.get_summary()

    def reset_stats(self):
        """Reset generation statistics."""
        self.stats = GenerationStats()

    def __repr__(self) -> str:
        return (
            f"LLaDAInference(model={self.model_path}, device={self.device}, "
            f"steps={self.steps}, gen_length={self.gen_length}, "
            f"block_length={self.block_length})"
        )


def main():
    """Example usage of LLaDAInference."""
    import argparse

    parser = argparse.ArgumentParser(description="Test LLaDA inference")
    parser.add_argument("--model", default="GSAI-ML/LLaDA-8B-Instruct", help="Model path")
    parser.add_argument("--device", default=None, help="Device (cuda/mps/cpu)")
    parser.add_argument("--prompt", default="What is the capital of France?", help="Test prompt")
    parser.add_argument("--steps", type=int, default=128, help="Sampling steps")
    parser.add_argument("--gen-length", type=int, default=128, help="Generation length")
    parser.add_argument("--block-length", type=int, default=32, help="Block length")

    args = parser.parse_args()

    # Initialize inference
    inference = LLaDAInference(
        model_path=args.model,
        device=args.device,
        steps=args.steps,
        gen_length=args.gen_length,
        block_length=args.block_length
    )

    print(f"\n{inference}")
    print(f"\nPrompt: {args.prompt}")
    print("-" * 80)

    # Generate response
    response = inference.generate_response(args.prompt)

    print(f"Response: {response}")
    print("-" * 80)
    print(f"\nStats: {inference.get_stats()}")


if __name__ == "__main__":
    main()
