"""
MMaDA (Multimodal Large Diffusion Language Model) Inference Wrapper

Provides a clean interface for text generation using MMaDA-8B-MixCoT model.
This wrapper focuses on text-only generation using the multimodal diffusion model.

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
from transformers import AutoTokenizer
import transformers

TRANSFORMERS_REQUIRED_VERSION = "4.46.0"
if transformers.__version__ != TRANSFORMERS_REQUIRED_VERSION:
    import warnings
    warnings.warn(
        f"MMaDA requires transformers=={TRANSFORMERS_REQUIRED_VERSION} but found {transformers.__version__}. "
        f"This may cause compatibility issues. Install with: pip install -r src/requirements_mmada.txt",
        UserWarning
    )

MMADA_PATH = Path(__file__).parent.parent.parent / "MMaDA"
if MMADA_PATH.exists():
    sys.path.insert(0, str(MMADA_PATH))
    from models import MMadaModelLM
    from models.modeling_mmada import add_gumbel_noise, get_num_transfer_tokens
else:
    raise ImportError(
        f"MMaDA repository not found at {MMADA_PATH}. "
        "Please ensure MMaDA is cloned in the project root."
    )

from utils import setup_logging, validate_device, format_chat_prompt, GenerationStats


class MMaDAInference:
    """
    Wrapper for MMaDA-8B-MixCoT model inference (text-only).

    This class provides a simplified interface for text generation using
    the MMaDA multimodal diffusion model with Chain-of-Thought capabilities.
    """

    # MMaDA uses same token IDs as LLaDA (based on same foundation)
    MASK_ID = 126336  # Token ID for [MASK]
    EOS_ID = 126081   # Token ID for EOS
    EOT_ID = 126348   # Token ID for EoT (End of Turn)

    def __init__(
        self,
        model_path: str = "Gen-Verse/MMaDA-8B-MixCoT",
        device: Optional[str] = None,
        steps: int = 128,
        gen_length: int = 128,
        block_length: int = 32,
        temperature: float = 0.0,
        remasking: str = "low_confidence",
        dtype: torch.dtype = torch.bfloat16,
        log_level: str = "INFO",
        log_file: Optional[str] = None
    ):
        """
        Initialize MMaDA inference wrapper.

        Args:
            model_path: HuggingFace model path or local path
            device: Device to run on ('cuda', 'mps', 'cpu', or None for auto)
            steps: Number of diffusion sampling steps (default: 128)
            gen_length: Maximum generation length (default: 128)
            block_length: Block size for semi-autoregressive remasking (default: 32)
                         Must divide gen_length evenly
            temperature: Sampling temperature for Gumbel noise (default: 0.0 = greedy)
            remasking: Remasking strategy ('low_confidence' or 'random')
            dtype: Model data type (default: bfloat16)
            log_level: Logging level
            log_file: Optional log file path
        """
        self.logger = setup_logging(log_level, log_file, name="MMaDAInference")

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
        if remasking not in ["low_confidence", "random"]:
            raise ValueError(f"remasking must be 'low_confidence' or 'random', got '{remasking}'")

        self.steps = steps
        self.gen_length = gen_length
        self.block_length = block_length
        self.temperature = temperature
        self.remasking = remasking
        self.dtype = dtype

        self.stats = GenerationStats()

        self._load_model()

    def _load_model(self):
        """Load MMaDA model and tokenizer."""
        self.logger.info(f"Loading MMaDA model: {self.model_path}")
        self.logger.info(f"Using device: {self.device}")
        self.logger.info(f"Model dtype: {self.dtype}")

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                trust_remote_code=True
            )
            self.logger.info("Tokenizer loaded successfully")

            self.model = MMadaModelLM.from_pretrained(
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
        Format prompt for MMaDA model.

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

    def _generate_diffusion(
        self,
        prompt_ids: torch.Tensor,
        steps: int,
        gen_length: int,
        block_length: int,
        temperature: float
    ) -> torch.Tensor:
        """
        Internal method implementing MMaDA's diffusion generation.

        This is adapted from MMaDA's generate functions but simplified for text-only.

        Args:
            prompt_ids: Tokenized prompt tensor [1, prompt_len]
            steps: Sampling steps
            gen_length: Generation length
            block_length: Block length for semi-autoregressive
            temperature: Sampling temperature

        Returns:
            Generated token IDs [1, prompt_len + gen_length]
        """
        x = torch.full(
            (1, prompt_ids.shape[1] + gen_length),
            self.MASK_ID,
            dtype=torch.long,
            device=self.device
        )
        x[:, :prompt_ids.shape[1]] = prompt_ids.clone()

        assert gen_length % block_length == 0
        num_blocks = gen_length // block_length

        assert steps % num_blocks == 0
        steps_per_block = steps // num_blocks

        for block_idx in range(num_blocks):
            block_start = prompt_ids.shape[1] + block_idx * block_length
            block_end = prompt_ids.shape[1] + (block_idx + 1) * block_length

            block_mask_index = (x[:, block_start:block_end] == self.MASK_ID)
            num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps_per_block)

            for step_idx in range(steps_per_block):
                mask_index = (x == self.MASK_ID)

                logits = self.model(x).logits

                mask_index[:, block_end:] = False

                if self.remasking == "low_confidence":
                    logits_with_noise = add_gumbel_noise(logits, temperature)
                    x0 = torch.argmax(logits_with_noise, dim=-1)

                    confidence = logits_with_noise.max(dim=-1).values

                    num_to_transfer = num_transfer_tokens[:, step_idx]

                    confidence = confidence.masked_fill(~mask_index, float('inf'))

                    _, indices = confidence.sort(dim=1)

                    transfer_index = torch.zeros_like(mask_index)
                    for b in range(x.shape[0]):
                        num_to_keep = mask_index[b].sum() - num_to_transfer[b]
                        if num_to_keep > 0:
                            keep_indices = indices[b, :num_to_keep]
                            transfer_mask = torch.ones(x.shape[1], dtype=torch.bool, device=self.device)
                            transfer_mask[keep_indices] = False
                            transfer_index[b] = transfer_mask & mask_index[b]
                        else:
                            transfer_index[b] = mask_index[b]

                    x[transfer_index] = x0[transfer_index]

                elif self.remasking == "random":
                    logits_with_noise = add_gumbel_noise(logits, temperature)
                    x0 = torch.argmax(logits_with_noise, dim=-1)

                    num_to_transfer = num_transfer_tokens[:, step_idx]

                    transfer_index = torch.zeros_like(mask_index)
                    for b in range(x.shape[0]):
                        masked_positions = mask_index[b].nonzero(as_tuple=False).squeeze(-1)
                        if len(masked_positions) > 0:
                            num_transfer = min(num_to_transfer[b].item(), len(masked_positions))
                            if num_transfer > 0:
                                selected = masked_positions[torch.randperm(len(masked_positions))[:num_transfer]]
                                transfer_index[b, selected] = True

                    x[transfer_index] = x0[transfer_index]

                if (x[:, block_start:block_end] == self.MASK_ID).sum() == 0:
                    break

        return x

    def generate_response(
        self,
        prompt: Union[str, List[dict]],
        steps: Optional[int] = None,
        gen_length: Optional[int] = None,
        block_length: Optional[int] = None,
        temperature: Optional[float] = None,
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
                output_ids = self._generate_diffusion(
                    prompt_ids=input_ids,
                    steps=steps,
                    gen_length=gen_length,
                    block_length=block_length,
                    temperature=temperature
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

        Note: Current implementation processes prompts sequentially.

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
            f"MMaDAInference(model={self.model_path}, device={self.device}, "
            f"steps={self.steps}, gen_length={self.gen_length}, "
            f"block_length={self.block_length})"
        )


def main():
    """Example usage of MMaDAInference."""
    import argparse

    parser = argparse.ArgumentParser(description="Test MMaDA inference")
    parser.add_argument("--model", default="Gen-Verse/MMaDA-8B-MixCoT", help="Model path")
    parser.add_argument("--device", default=None, help="Device (cuda/mps/cpu)")
    parser.add_argument("--prompt", default="What is the capital of France?", help="Test prompt")
    parser.add_argument("--steps", type=int, default=128, help="Sampling steps")
    parser.add_argument("--gen-length", type=int, default=128, help="Generation length")
    parser.add_argument("--block-length", type=int, default=32, help="Block length")

    args = parser.parse_args()

    inference = MMaDAInference(
        model_path=args.model,
        device=args.device,
        steps=args.steps,
        gen_length=args.gen_length,
        block_length=args.block_length
    )

    print(f"\n{inference}")
    print(f"\nPrompt: {args.prompt}")
    print("-" * 80)

    response = inference.generate_response(args.prompt)

    print(f"Response: {response}")
    print("-" * 80)
    print(f"\nStats: {inference.get_stats()}")


if __name__ == "__main__":
    main()
