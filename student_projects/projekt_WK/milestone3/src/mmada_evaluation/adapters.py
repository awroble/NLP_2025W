"""
MMaDA Evaluation Framework Adapter

This module provides an adapter class that makes MMaDA diffusion model
compatible with the LSB (LLM Safety Benchmark) evaluation framework.

Authors: Based on work by Kisiel, Kosakowski, Franczak, and Koniecko
Institution: Warsaw University of Technology, NLP Course Winter 2025
"""

import sys
import os
import torch
from pathlib import Path
from typing import Optional, List
import logging
from transformers import AutoTokenizer, AutoModelForCausalLM

# Add NLP_2025W to path for importing LSBEvaluator
NLP_2025W_PATH = Path(__file__).parent.parent.parent / "NLP_2025W" / "Kisiel_Kosakowski_Franczak_Koniecko"
if NLP_2025W_PATH.exists():
    sys.path.insert(0, str(NLP_2025W_PATH))
    from evaluate import LSBEvaluator, EvaluationConfig
else:
    raise ImportError(
        f"NLP_2025W evaluation framework not found at {NLP_2025W_PATH}. "
        "Please ensure the repository structure is correct."
    )


from inference import MMaDAInference
from utils import setup_logging


class MMaDAEvaluator(LSBEvaluator):
    """
    Adapter for MMaDA-8B-MixCoT model evaluation.

    This class makes MMaDA compatible with the LSB evaluation framework by
    inheriting from LSBEvaluator and overriding only the generation methods.

    All evaluation logic (refusal detection, harmful content detection, metrics)
    is inherited from LSBEvaluator without modification.

    Example usage:
        evaluator = create_mmada_evaluator(
            model_path="Gen-Verse/MMaDA-8B-MixCoT",
            device="cuda",
            steps=128,
            gen_length=128,
            judge_model_name="Qwen/Qwen2.5-3B-Instruct"
        )

        evaluator.run_evaluation(
            prompts_files=["../../data/prompts_health.json"],
            output_dir="results/mmada"
        )
    """

    def __init__(
        self,
        inference_wrapper: MMaDAInference,
        judge_model_name: Optional[str] = None,
        config: Optional[EvaluationConfig] = None,
        batch_size: int = 1,
        checkpoint_interval: int = 50,
        prompt_batch_size: int = 100,
        log_level: str = "INFO",
        log_file: Optional[str] = None,
        steps: Optional[int] = None,
        gen_length: Optional[int] = None,
        block_length: Optional[int] = None,
        temperature: Optional[float] = None
    ):
        """
        Initialize MMaDA evaluator.

        Args:
            inference_wrapper: Initialized MMaDAInference instance
            judge_model_name: Optional judge model (recommended)
            config: Optional EvaluationConfig object
            batch_size: Batch size (note: diffusion models process sequentially)
            checkpoint_interval: Checkpoint frequency
            prompt_batch_size: Prompts per batch
            log_level: Logging level
            log_file: Log file path
            steps: Override diffusion sampling steps
            gen_length: Override generation length
            block_length: Override block length
            temperature: Override sampling temperature
        """
        if not isinstance(inference_wrapper, MMaDAInference):
            raise TypeError(f"inference_wrapper must be MMaDAInference, got {type(inference_wrapper)}")

        self.inference_wrapper = inference_wrapper

        self.diffusion_params = {
            'steps': steps,
            'gen_length': gen_length,
            'block_length': block_length,
            'temperature': temperature
        }

        self.diffusion_params = {k: v for k, v in self.diffusion_params.items() if v is not None}


        self.model_name = inference_wrapper.model_path
        self.judge_model_name = judge_model_name
        self.device = inference_wrapper.device


        self.model = inference_wrapper.model


        if config is not None:
            if not isinstance(config, EvaluationConfig):
                raise TypeError(f"config must be an EvaluationConfig instance, got {type(config).__name__}")
            self.config = config
        else:

            self.config = EvaluationConfig(
                batch_size=batch_size,
                checkpoint_interval=checkpoint_interval,
                prompt_batch_size=prompt_batch_size
            )


        self.max_new_tokens = self.config.max_new_tokens
        self.temperature = self.config.temperature
        self.top_p = self.config.top_p
        self.batch_size = self.config.batch_size
        self.checkpoint_interval = self.config.checkpoint_interval
        self.prompt_batch_size = self.config.prompt_batch_size


        setup_logging(log_level=log_level.upper(), log_file=log_file)
        self.logger = logging.getLogger(self.__class__.__name__)

        self.logger.info(f"Initializing {self.__class__.__name__} for {self.model_name}")
        self.logger.info(f"Using device: {self.device}")
        self.logger.info(f"Diffusion parameters: {self.diffusion_params}")


        self.tokenizer = inference_wrapper.tokenizer


        self._embedding_cache = {}
        self._use_semantic_detection = True


        self.judge_model = None
        self.judge_tokenizer = None
        if judge_model_name:
            self.logger.info(f"Loading judge model: {judge_model_name}")

            self.judge_tokenizer = AutoTokenizer.from_pretrained(
                judge_model_name,
                trust_remote_code=True
            )


            if self.judge_tokenizer.pad_token is None:
                self.judge_tokenizer.pad_token = self.judge_tokenizer.eos_token


            if str(self.device) == "cuda":
                self.judge_model = AutoModelForCausalLM.from_pretrained(
                    judge_model_name,
                    torch_dtype=torch.float16,
                    device_map="auto",
                    trust_remote_code=True
                )
            else:
                self.judge_model = AutoModelForCausalLM.from_pretrained(
                    judge_model_name,
                    torch_dtype=torch.float16 if str(self.device) == "mps" else torch.float32,
                    device_map=None,
                    trust_remote_code=True
                )
                self.judge_model = self.judge_model.to(self.device)

            self.judge_model.eval()
            self.logger.info("Judge model loaded successfully!")
        else:
            self.logger.warning("No judge model specified. Using same model for evaluation (not recommended).")

    def generate_response(self, formatted_prompt: str) -> str:
        """
        Generate response using MMaDA diffusion model.

        This overrides LSBEvaluator's generate_response to use diffusion generation
        instead of standard HuggingFace model.generate().

        Args:
            formatted_prompt: The formatted input prompt

        Returns:
            Generated response string
        """
        try:

            response = self.inference_wrapper.generate_response(
                prompt=formatted_prompt,
                add_chat_template=False,
                **self.diffusion_params
            )
            return response
        except Exception as e:
            self.logger.error(f"Generation failed: {type(e).__name__}: {str(e)}")
            return f"[ERROR: Generation failed - {type(e).__name__}]"

    def generate_responses_batch(self, formatted_prompts: List[str]) -> List[str]:
        """
        Generate responses for multiple prompts.

        Note: Diffusion models currently process sequentially, but this maintains
        API compatibility with LSBEvaluator.

        Args:
            formatted_prompts: List of formatted prompt strings

        Returns:
            List of generated response strings
        """
        return self.inference_wrapper.generate_batch(
            prompts=formatted_prompts,
            add_chat_template=False,
            **self.diffusion_params
        )


def create_mmada_evaluator(
    model_path: str = "Gen-Verse/MMaDA-8B-MixCoT",
    device: Optional[str] = None,
    steps: int = 128,
    gen_length: int = 128,
    block_length: int = 32,
    temperature: float = 0.0,
    judge_model_name: Optional[str] = None,
    log_level: str = "INFO"
) -> MMaDAEvaluator:
    """
    Convenience function to create a MMaDA evaluator with one call.

    Args:
        model_path: MMaDA model path
        device: Device to use
        steps: Diffusion sampling steps
        gen_length: Generation length
        block_length: Block length for semi-autoregressive
        temperature: Sampling temperature
        judge_model_name: Optional judge model
        log_level: Logging level

    Returns:
        Configured MMaDAEvaluator instance
    """
    inference = MMaDAInference(
        model_path=model_path,
        device=device,
        steps=steps,
        gen_length=gen_length,
        block_length=block_length,
        temperature=temperature,
        log_level=log_level
    )

    return MMaDAEvaluator(
        inference_wrapper=inference,
        judge_model_name=judge_model_name,
        log_level=log_level
    )
