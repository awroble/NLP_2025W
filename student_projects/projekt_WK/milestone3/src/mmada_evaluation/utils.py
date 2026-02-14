"""
Shared utilities for diffusion LLM inference and evaluation.

Authors: Based on work by Kisiel, Kosakowski, Franczak, and Koniecko
Institution: Warsaw University of Technology, NLP Course Winter 2025
"""

import logging
import sys
from typing import Optional
from datetime import datetime


def setup_logging(
    log_level: str = "INFO",
    log_file: Optional[str] = None,
    name: Optional[str] = None
) -> logging.Logger:
    """
    Configure logging with both console and optional file handlers.

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional path to log file. If None, only console logging is used.
        name: Logger name. If None, returns root logger.

    Returns:
        Configured logger instance
    """
    # Convert string level to logging constant
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)

    # Create formatters
    detailed_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    simple_formatter = logging.Formatter(
        '%(levelname)s - %(message)s'
    )

    # Get logger
    logger = logging.getLogger(name)
    logger.setLevel(numeric_level)

    # Clear existing handlers
    logger.handlers.clear()

    # Console handler (simpler format for user-facing messages)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(numeric_level)
    console_handler.setFormatter(simple_formatter)
    logger.addHandler(console_handler)

    # File handler (detailed format for debugging)
    if log_file:
        try:
            import os
            # Create directory if it doesn't exist
            log_dir = os.path.dirname(log_file)
            if log_dir and not os.path.exists(log_dir):
                os.makedirs(log_dir, exist_ok=True)

            file_handler = logging.FileHandler(log_file, encoding='utf-8')
            file_handler.setLevel(logging.DEBUG)  # Always log everything to file
            file_handler.setFormatter(detailed_formatter)
            logger.addHandler(file_handler)
        except (IOError, OSError) as e:
            logger.warning(f"Failed to set up file logging to {log_file}: {e}")

    return logger


def validate_device(device: Optional[str]) -> str:
    """
    Validate and return the best available device.

    Args:
        device: Requested device ('cuda', 'mps', 'cpu', or None for auto)

    Returns:
        Valid device string
    """
    import torch

    if device is None:
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"

    device = device.lower()
    if device not in ["cuda", "mps", "cpu"]:
        raise ValueError(f"Invalid device '{device}'. Must be 'cuda', 'mps', 'cpu', or None")

    # Validate requested device is available
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available")
    if device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS requested but not available")

    return device


def format_chat_prompt(
    messages: list,
    tokenizer,
    fallback: bool = True
) -> str:
    """
    Format messages using tokenizer's chat template with fallback.

    Args:
        messages: List of message dicts with 'role' and 'content'
        tokenizer: HuggingFace tokenizer instance
        fallback: Whether to use simple fallback if chat template fails

    Returns:
        Formatted prompt string
    """
    try:
        formatted = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        return formatted
    except (AttributeError, KeyError, TypeError, ValueError) as e:
        if not fallback:
            raise

        # Simple fallback formatting
        formatted_parts = []
        for msg in messages:
            role = msg.get("role", "user").capitalize()
            content = msg.get("content", "")
            formatted_parts.append(f"{role}: {content}")
        formatted_parts.append("Assistant:")
        return "\n".join(formatted_parts)


def get_model_memory_estimate(model_size_billions: float, dtype_bits: int = 16) -> float:
    """
    Estimate memory requirements for a model.

    Args:
        model_size_billions: Model size in billions of parameters
        dtype_bits: Bits per parameter (16 for bfloat16/float16, 32 for float32)

    Returns:
        Estimated memory in GB
    """
    bytes_per_param = dtype_bits / 8
    bytes_total = model_size_billions * 1e9 * bytes_per_param
    gb_total = bytes_total / (1024**3)

    # Add ~20% overhead for activations, gradients, etc.
    return gb_total * 1.2


def truncate_text(text: str, max_length: int = 150, suffix: str = "...") -> str:
    """
    Truncate text to maximum length with suffix.

    Args:
        text: Text to truncate
        max_length: Maximum length
        suffix: Suffix to add if truncated

    Returns:
        Truncated text
    """
    if len(text) <= max_length:
        return text
    return text[:max_length - len(suffix)] + suffix


class GenerationStats:
    """Track generation statistics for performance monitoring."""

    def __init__(self):
        self.total_prompts = 0
        self.total_tokens_generated = 0
        self.total_time_seconds = 0.0
        self.errors = 0

    def add_generation(
        self,
        tokens_generated: int,
        time_seconds: float,
        error: bool = False
    ):
        """Record a generation event."""
        self.total_prompts += 1
        self.total_tokens_generated += tokens_generated
        self.total_time_seconds += time_seconds
        if error:
            self.errors += 1

    def get_summary(self) -> dict:
        """Get summary statistics."""
        avg_tokens = (
            self.total_tokens_generated / self.total_prompts
            if self.total_prompts > 0
            else 0
        )
        avg_time = (
            self.total_time_seconds / self.total_prompts
            if self.total_prompts > 0
            else 0
        )
        tokens_per_second = (
            self.total_tokens_generated / self.total_time_seconds
            if self.total_time_seconds > 0
            else 0
        )

        return {
            "total_prompts": self.total_prompts,
            "total_tokens_generated": self.total_tokens_generated,
            "total_time_seconds": self.total_time_seconds,
            "avg_tokens_per_prompt": avg_tokens,
            "avg_time_per_prompt": avg_time,
            "tokens_per_second": tokens_per_second,
            "errors": self.errors,
            "error_rate": self.errors / self.total_prompts if self.total_prompts > 0 else 0
        }

    def __repr__(self) -> str:
        stats = self.get_summary()
        return (
            f"GenerationStats(prompts={stats['total_prompts']}, "
            f"tokens={stats['total_tokens_generated']}, "
            f"time={stats['total_time_seconds']:.2f}s, "
            f"tok/s={stats['tokens_per_second']:.2f}, "
            f"errors={stats['errors']})"
        )
