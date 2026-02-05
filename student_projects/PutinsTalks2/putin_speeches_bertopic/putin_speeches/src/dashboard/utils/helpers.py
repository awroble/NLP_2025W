"""
Utility helper functions for data processing and formatting.
"""

import ast
import pandas as pd
from typing import List, Tuple, Any, Union

from config import DOCUMENT_PREVIEW_LENGTH


def format_date(date_value: Any, output_format: str = "%B %d, %Y") -> str:
    """
    Format a date value to a human-readable string.
    
    Args:
        date_value: Date string, datetime, or None
        output_format: strftime format string
    
    Returns:
        Formatted date string or "Unknown date"
    """
    if not date_value or date_value == "Unknown date":
        return "Unknown date"
    
    if isinstance(date_value, str):
        try:
            date_obj = pd.to_datetime(date_value)
            return date_obj.strftime(output_format)
        except (ValueError, TypeError):
            return date_value
    
    try:
        return date_value.strftime(output_format)
    except (ValueError, AttributeError):
        return str(date_value)


def truncate_text(text: str, max_length: int = DOCUMENT_PREVIEW_LENGTH) -> str:
    """
    Truncate text to a maximum length with ellipsis.
    
    Args:
        text: Text to truncate
        max_length: Maximum length before truncation
    
    Returns:
        Truncated text with "..." if needed
    """
    if len(text) <= max_length:
        return text
    return text[:max_length] + "..."


def parse_topic_words(
    top_n_words: Union[str, List[Tuple[str, float]], None]
) -> List[Tuple[str, float]]:
    """
    Parse topic words from various input formats.
    
    Args:
        top_n_words: String representation, list of tuples, or None
    
    Returns:
        List of (word, score) tuples
    """
    if not top_n_words:
        return []
    
    if isinstance(top_n_words, str):
        try:
            return ast.literal_eval(top_n_words)
        except (ValueError, SyntaxError):
            return []
    
    return top_n_words


def parse_representation(representation: Union[str, List[str], None]) -> List[str]:
    """
    Parse topic representation words.
    
    Args:
        representation: String representation, list of words, or None
    
    Returns:
        List of representation words
    """
    if not representation:
        return []
    
    if isinstance(representation, str):
        try:
            return ast.literal_eval(representation)
        except (ValueError, SyntaxError):
            return []
    
    return representation


def calculate_text_statistics(text: str) -> dict:
    """
    Calculate various statistics for a text document.
    
    Args:
        text: Document text
    
    Returns:
        Dictionary with character_count, word_count, sentence_count, avg_word_length
    """
    words = text.split()
    word_count = len(words)
    char_count = len(text)
    sentence_count = (
        text.count('.') + text.count('!') + text.count('?')
    )
    avg_word_length = (
        sum(len(w) for w in words) / max(word_count, 1)
    )
    
    return {
        "character_count": char_count,
        "word_count": word_count,
        "sentence_count": sentence_count,
        "avg_word_length": round(avg_word_length, 1)
    }
