"""
Data and model loading for the BERTopic Dashboard.

Replace the placeholders with your actual model and data loading code.
"""

import pandas as pd
from typing import Dict, List, Any, Optional
import pickle
import pathlib
import torch
import io
# =============================================================================
# PLACEHOLDER: Replace these with your actual model and data
# =============================================================================

# Uncomment and configure:
# from bertopic import BERTopic
# topic_model = BERTopic.load("path/to/your/model")

class CPUUnpickler(pickle.Unpickler):
    """Unpickler that loads PyTorch tensors to CPU."""
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu', weights_only=False)
        return super().find_class(module, name)

def load_pickle(filepath):
    with open(filepath, 'rb') as f:
        return CPUUnpickler(f).load()

# Path relative to this file's location
load_path = pathlib.Path(__file__).parent.parent.parent / "data" / "model_1"

topic_model = load_pickle(load_path / "topic_model.pickle")
list_of_speeches = load_pickle(load_path / "list_of_speeches.pickle")
putin_speeches = load_pickle(load_path / "putin_speeches.pickle")
topics_over_time = load_pickle(load_path / "topics_over_time.pickle")



def load_model(model_path: str) -> None:
    """
    Load the BERTopic model from the specified path.
    
    Args:
        model_path: Path to the saved BERTopic model
    """
    global topic_model
    from bertopic import BERTopic
    topic_model = BERTopic.load(model_path)


def load_documents(documents: List[str], metadata: Dict[int, Dict[str, Any]]) -> None:
    """
    Load documents and their metadata.
    
    Args:
        documents: List of document texts
        metadata: Dictionary mapping document indices to metadata dicts
    """
    global list_of_speeches, putin_speeches
    list_of_speeches = documents
    putin_speeches = metadata


def load_topics_over_time(df: pd.DataFrame) -> None:
    """
    Load the topics over time DataFrame.
    
    Args:
        df: DataFrame with Topic, Words, Frequency, Timestamp, Name columns
    """
    global topics_over_time
    topics_over_time = df


def get_document_metadata(doc_id: int, field: str, default: Any = None) -> Any:
    """
    Safely get metadata for a document.

    Args:
        doc_id: Document index (list index)
        field: Metadata field name (title, location, url, date)
        default: Default value if field not found

    Returns:
        The metadata value or default
    """
    try:
        if 0 <= doc_id < len(putin_speeches):
            return putin_speeches[doc_id].get(field, default)
        return default
    except (IndexError, KeyError, TypeError, AttributeError):
        return default


def get_topic_time_data(topic_no: int) -> pd.DataFrame:
    """
    Get time series data for a specific topic.
    
    Args:
        topic_no: Topic number
    
    Returns:
        Filtered and sorted DataFrame
    """
    return topics_over_time[
        topics_over_time["Topic"] == topic_no
    ].sort_values("Timestamp")


def get_all_topic_time_data(topic_numbers: List[int]) -> pd.DataFrame:
    """
    Get time series data for multiple topics.
    
    Args:
        topic_numbers: List of topic numbers
    
    Returns:
        Filtered DataFrame
    """
    return topics_over_time[topics_over_time["Topic"].isin(topic_numbers)]
