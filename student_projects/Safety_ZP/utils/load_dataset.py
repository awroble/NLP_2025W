from datasets import load_dataset


def load_hf_dataset(dataset_name: str, config_name: str = None):
    """
    Loads a HuggingFace dataset with optional configuration.

    Args:
        dataset_name (str): e.g. "walledai/HarmBench"
        config_name (str, optional): e.g. "contextual"

    Returns:
        DatasetDict or Dataset

    Raises:
        RuntimeError if dataset cannot be loaded
    """
    try:
        if config_name is not None:
            dataset = load_dataset(dataset_name, config_name)
        else:
            dataset = load_dataset(dataset_name)

        return dataset

    except Exception as e:
        raise RuntimeError(
            f"Failed to load dataset '{dataset_name}'"
            + (f" with config '{config_name}'" if config_name else "")
            + f": {e}"
        )