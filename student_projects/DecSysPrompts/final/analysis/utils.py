"""Utility functions for prompts analysis."""

import os
import json
from config import DATASET_PATHS
import pandas as pd

def read_prompt(file_path):
    """Read the prompt text from a file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        text = ''.join(lines)
    return text, lines

def read_statements_json(file_path):
    """Read the statements JSON from a file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        statements_json = json.load(f)
    return statements_json

def change_extension(file_path, new_extension = 'json'):
    """Change the file extension of a given file path."""
    if '.' not in file_path:
        return f"{file_path}.{new_extension}"
    base = '.'.join(file_path.split('.')[:-1])
    return f"{base}.{new_extension}"

def load_prompts(use_cleaned = True):
    """Load prompts and metadata into a DataFrame."""
    rows = []
    base_path = DATASET_PATHS['DATASET_CLEANED'] if use_cleaned else DATASET_PATHS['DATASET']
    print(f"Loading prompts from {'cleaned' if use_cleaned else 'original'} dataset at {base_path}...")

    for root, dirs, files in os.walk(base_path):
        if root == base_path:
            continue
        
        for file in files:
            file_path = os.path.join(root, file)

            # Extract provider from subdirectory name
            rel_path = os.path.relpath(file_path, base_path)
            path_parts = rel_path.split(os.sep)
            provider = path_parts[0]

            # Extract model from filename without extension
            model = os.path.splitext(file)[0]

            try:
                text, _ = read_prompt(file_path)
                rows.append({
                    'path': rel_path,
                    'provider': provider,
                    'model': model,
                    'version': '',
                    'release_date': '',
                    'text': text,
                })
            except Exception as e:
                print(f"Warning: Could not read {file_path}: {e}")

    return pd.DataFrame(rows)