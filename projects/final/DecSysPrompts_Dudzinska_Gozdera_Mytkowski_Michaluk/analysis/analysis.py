"""Analyze prompts and compute various indicators."""

from datetime import datetime
import pandas as pd
from metadata import METADATA
from config import DATASET_PATHS
from utils import read_prompt, read_statements_json, change_extension
from indicators import S, NCtW, NStNC, CStA, StRS, CtRS, PRtRS, PHtRS, get_regulative_components
import textstat

def analyze_prompt(path, data):
    """Analyze a single prompt and return its indicators along with metadata."""
    
    prompt, _ = read_prompt(f"{DATASET_PATHS['DATASET']}{path}")
    prompt_no_code, _ = read_prompt(f"{DATASET_PATHS['DATASET_CLEANED']}{path}")
    statements_json = read_statements_json(f"{DATASET_PATHS['STATEMENTS_JSON']}{change_extension(path)}")

    return {
        'provider': data['provider'],
        'model': data['model'],
        'version': data['version'],
        'release_date': datetime.strptime(data['release_date'], '%d.%m.%Y'),
        'S': S(prompt),
        'NCtW': NCtW(prompt, prompt_no_code),
        'NStNC': NStNC(statements_json),
        'CStA': CStA(statements_json),
        'StRS': StRS(statements_json),
        'CtRS': CtRS(statements_json),
        'PRtRS': PRtRS(statements_json),
        'PHtRS': PHtRS(statements_json),
        'FRE': round(textstat.flesch_reading_ease(prompt), 1),
        'FKGL': round(textstat.flesch_kincaid_grade(prompt), 1),
        'regulative_components': get_regulative_components(statements_json)
    }

def analyze_all_prompts():
    """Analyze all prompts in the dataset and return a DataFrame with their indicators."""
    prompts_analysis = []
    for path, data in METADATA.items():
        analysis = analyze_prompt(path, data)
        prompts_analysis.append(analysis)

    return pd.DataFrame(prompts_analysis)
