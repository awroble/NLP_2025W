"""
- Text models (singleturn vs multiturn separately)
- Multimodal models (separate from text)
"""

import json
import os
from pathlib import Path
from collections import defaultdict

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
from scipy import stats

BASE = Path(__file__).resolve().parent.parent.parent

EDA_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = EDA_DIR / "plots"
OUTPUT_DIR.mkdir(exist_ok=True)


########### Loading ######################################################################

def load_json(path):
    """Load JSON file"""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def flatten_entry(entry, model_name=None, modality=None):
    """Flatten a single entry with optional model name and modality"""
    prompt_text = " ".join([p["text"] for p in entry.get("prompt", [])])
    assistant_response = next((t["content"] for t in entry.get("conversation", []) if t["role"] == "assistant"), "")
    meta = entry.get("metadata", {})
    slots = meta.get("slots_filled", {})

    # Handle verdict - filter out invalid values (only 0 or 1 are valid)
    verdict_raw = entry.get("judge_verdict", 0)
    try:
        verdict = int(verdict_raw)
        # Validate: only 0 or 1 are valid
        if verdict not in [0, 1]:
            return None  # Invalid verdict, skip this entry
    except (ValueError, TypeError):
        # If it's a string that can't be converted, try to extract number
        if isinstance(verdict_raw, str):
            if "1" in verdict_raw and "0" not in verdict_raw:
                verdict = 1
            elif "0" in verdict_raw and "1" not in verdict_raw:
                verdict = 0
            else:
                return None  # Can't determine, skip
        else:
            return None  # Invalid, skip

    result = {
        "id": entry.get("id"),
        "prompt_text": prompt_text,
        "assistant_response": assistant_response,
        "verdict": verdict,
        **slots,
        **meta
    }
    
    if model_name:
        result["model"] = model_name
    if modality:
        result["modality"] = modality
    
    return result


def load_text_models_data():
    """
    Returns: dict with keys 'singleturn' and 'multiturn', each containing model data
    """
    text_models = ['Gemma2', 'Phi3', 'Qwen2']
    categories = ["sensitive-data-extraction", "social-engineering", "mental-physical-health"]
    
    data = {
        'singleturn': defaultdict(list),
        'multiturn': defaultdict(list)
    }
    
    for category in categories:
        for modality in ['singleturn', 'multiturn']:
            for model in text_models:
                filename = f"{category}-{modality}-{model}.json"
                filepath = os.path.join(BASE, "data", "eval", filename)
                
                if os.path.exists(filepath):
                    entries = load_json(filepath)
                    for entry in entries:
                        flattened = flatten_entry(entry, model_name=model, modality=modality)
                        if flattened is not None:  # Only add valid entries
                            data[modality][model].append(flattened)
    
    return data


def load_multimodal_models_data():
    """
    Returns: dict with model names as keys
    """
    multimodal_models = ['Llava1.5', 'SmolVLM']
    categories = ["sensitive-data-extraction", "social-engineering", "mental-physical-health"]
    
    data = defaultdict(list)
    
    for category in categories:
        for model in multimodal_models:
            filename = f"{category}-multimodal-{model}.json"
            filepath = os.path.join(BASE, "data", "eval", filename)
            
            if os.path.exists(filepath):
                entries = load_json(filepath)
                for entry in entries:
                    flattened = flatten_entry(entry, model_name=model, modality='multimodal')
                    if flattened is not None:  # Only add valid entries
                        data[model].append(flattened)
    
    return data


############## Linguistic Features ########################################################

def compute_linguistic_features(df):
    """Compute linguistic features for prompts and responses"""
    df["prompt_length_chars"] = df["prompt_text"].str.len()
    df["prompt_length_tokens"] = df["prompt_text"].str.split().str.len()
    df["contains_please"] = df["prompt_text"].str.contains(r"\bplease\b", case=False, na=False)
    df["contains_urgent"] = df["prompt_text"].str.contains(r"\burgent|immediately|asap|now\b", case=False, na=False)
    df["contains_emotional"] = df["prompt_text"].str.contains(r"smartest|trust|only you|help me|helpful", case=False, na=False)
    df["response_length_tokens"] = df["assistant_response"].str.split().str.len()
    df["explicit_refusal"] = df["assistant_response"].str.contains(
        r"\b(cannot assist|cannot help|not able|sorry|cannot provide)\b", case=False, na=False)
    df["gives_example"] = df["assistant_response"].str.contains(r"(example|script|template|scenario)", case=False, na=False)
    return df


############## Analysis Functions ###################################################
def analyze_over_refusal(df, modality_name=""):
    """
    Compute over-refusal rate:
    verdict = 0 AND variant = 'safe'
    """
    if "variant" not in df.columns:
        print(f"[WARN] No 'variant' column found for {modality_name}. Skipping over-refusal analysis.")
        return None

    safe_df = df[df["variant"] == "safe"]
    if safe_df.empty:
        print(f"[INFO] No safe prompts found for {modality_name}.")
        return None

    over_refusal_rate = (safe_df["verdict"] == 0).mean() * 100
    total_safe = len(safe_df)

    print(f"\n--- OVER-REFUSAL ANALYSIS ({modality_name}) ---")
    print(f"Total safe prompts: {total_safe}")
    print(f"Over-refusal rate (verdict=0 on safe prompts): {over_refusal_rate:.2f}%")

    # Optional: by model
    if "model" in df.columns:
        by_model = safe_df.groupby("model")["verdict"].apply(lambda x: (x == 0).mean() * 100)
        print("\nOver-refusal by model:")
        print(by_model.round(2))

    return {
        "total_safe": total_safe,
        "over_refusal_rate": over_refusal_rate,
        "by_model": by_model if "model" in df.columns else None
    }

def analyze_text_models_singleturn(singleturn_data):
    """
    Analyze text models on singleturn tasks
    """
    print("TEXT MODELS - SINGLETURN ANALYSIS")
    
    # Combine data for singleturn
    all_data = []
    for model, entries in singleturn_data.items():
        all_data.extend(entries)
    
    df = pd.DataFrame(all_data)
    df = compute_linguistic_features(df)
    
    print(f"\nTotal valid entries: {len(df)}")
    print(f"Models: {sorted(df['model'].unique())}")
    print(f"Categories: {sorted(df['category'].unique())}")
    
    # Overall pass rate by model
    print("\n--- PASS RATE BY MODEL (SINGLETURN) ---")
    pass_rate = df.groupby('model')['verdict'].agg(['mean', 'count'])
    pass_rate['mean'] *= 100  # Convert to percentage
    pass_rate.columns = ['Pass Rate (%)', 'Count']
    pass_rate = pass_rate.sort_values('Pass Rate (%)')
    print(pass_rate)
    
    # Plot
    plt.figure(figsize=(10, 6))
    ax = pass_rate['Pass Rate (%)'].plot(kind='barh', color=sns.color_palette("rocket_r", len(pass_rate)))
    plt.xlabel('Pass Rate (%)', fontsize=12)
    plt.ylabel('Model', fontsize=12)
    plt.title('Text Models - Singleturn Pass Rate', fontsize=14, weight='bold')
    plt.tight_layout()
    plt.xlim(0,100)
    plt.savefig(OUTPUT_DIR / "text_singleturn_pass_rate.png", dpi=300)
    plt.show()
    
    # Pass rate by category
    print("\n--- PASS RATE BY CATEGORY AND MODEL (SINGLETURN) ---")
    category_comp = df.groupby(['category', 'model'])['verdict'].mean() * 100
    category_comp = category_comp.unstack()
    print(category_comp.round(2))
    
    # Plot
    fig, ax = plt.subplots(figsize=(12, 7))
    category_comp.plot(kind='bar', ax=ax)
    plt.xlabel('Category', fontsize=12)
    plt.ylabel('Pass Rate (%)', fontsize=12)
    plt.title('Text Models - Singleturn Pass Rate by Category', fontsize=14, weight='bold')
    plt.legend(title='Model')
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.3)
    plt.ylim(0,100)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "text_singleturn_by_category.png", dpi=300)
    plt.show()
    
    return df, pass_rate, category_comp


def analyze_text_models_multiturn(multiturn_data):
    """
    Analyze text models on multiturn tasks
    """
    print("TEXT MODELS - MULTITURN ANALYSIS")
    
    # Combine data for multiturn
    all_data = []
    for model, entries in multiturn_data.items():
        all_data.extend(entries)
    
    df = pd.DataFrame(all_data)
    df = compute_linguistic_features(df)
    
    print(f"\nTotal valid entries: {len(df)}")
    print(f"Models: {sorted(df['model'].unique())}")
    print(f"Categories: {sorted(df['category'].unique())}")
    
    # Overall pass rate by model
    print("\n--- PASS RATE BY MODEL (MULTITURN) ---")
    pass_rate = df.groupby('model')['verdict'].agg(['mean', 'count'])
    pass_rate['mean'] *= 100
    pass_rate.columns = ['Pass Rate (%)', 'Count']
    pass_rate = pass_rate.sort_values('Pass Rate (%)')
    print(pass_rate)
    
    # Plot
    plt.figure(figsize=(10, 6))
    ax = pass_rate['Pass Rate (%)'].plot(kind='barh', color=sns.color_palette("viridis_r", len(pass_rate)))
    plt.xlabel('Pass Rate (%)', fontsize=12)
    plt.ylabel('Model', fontsize=12)
    plt.title('Text Models - Multiturn Pass Rate', fontsize=14, weight='bold')
    plt.tight_layout()
    plt.xlim(0,100)
    plt.savefig(OUTPUT_DIR / "text_multiturn_pass_rate.png", dpi=300)
    plt.show()
    
    # Pass rate by category
    print("\n--- PASS RATE BY CATEGORY AND MODEL (MULTITURN) ---")
    category_comp = df.groupby(['category', 'model'])['verdict'].mean() * 100
    category_comp = category_comp.unstack()
    print(category_comp.round(2))
    
    # Plot
    fig, ax = plt.subplots(figsize=(12, 7))
    category_comp.plot(kind='bar', ax=ax)
    plt.xlabel('Category', fontsize=12)
    plt.ylabel('Pass Rate (%)', fontsize=12)
    plt.title('Text Models - Multiturn Pass Rate by Category', fontsize=14, weight='bold')
    plt.legend(title='Model')
    plt.ylim(0,100)
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "text_multiturn_by_category.png", dpi=300)
    plt.show()
    
    return df, pass_rate, category_comp


def analyze_multimodal_models(multimodal_data):
    """
    Analyze multimodal models
    """
    print("MULTIMODAL MODELS ANALYSIS")
    
    # Combine data
    all_data = []
    for model, entries in multimodal_data.items():
        all_data.extend(entries)
    
    df = pd.DataFrame(all_data)
    df = compute_linguistic_features(df)
    
    print(f"\nTotal valid entries: {len(df)}")
    print(f"Models: {sorted(df['model'].unique())}")
    print(f"Categories: {sorted(df['category'].unique())}")
    
    # Overall pass rate by model
    print("\n--- PASS RATE BY MODEL (MULTIMODAL) ---")
    pass_rate = df.groupby('model')['verdict'].agg(['mean', 'count'])
    pass_rate['mean'] *= 100
    pass_rate.columns = ['Pass Rate (%)', 'Count']
    pass_rate = pass_rate.sort_values('Pass Rate (%)')
    print(pass_rate)
    
    # Plot
    plt.figure(figsize=(10, 6))
    ax = pass_rate['Pass Rate (%)'].plot(kind='barh', color=sns.color_palette("crest_r", len(pass_rate)))
    plt.xlabel('Pass Rate (%)', fontsize=12)
    plt.ylabel('Model', fontsize=12)
    plt.xlim(0, 100)
    plt.title('Multimodal Models Pass Rate', fontsize=14, weight='bold')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "multimodal_pass_rate.png", dpi=300)
    plt.show()
    
    # Pass rate by category
    print("\n--- PASS RATE BY CATEGORY AND MODEL (MULTIMODAL) ---")
    category_comp = df.groupby(['category', 'model'])['verdict'].mean() * 100
    category_comp = category_comp.unstack()
    print(category_comp.round(2))
    
    # Plot
    fig, ax = plt.subplots(figsize=(12, 7))
    category_comp.plot(kind='bar', ax=ax)
    plt.xlabel('Category', fontsize=12)
    plt.ylabel('Pass Rate (%)', fontsize=12)
    plt.title('Multimodal Models Pass Rate by Category', fontsize=14, weight='bold')
    plt.legend(title='Model')
    plt.xticks(rotation=45, ha='right')
    plt.ylim(0, 100)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "multimodal_by_category.png", dpi=300)
    plt.show()
    
    return df, pass_rate, category_comp


def compare_categories(text_singleturn_df, text_multiturn_df, multimodal_df):
    """
    Compare categories across all data
    """
    print("CATEGORY COMPARISON ACROSS ALL MODELS AND MODALITIES")
    
    # Combine all data
    all_df = pd.concat([text_singleturn_df, text_multiturn_df, multimodal_df], ignore_index=True)
    
    # Pass rate by category
    print("\n--- OVERALL PASS RATE BY CATEGORY ---")
    category_pass_rate = all_df.groupby('category')['verdict'].agg(['mean', 'count'])
    category_pass_rate['mean'] *= 100
    category_pass_rate.columns = ['Pass Rate (%)', 'Count']
    category_pass_rate = category_pass_rate.sort_values('Pass Rate (%)', ascending=False)
    print(category_pass_rate)
    
    # Plot
    plt.figure(figsize=(10, 6))
    ax = category_pass_rate['Pass Rate (%)'].plot(kind='barh', color=sns.color_palette("flare", len(category_pass_rate)))
    plt.xlabel('Pass Rate (%)', fontsize=12)
    plt.ylabel('Category', fontsize=12)
    plt.title('Pass Rate by Category (All Models)', fontsize=14, weight='bold')
    plt.tight_layout()
    plt.xlim(0, 100)
    plt.savefig(OUTPUT_DIR / "category_comparison.png", dpi=300)
    plt.show()
    
    return category_pass_rate


def compare_singleturn_vs_multiturn(text_data):
    """
    Compare singleturn vs multiturn for text models
    """
    print("SINGLETURN VS MULTITURN COMPARISON (TEXT MODELS)")
    
    # Combine singleturn and multiturn
    singleturn_data = []
    for model, entries in text_data['singleturn'].items():
        singleturn_data.extend(entries)
    
    multiturn_data = []
    for model, entries in text_data['multiturn'].items():
        multiturn_data.extend(entries)
    
    st_df = pd.DataFrame(singleturn_data)
    mt_df = pd.DataFrame(multiturn_data)
    
    # Pass rates
    st_pass_rate = st_df['verdict'].mean() * 100
    mt_pass_rate = mt_df['verdict'].mean() * 100
    
    print(f"\nSingleturn pass rate: {st_pass_rate:.2f}%")
    print(f"Multiturn pass rate: {mt_pass_rate:.2f}%")
    print(f"Difference: {mt_pass_rate - st_pass_rate:+.2f}%")
    
    # By model
    print("\n--- BY MODEL ---")
    comparison = pd.DataFrame({
        'Singleturn': st_df.groupby('model')['verdict'].mean() * 100,
        'Multiturn': mt_df.groupby('model')['verdict'].mean() * 100
    })
    comparison['Difference'] = comparison['Multiturn'] - comparison['Singleturn']
    print(comparison.round(2))
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    comparison[['Singleturn', 'Multiturn']].plot(kind='bar', ax=ax)
    plt.xlabel('Model', fontsize=12)
    plt.ylabel('Pass Rate (%)', fontsize=12)
    plt.title('Singleturn vs Multiturn Comparison', fontsize=14, weight='bold')
    plt.legend(title='Modality')
    plt.ylim(0, 100)
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "singleturn_vs_multiturn.png", dpi=300)
    plt.show()
    
    return comparison


def analyze_interesting_patterns(text_singleturn_df, text_multiturn_df, multimodal_df):
    """
    Find interesting patterns in the data
    """
    print("INTERESTING PATTERNS AND INSIGHTS")
    
    # Combine all data for some analyses
    all_df = pd.concat([text_singleturn_df, text_multiturn_df, multimodal_df], ignore_index=True)
    
    # 1. Response length vs safety
    print("\n--- RESPONSE LENGTH VS SAFETY ---")
    response_length_data = []
    for name, df in [("Singleturn", text_singleturn_df), ("Multiturn", text_multiturn_df), ("Multimodal", multimodal_df)]:
        safe_length = df[df['verdict'] == 0]['response_length_tokens'].mean()
        unsafe_length = df[df['verdict'] == 1]['response_length_tokens'].mean()
        print(f"{name}:")
        print(f"  Safe responses: {safe_length:.1f} tokens")
        print(f"  Unsafe responses: {unsafe_length:.1f} tokens")
        print(f"  Difference: {unsafe_length - safe_length:+.1f} tokens")
        
        response_length_data.append({
            'Modality': name,
            'Safe (verdict=0)': safe_length,
            'Unsafe (verdict=1)': unsafe_length
        })
    
    # Plot response length comparison
    response_df = pd.DataFrame(response_length_data).set_index('Modality')
    fig, ax = plt.subplots(figsize=(10, 6))
    response_df.plot(kind='bar', ax=ax, color=['#2ecc71', '#e74c3c'])
    plt.xlabel('Modality', fontsize=12)
    plt.ylabel('Average Response Length (tokens)', fontsize=12)
    plt.title('Response Length vs Safety', fontsize=14, weight='bold')
    plt.legend(title='Response Type')
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "response_length_vs_safety.png", dpi=300)
    plt.show()
    
    # 2. Prompt length sensitivity
    print("\n--- PROMPT LENGTH SENSITIVITY ---")
    corr_data = []
    for model in sorted(all_df['model'].unique()):
        model_df = all_df[all_df['model'] == model]
        corr = model_df[['prompt_length_tokens', 'verdict']].corr().iloc[0, 1]
        print(f"{model}: correlation = {corr:.3f}")
        corr_data.append({'Model': model, 'Correlation': corr})
    
    # Plot correlations
    corr_df = pd.DataFrame(corr_data)
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ['#e74c3c' if c < 0 else '#3498db' for c in corr_df['Correlation']]
    ax.barh(corr_df['Model'], corr_df['Correlation'], color=colors)
    plt.xlabel('Correlation (Prompt Length ↔ Vulnerability)', fontsize=12)
    plt.ylabel('Model', fontsize=12)
    plt.title('Prompt Length Sensitivity by Model', fontsize=14, weight='bold')
    plt.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "prompt_length_sensitivity.png", dpi=300)
    plt.show()
    
    # 3. Emotional manipulation effectiveness
    print("\n--- EMOTIONAL MANIPULATION EFFECTIVENESS ---")
    manipulation_data = []
    for feature in ['contains_please', 'contains_urgent', 'contains_emotional']:
        with_feature = all_df[all_df[feature]]['verdict'].mean() * 100
        without_feature = all_df[~all_df[feature]]['verdict'].mean() * 100
        diff = with_feature - without_feature
        print(f"{feature}:")
        print(f"  With: {with_feature:.1f}%")
        print(f"  Without: {without_feature:.1f}%")
        print(f"  Difference: {diff:+.1f}%")
        
        feature_name = feature.replace('contains_', '').replace('_', ' ').title()
        manipulation_data.append({
            'Feature': feature_name,
            'With Feature': with_feature,
            'Without Feature': without_feature
        })
    
    # Plot manipulation effectiveness
    manip_df = pd.DataFrame(manipulation_data).set_index('Feature')
    fig, ax = plt.subplots(figsize=(10, 6))
    manip_df.plot(kind='bar', ax=ax, color=['#e74c3c', '#2ecc71'])
    plt.xlabel('Prompt Feature', fontsize=12)
    plt.ylabel('Pass Rate (%)', fontsize=12)
    plt.title('Emotional Manipulation Effectiveness', fontsize=14, weight='bold')
    plt.legend(title='Presence')
    plt.ylim(0, 100)
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "emotional_manipulation.png", dpi=300)
    plt.show()
    
    # 4. Refusal patterns
    print("\n--- EXPLICIT REFUSAL EFFECTIVENESS ---")
    refusal_data = []
    for name, df in [("Singleturn", text_singleturn_df), ("Multiturn", text_multiturn_df), ("Multimodal", multimodal_df)]:
        refusal_rate = df['explicit_refusal'].mean() * 100
        safe_given_refusal = df[df['explicit_refusal']]['verdict'].mean() * 100
        safe_given_no_refusal = df[~df['explicit_refusal']]['verdict'].mean() * 100
        print(f"{name}:")
        print(f"  Refusal rate: {refusal_rate:.1f}%")
        print(f"  Pass rate when refusing: {safe_given_refusal:.1f}%")
        print(f"  Pass rate when not refusing: {safe_given_no_refusal:.1f}%")
        
        refusal_data.append({
            'Modality': name,
            'When Refusing': safe_given_refusal,
            'When Not Refusing': safe_given_no_refusal
        })
    
    # Plot refusal effectiveness
    refusal_df = pd.DataFrame(refusal_data).set_index('Modality')
    fig, ax = plt.subplots(figsize=(10, 6))
    refusal_df.plot(kind='bar', ax=ax, color=['#9b59b6', '#3498db'])
    plt.xlabel('Modality', fontsize=12)
    plt.ylabel('Pass Rate (%)', fontsize=12)
    plt.title('Explicit Refusal Effectiveness', fontsize=14, weight='bold')
    plt.legend(title='Refusal Strategy')
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.3)
    plt.ylim(0, 100)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "refusal_effectiveness.png", dpi=300)
    plt.show()


############## Main Execution ###################################################################

def main_model_comparison():
    """
    Main function to run all analyses
    """
    print("NLP SAFETY BENCHMARK - EXTENDED EDA")
    print("Separate analysis for text (singleturn/multiturn) and multimodal models")
    
    # Set plot style
    sns.set_theme(style="whitegrid", context="talk", font_scale=1.0)
    plt.rcParams.update({
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "figure.figsize": (12, 7)
    })
    
    # Load data
    print("\nLoading text models data...")
    text_data = load_text_models_data()
    
    print("Loading multimodal models data...")
    multimodal_data = load_multimodal_models_data()
    
    # Print summary
    print("\n--- DATA SUMMARY ---")
    print(f"Text models (singleturn):")
    for model, entries in text_data['singleturn'].items():
        print(f"  {model}: {len(entries)} entries")
    
    print(f"\nText models (multiturn):")
    for model, entries in text_data['multiturn'].items():
        print(f"  {model}: {len(entries)} entries")
    
    print(f"\nMultimodal models:")
    for model, entries in multimodal_data.items():
        print(f"  {model}: {len(entries)} entries")
    
    # Run analyses
    st_df, st_pass, st_cat = analyze_text_models_singleturn(text_data['singleturn'])
    mt_df, mt_pass, mt_cat = analyze_text_models_multiturn(text_data['multiturn'])
    mm_df, mm_pass, mm_cat = analyze_multimodal_models(multimodal_data)

    # Over-refusal analysis
    analyze_over_refusal(st_df, modality_name="Single-turn Text")
    analyze_over_refusal(mm_df, modality_name="Multimodal")
    
    # Comparisons
    cat_comp = compare_categories(st_df, mt_df, mm_df)
    st_vs_mt = compare_singleturn_vs_multiturn(text_data)
    
    # Interesting patterns
    analyze_interesting_patterns(st_df, mt_df, mm_df)
        
    return {
        'text_singleturn': {'df': st_df, 'pass_rate': st_pass, 'category': st_cat},
        'text_multiturn': {'df': mt_df, 'pass_rate': mt_pass, 'category': mt_cat},
        'multimodal': {'df': mm_df, 'pass_rate': mm_pass, 'category': mm_cat},
        'category_comparison': cat_comp,
        'singleturn_vs_multiturn': st_vs_mt
    }


if __name__ == "__main__":
    results = main_model_comparison()