# Jailbreak Research

This directory contains experiments on jailbreak attacks against large language models (LLMs), including both text-based and multimodal approaches. The project evaluates model robustness and defense mechanisms using standardized benchmarks.

## Overview

The research investigates:
- **Text-based jailbreaks**: Testing LLMs with harmful prompts from JailbreakBench
- **Multimodal jailbreaks**: Using image-centric prompts from VSCBench
- **Defense mechanisms**: Evaluating Llama Guard for prompt classification
- **Model behavior analysis**: Comparing responses across different models with and without system prompts

## Datasets

### Text Prompts: JailbreakBench (JBB-Behaviors)

Source: [Hugging Face - JailbreakBench/JBB-Behaviors](https://huggingface.co/datasets/JailbreakBench/JBB-Behaviors)

The dataset contains 200 examples of 100 different model behaviors across multiple risk categories. Each behavior includes:
- **Benign variant**: Legitimate request version
- **Harmful variant**: Jailbreak attempt version

Examples sourced from:
- AdvBench
- HarmBench
- Original JailbreakBench contributions

### Multimodal Prompts: VSCBench

Source: [GitHub - VSCBench](https://github.com/jiahuigeng/VSCBench/tree/main)

Visual Safety Challenge Benchmark for testing multimodal models with potentially harmful image-text combinations.

Dataset subset includes multiple prompts across 5 categories:
- Discrimination
- Drugs
- Illegal Activity
- Religion
- Violence

The subset od prompt we used is in file `vscbench_subset.csv`.

## File Structure

### Execution Scripts

- **`execute_prompts.py`**: Main script for testing text-based LLMs
  - Models tested: Qwen3:8b, Llama3.1:8b, Mistral:latest
  - Supports system prompt configuration

- **`execute_multimodal.py`**: Script for multimodal model testing
  - Model tested: LLaVA:7b
  - Processes image + text prompts from VSCBench

- **`check_llama_guard.py`**: Llama Guard evaluation script
  - Classifies prompts as safe/unsafe
  - Validates benign vs. harmful labeling accuracy

- **`extract_for_multi_modal.py`**: Data preprocessing utility
  - Converts Excel to CSV format
  - Extracts subset of 8 prompts per category
  - Prepares VSCBench data for experiments

- **`evaluate_inference_time.py`**: Inference time benchmark script
  - Tests all models: Qwen3:8b, Llama3.1:8b, Mistral:latest, LLaVA:7b, Llama-Guard3:8b
  - Measures inference time per prompt (default: 50 prompts per model)
  - Calculates statistics: mean, median, std, min/max
  - Outputs results to `inference_time_results/` directory

### Analysis Notebooks

- **`small_eda_and_baseline_results_analysis.ipynb`**: 
  - Exploratory data analysis of JailbreakBench dataset
  - Baseline results visualization
  - Model comparison across configurations

- **`defense_results_analysis.ipynb`**:
  - Llama Guard classification performance
  - False positive/negative analysis
  - Defense effectiveness evaluation
# Data Directories

- **`model_responses/`**: JSON files containing LLM responses
  - Format: `responses_{model}_{timestamp}_{config}.json`
  - Includes both baseline and system-prompted variants
  
- **`multimodal_responses/`**: LLaVA model responses
  - Text + image prompt responses
  - Baseline and system-prompted configurations

- **`prompts_evaluated/`**: Llama Guard classifications
  - Safety classifications for all prompts
  - Ground truth vs. predicted labels

- **`inference_time_results/`**: Inference time benchmarks
  - JSON files with timing statistics for all models
  - Format: `inference_times_{timestamp}.json`
  - Includes mean, median, std, min/max per model

## Results

Graded model responses are available on Google Drive:
https://drive.google.com/file/d/1xoO-WVASBrEvlJGN6YTAX2akUjT5hrIg/view?usp=sharing

## Notes

- All experiments use `SEED = 123` for reproducibility
- Progress is automatically saved at intervals to prevent data loss
