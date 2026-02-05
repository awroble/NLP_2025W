# LLM Safety Evaluation

## Overview
The goal of this project is to evaluate Large Language Models (LLMs) against safety risks such as bias, jailbreak roleplay, and multi-turn persuasion.
We create a diverse prompt dataset containing neutral and potentially harmful prompts, covering single-turn, multi-turn, and multimodal interactions. 
Using this dataset, we test LLM responses, identify vulnerabilities, and analyze how manipulative techniques affect model behavior.

## Project Structure

```
config/                  # YAML configuration files for models and settings
data/
├─ img/                  # Images for multimodal prompts
├─ prompts/              # Organized prompts (bias, jailbreak, multiturn) with detailed description in logic_prompts.txt
data_generation/         # Scripts and notebooks for generating prompts
evaluations/             # Files used to process results, scores, time and memory measurements
judge_prompts/           # Judge prompts generation
results/                 # Model outputs organized by model
src/
├─ judge/                # Judge model class
├─ loaders/              # Model and prompt loading utilities
├─ runners/              # Execution scripts for running generation
├─ utils/                # Utility functions
zeroshot/                # Evaluation testing
main.py                  # Entry point for running experiments
requirements.txt         # Project dependencies
README.md                # Project documentation
```

## Usage

1. Install dependencies:
```bash
pip install -r requirements.txt
```
2. Configure models and generation settings in `config/models.yaml` and `config/settings.yaml`.

3. Run the main experiment:
```bash
python main.py
```

4. Generated outputs will be saved in the `results` directory.

