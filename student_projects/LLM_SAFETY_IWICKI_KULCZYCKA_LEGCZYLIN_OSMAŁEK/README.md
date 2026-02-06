# NLP Safety Benchmark

## Overview

This project contains our LLM safety benchmark. We prepared a **scalable and reproducible method** for preparing large numbers of prompts for the purpose of testing LLM responses. We provide an evaluation method based on the **LLM-as-a-Judge** approach and user-provided **expected_behaviours** for specific scenarios.

Using a small set of templates that we created (and that users can extend), we can prepare hundreds of prompts for each category.

Our prompts also contain metadata, which can be useful for human-based validation.

---

## Prompt Categories

We currently work with the following prompt categories:

- `mental-physical-health`
- `sensitive-data-extraction`
- `social-engineering`

---

## Repository Contents

All necessary codes and logic is stored in the `src` folder.

Our prompts, results and data is stored in data folders
- `data/prompts` contains prompt datasets prepared using our methods  
- `data/eval` contains results of our own experiments  
- `data/eval_judge` contatins samples of judge verdicts with human verdicts
 - `data/judge_analysis` contain tables and plots from experiments

---

## Prompt Types

We divide prompts into the following types, as each requires different handling and evaluation strategies:

- **Single-turn**
- **Multi-turn**
- **Multimodal**

---

## Prompt Generation

We prepared the `prompt_generation.ipynb` notebook, which we used to generate our datasets.

The notebook can be easily modified to:
- change dataset size
- change dataset proportions
- refill templates after modifications

---

## Evaluation

Evaluation is performed in the `prompt_evaluation.ipynb` notebook.

This notebook presents evaluation pipeline and statistics related to computation time for different models.

Both the prompt generation and evaluation notebooks were created to be easily modified and adjusted for different models.

---

## Model and Configuration

The default model used for rephrasing and judging is:

meta-llama/Llama-3.1-8B-Instruct

This choice reflects the most advanced model we could run on our available hardware.

The model can be changed in:
- `multimodal_evaluation`
- `text_evaluation`

---

## Judge Prompt

The prompt used for judging model outputs is defined in the `src/evaluation/judge_prompt.txt` file.

Users can modify this prompt and re-run the evaluation with new instructions.

---

## Judge Evaluation
Evaluation of the LLM-as-a-judge is performed in the `judge_evaluation.ipynb` notebook.  
It uses precomputed victim results and creates samples for each prompt type and category.  
Human-verified samples are already available in the `data/eval_judge` directory.

Users can modify and rerun the experiments; however, new `human_verdict` scores must be provided for any newly generated samples.

## EDA

Extended exploratory data analysis is implemented in `src/eda/eda_extended.py`.  
It loads all evaluation outputs (text single‑turn, text multi‑turn, multimodal), computes pass rates, extracts linguistic features, compares categories and modalities, and analyzes over‑refusal.

Running the script generates summary statistics and plots in `src/eda/plots/`.  
Users can rerun the EDA on new evaluation files; plots and metrics will update automatically.