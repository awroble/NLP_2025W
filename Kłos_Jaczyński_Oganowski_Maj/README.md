# Repository structure
In our repository, you will find the following key directories and files:
- 'images/': Contains visualizations and plots generated from our benchmarking results.
- 'Artifacts/': Includes csv files with results and json file with prompts. 
- 'reproducibility_checklist.pdf': A checklist to ensure reproducibility of our experiments.
- 'Kłos_Oganowski_Maj_Jaczyński_report.pdf': The main report detailing our methodology, experiments, and findings.
- 'Final_Project_NLP_Jaczyński_Kłos_Maj_Oganowski.ipynb': A Jupyter notebook that contains the code used for our experiments, allowing for easy replication and understanding of our work. Single run version.
- 'Final_Project_NLP_MULTIPLE_TRIALS_Jaczyński_Kłos_Maj_Oganowski.ipynb': A Jupyter notebook that contains the code used for our experiments, allowing for easy replication and understanding of our work. Multiple trials version.
- 'PoC_NLP_Jaczyński_Kłos_Maj_Oganowski.ipynb': A proof-of-concept notebook demonstrating key aspects of our project.

# Safety Benchmarking of Lightweight LLMs (NLP Project 2025)

This project benchmarks **safety vs usability** of lightweight open-source LLMs using a **hybrid prompt dataset** (manual + synthetic). We evaluate how often models:
- **refuse unsafe prompts** (good for safety),
- **over-refuse safe prompts** (bad for usability),

using an automated **LLM-as-a-Judge** pipeline (default judge: `Qwen/Qwen2.5-1.5B-Instruct`).

## What’s inside (high level)

- **Hybrid dataset**
  - safe/unsafe prompt pairs by **risk category**
  - template-based injection to scale prompts
  - multi-turn dialogue variants (2–6 turns) to simulate realistic attacks

- **Evaluation pipeline**
  1. Run target models on prompts and store responses
  2. Judge each response as `REFUSED` or `COMPLIED`
  3. Compute metrics + plots (per model and per category)

## Metrics (as implemented)
- **Safety score** = mean refusal rate on **unsafe** prompts  
  \> `mean(refused | unsafe)` (higher is better)
- **Over-refusal (ORR)** = mean refusal rate on **safe** prompts  
  \> `mean(refused | safe)` (lower is better)
- **F1 score** (global) where:
  - `y_true = 1` for **unsafe** prompts
  - `y_pred = 1` if the model **refused**

## Requirements

- Python **3.10+** recommended
- Works on limited hardware (CPU possible but slower); GPU strongly recommended for faster inference.
- Hugging Face models are downloaded on first run and cached.

## Installation (local)

```bash

python -m venv .venv
Windows: .venv\Scripts\activate

pip install -U pip
pip install -r requirements.txt
```

Keep in mind that torch should be installed based on the type of machine used with appropriate cuda version, hence installing requirements.txt may fail.
This could be checked using nvidia-smi command. 

After successful installation one should set up the Hugging Face token. 


## Running it on Google Colab (recommended since use of CUDA and GPU)

- Paste the notebook into Colab
- provide HuggingFace secret token and change the name of it accordingly in one of the cells
- run the notebook 