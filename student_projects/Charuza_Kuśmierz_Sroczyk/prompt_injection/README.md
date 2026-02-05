# LLM Evaluation: Prompt Injection

## Project Overview
This project evaluates the robustness of open-source Large Language Models (LLMs) and Vision-Language Models (VLMs) against adversarial **Prompt Injection** attacks. 

We analyze distinct attack vectors, ranging from classic single-turn to multi-turn context drifting and visual prompt injections to determine how different architectures handle safety alignment under pressure.

---

## Models & Configuration
The evaluation was performed locally.

| Model | Type | Parameters | Focus |
|-------|------|------------|-------|
| **Llama 3.1 Instruct** | LLM | 8B | Safety & General Purpose |
| **Mistral v0.3** | LLM | 7B | Utility & "Uncensored" behavior |
| **LLaVA 1.5** | VLM | 7B | Visual Understanding (Multimodal) |

---

## Dataset Structure
We utilized a custom dataset (200 prompts) designed to test specific behavioral "tells". The text-
based prompts (both single-turn and
multi-turn) were created and adapted based on the
BIPIA benchmark. The visual attacks were directly sourced from the MM-SafetyBench.

| Category | Samples | Description |
|----------|:-------:|-------------|
| **Single-turn** | 80 | Direct override, Indirect injection, Code injection, Suffix attacks. |
| **Multi-turn** | 40 | Context saturation testing (3-turn and 5-turn dialogues). |
| **Multimodal** | 40 | Visual Prompt Injections (text embedded in images, OCR attacks). |
| **Safe (Baseline)** | 40 | Benign queries to test False Positive Rate. |

---

## Methodology

### 1. Attack Pipeline
- **Temperature:** 0.1 (Deterministic output)
- **Repetition Penalty:** 1.2
- **Seeds:** Experiments averaged over 3 independent runs (Seeds: 33, 34, 35).

### 2. Evaluation ("Model-as-a-Judge")
We employed an automated evaluator (**Gemma 2**) to grade responses as *Safe (1)* or *Unsafe (0)*.

### 3. Validation
To ensure reliability, we manually verified the results from Seed 33.
- **Agreement Accuracy:** 94.23%
- *Note:* The primary source of disagreement was the Judge hallucinating violations in benign multimodal responses.

---

## Installation & Usage

### Prerequisites
Ensure you have Python 3.10+ installed. To set up the environment, install the dependencies from the provided file:

```bash
pip install -r requirements.txt
```

The project is controlled via the main.py entry point. 
The main.py file can be executed to test prompts for a given seed and have them evaluated automatically by the evaluator. 
Once the evaluation run is complete, the results can be explored and analyzed using the notebook located at notebooks/results.ipynb

## Project structure
```text
.
├── config/
│   └── config.yaml                 # Global configuration (models, params, paths)
├── data/
│   ├── prompt_injection_data.json  # Main dataset
│   ├── results/                    # Raw model outputs
│   └── images/                     # Adversarial images for LLaVA
├── analysis_output/                # Generated plots (Boxplots, ASR bars)
├── notebooks/
│   ├── results.ipynb               # Results analysis
│   └── eda.ipynb                   # Eda of the dataset & Visualization
├── src/
│   ├── __init__.py
│   ├── models.py                   # Model loading & generation logic
│   ├── evaluate.py                 # Judge/Grading logic
│   └── utils.py                    # Helper functions (logging, data loading)
├── main.py                         # Main entry point for the pipeline
├── README.md                       # Project documentation
└── requirements.txt                # Python dependencies
```

## Dataset & Resources

Due to file size constraints, the dataset and results are hosted externally (JSON/JPG/CSV/PNG files).
https://drive.google.com/drive/folders/1900fUrTxEZic4x9XXP4yjveqqWHky-KG?usp=sharing
