# Safety Evaluation of Diffusion Large Language Models

**Natural Language Processing Course - Final Project**
Data Science, Faculty of Mathematics and Information Science
Warsaw University of Technology, Winter 2025

***Author:*** Wojciech Kutak
***Supervisor:*** Anna Wróblewska, PhD

---

## 📋 Project Overview

This repository contains the first comprehensive safety evaluation of **diffusion-based Large Language Models (LLMs)** using the **LLM Safety Benchmark (LSB)** framework developed by Kisiel et al. (2025) at Warsaw University of Technology.

### Research Question

**Do diffusion-based LLMs exhibit different safety characteristics compared to autoregressive models?**

### Models Evaluated

- **LLaDA-8B-Instruct** ([GSAI-ML/LLaDA-8B-Instruct](https://huggingface.co/GSAI-ML/LLaDA-8B-Instruct)) - Masked diffusion language model with instruction tuning
- **MMaDA-8B-MixCoT** ([Gen-Verse/MMaDA-8B-MixCoT](https://huggingface.co/Gen-Verse/MMaDA-8B-MixCoT)) - Multimodal diffusion model with chain-of-thought reasoning

### Key Findings

🎯 **LLaDA-8B-Instruct achieves exceptional safety:**
- **Attack Success Rate (ASR): 27.5%** - 37% better than best autoregressive baseline (Llama-3.2-1B: 43.8%)
- **50.9% improvement** over autoregressive average (56.0% ASR)
- **Low factual error rate: 3.3%** - enabling accurate misinformation correction
- **Balanced refusal behavior: 43.3%** - maintains utility while ensuring safety

⚠️ **MMaDA-8B-MixCoT shows high vulnerability:**
- **ASR: 65.2%** - comparable to minimally-trained TinyLlama (68.1%)
- **High safety failures: 41.0%** - fails to refuse harmful requests
- **High factual errors: 20.7%** - provides incorrect information
- Suggests multimodal training introduces safety challenges

---

## Repository Structure

```
nlp-course/
├── data/                           # LSB benchmark prompts (600 prompts)
│   ├── prompts_health.json         # Health domain (200 prompts)
│   ├── prompts_misinformation.json # Misinformation domain (200 prompts)
│   └── prompts_disinformation.json # Disinformation domain (200 prompts)
│
├── src/                            # Source code
│   ├── llada_evaluation/           # LLaDA evaluation infrastructure
│   │   ├── inference.py            # LLaDA inference wrapper
│   │   ├── adapters.py             # LSB adapter for LLaDA
│   │   ├── run_evaluation.py       # Main evaluation script
│   │   ├── reproducibility.py      # Seed management and determinism
│   │   ├── test_single_sample.py   # Single-prompt testing
│   │   └── requirements.txt        # Dependencies (transformers==4.38.2)
│   │
│   ├── mmada_evaluation/           # MMaDA evaluation infrastructure
│   │   ├── inference.py            # MMaDA inference wrapper
│   │   ├── adapters.py             # LSB adapter for MMaDA
│   │   ├── run_evaluation.py       # Main evaluation script
│   │   ├── reproducibility.py      # Seed management and determinism
│   │   ├── test_single_sample.py   # Single-prompt testing
│   │   └── requirements.txt        # Dependencies (transformers==4.46.0)
│   │
│   └── analysis/                   # Analysis and visualization tools
│       ├── analyze_results.py      # Metrics calculation
│       ├── visualize_results.py    # Plot generation
│       ├── compare_with_baseline.py # Baseline comparison
│       ├── generate_model_comparison.py # Model comparison plot
│       └── requirements.txt        # Analysis dependencies
│
├── results/                        # Evaluation results
│   ├── llada/                      # LLaDA results
│   │   ├── results_LLaDA-8B-Instruct_*.json  # Full evaluation data
│   │   ├── results_LLaDA-8B-Instruct_*.csv   # Tabular summary
│   │   ├── config.json             # Evaluation configuration
│   │   ├── evaluation_*.log        # Execution log
│   │   └── plots/                  # Visualizations
│   │       ├── model_comparison.png
│   │       ├── domain_breakdown_*.png
│   │       ├── tier_analysis_*.png
│   │       └── ...
│   │
│   └── mmada/                      # MMaDA results (same structure)
│
├── NLP_2025W/                      # LSB framework (from course)
│   ├── evaluate.py                 # Core evaluation logic
│   ├── README.md                   # LSB documentation
│   └── ...
│
├── LLaDA/                          # LLaDA model repository
├── MMaDA/                          # MMaDA model repository
└── README.md                       # This file
```

---

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- CUDA-capable GPU (16GB+ VRAM recommended)
- Google Colab Pro (alternatively, for free GPU access)

### Installation

1. **Clone repository:**
   ```bash
   git clone https://github.com/your-username/nlp-course.git
   cd nlp-course
   ```

2. **Install analysis tools:**
   ```bash
   cd src/analysis
   pip install -r requirements.txt
   ```

3. **For evaluation (choose one):**

   **LLaDA:**
   ```bash
   cd src/llada_evaluation
   pip install -r requirements.txt
   ```

   **MMaDA:**
   ```bash
   cd src/mmada_evaluation
   pip install -r requirements.txt
   ```

### View Results

Results are already computed and included in `results/` directory. To analyze:

```bash
cd src/analysis

# Analyze LLaDA results
python analyze_results.py \
    --results ../../results/llada/results_*.json \
    --name "LLaDA-8B-Instruct"

# Analyze MMaDA results
python analyze_results.py \
    --results ../../results/mmada/results_*.json \
    --name "MMaDA-8B-MixCoT"

# Generate comparison plot
python generate_model_comparison.py \
    --llada ../../results/llada/results_*.json \
    --mmada ../../results/mmada/results_*.json \
    --output ../../results/plots/model_comparison.png

# Compare with baselines
python compare_with_baseline.py \
    --results ../../results/llada/results_*.json \
    --baseline-model autoregressive-average
```

---

## 📊 Evaluation Methodology

### LSB Framework

The **LLM Safety Benchmark (LSB)** evaluates model resilience against:
- **3 domains:** Health, Misinformation, Disinformation
- **4 difficulty tiers:** Basic (Tier 1) → Sophisticated (Tier 4)
- **6 attack methods:** No attack, topic selection, implicit misinformation, prompt injection, jailbreak/roleplay, multi-turn persuasion
- **600 total prompts:** 200 per domain × 4 tiers × various attack methods

### Metrics

| Metric | Description |
|--------|-------------|
| **Attack Success Rate (ASR)** | % prompts where model failed to respond safely (lower is better) |
| **Refusal Rate** | % prompts model refused to answer (moderate ~30-40% is ideal) |
| **Safety Failure Rate** | % harmful requests model failed to refuse (lower is better) |
| **False Positive Rate** | % safe requests model incorrectly refused (lower is better) |
| **Factual Error Rate** | % factual prompts with incorrect information (lower is better) |

### Baselines

Compared against autoregressive models from LSB paper:
- **Llama-3.2-1B-Instruct** (ASR: 43.8%, Refusal: 31.8%) - Best autoregressive baseline
- **TinyLlama-1.1B-Chat** (ASR: 68.1%, Refusal: 9.2%) - Minimally-trained baseline
- **OpenMath-Nemotron-1.5B** (ASR: 50.9%, Refusal: 82.9%) - Over-cautious baseline
- **Autoregressive Average** (ASR: 56.0%, Refusal: 20.5%) - Mean of Llama + TinyLlama

---

## 📈 Key Results

### Overall Performance

| Model | ASR | Refusal | Safety Failures | Factual Errors |
|-------|-----|---------|-----------------|----------------|
| **LLaDA-8B-Instruct** ✓ | **27.5%** | **43.3%** | **18.8%** | **3.3%** |
| MMaDA-8B-MixCoT | 65.2% | 19.3% | 41.0% | 20.7% |
| Llama-3.2-1B (baseline) | 43.8% | 31.8% | - | - |
| Autoregressive Avg | 56.0% | 20.5% | - | - |

### Domain-Specific Performance

**Health Domain:**
- LLaDA: 30.5% ASR | MMaDA: 70.0% ASR
- **39.5 percentage point gap** - LLaDA far safer for health applications

**Misinformation Domain:**
- LLaDA: 26.5% ASR | MMaDA: 69.5% ASR
- LLaDA's 3.3% factual error rate enables accurate misinformation correction

**Disinformation Domain:**
- LLaDA: 25.5% ASR | MMaDA: 56.0% ASR
- LLaDA appropriately refuses harmful content generation (54% refusal)

### Tier Progression

**LLaDA (Adaptive Safety):**
- Tier 1: 22.7% ASR, 15.3% Refusal
- Tier 4: 35.3% ASR, 60.7% Refusal
- **Refusal rate scales with attack sophistication** (adaptive behavior)

**MMaDA (Flat Vulnerability):**
- Tier 1-4: 54.7%-71.3% ASR (consistently vulnerable)
- No improvement with simpler prompts (fundamental alignment issues)

---

## 🔬 Running Evaluations

### Run Single Evaluation (LLaDA Example)

```bash
cd src/llada_evaluation

# Test single prompt first
python test_single_sample.py

python run_evaluation.py \
    --model-name GSAI-ML/LLaDA-8B-Instruct \
    --data-dir ../../data \
    --output-dir ../../results/llada \
    --judge-model Qwen/Qwen2.5-3B-Instruct \
    --seed 42 \
    --temperature 0.0 \
    --batch-size 8
```
---

## 🔧 Technical Details

### Reproducibility

- **Fixed random seed:** 42 (PyTorch, NumPy, Python random)
- **Deterministic generation:** Temperature 0.0
- **Configuration tracking:** All parameters logged in config.json
- **Version pinning:** Exact dependency versions in requirements.txt

### Environment Separation

LLaDA and MMaDA require different transformers versions:
- **LLaDA:** transformers==4.38.2
- **MMaDA:** transformers==4.46.0

Separate environments prevent dependency conflicts.

### Computational Requirements

**Per Model Evaluation:**
- **Time:** ~2-3 hours (LLaDA: 138 min, MMaDA: 140 min)
- **GPU:** NVIDIA A100
- **Memory:** ~10GB RAM for full evaluation
- **Storage:** ~150MB per evaluation result (JSON + CSV + logs)

---

## 📊 Data

### LSB Benchmark Dataset

- **Source:** NLP_2025W course framework
- **Total prompts:** 600
- **Domains:** Health (200), Misinformation (200), Disinformation (200)
- **Tiers:** 4 difficulty levels (150 prompts each)
- **Attack methods:** 6 types (no attack, topic selection, implicit misinformation, prompt injection, jailbreak/roleplay, multi-turn persuasion)

### Dataset Statistics

| Domain | Expected Behavior | Risk Categories |
|--------|-------------------|-----------------|
| Health | Factual (38.5%), Blocked (54.0%), Threat Assessment (7.5%) | Health advice, Physical safety, Threat detection |
| Misinformation | Factual (60.0%), Blocked (40.0%) | Common misconceptions, Explicit/implicit misinformation, Logical fallacies |
| Disinformation | Blocked (76.5%), Safe (11.0%), Factual (12.5%) | False narratives, Coordinated inauthentic behavior |

---

## 🤝 Acknowledgments

- **LSB Framework Authors:** Kamil Kisiel, Kinga Franczak, Wiktoria Koniecko, Piotr Kosakowski (Warsaw University of Technology)
- **Model Developers:** GSAI-ML (LLaDA), Gen-Verse (MMaDA)
- **Judge Model:** Qwen Team (Qwen2.5-3B-Instruct)
- **Course Instructor:** Anna Wróblewska, Warsaw University of Technology

---

## 📖 Citation

If you use this work, please cite:

```bibtex
@techreport{kutak2026diffusion_safety,
  title={Safety Evaluation of Diffusion Large Language Models: A Comparative Study Using the LSB Framework},
  author={Kutak, Wojciech},
  institution={Warsaw University of Technology},
  year={2026},
  type={Final Project Report},
  course={Natural Language Processing}
}
```

**LSB Framework:**
```bibtex
@techreport{lsb2025,
  title={LSB: LLM Safety Benchmark—A Unified Evaluation Framework for LLM Robustness},
  author={Kisiel, Kamil and Franczak, Kinga and Koniecko, Wiktoria and Kosakowski, Piotr},
  institution={Warsaw University of Technology},
  year={2025}
}
```

---

## 🔗 Links

- **Models:**
  - [LLaDA-8B-Instruct](https://huggingface.co/GSAI-ML/LLaDA-8B-Instruct)
  - [MMaDA-8B-MixCoT](https://huggingface.co/Gen-Verse/MMaDA-8B-MixCoT)
  - [Qwen2.5-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct) (Judge)

- **Baseline Models (from LSB paper):**
  - [Llama-3.2-1B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct)
  - [TinyLlama-1.1B-Chat](https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0)
  - [OpenMath-Nemotron-1.5B](https://huggingface.co/nvidia/OpenMath-Nemotron-1.5B)

- **Resources:**
  - [SafetyBench](https://github.com/thu-coai/SafetyBench)
  - [HarmBench](https://github.com/centerforaisafety/HarmBench)

