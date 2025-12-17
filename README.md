# NLP Safety Benchmark (POC)

This repository contains the first proof‑of‑concept (POC) implementation of a modular benchmarking framework for evaluating the safety of large language models (LLMs) under adversarial prompting conditions. The long‑term goal is to develop a comprehensive, multi‑category benchmark with scalable generation, evaluation, and adversarial attack components.

This POC focuses on a single category:

**✓ Social Engineering (Category 1 of 3)**  


Implemented using template‑based prompt generation, an LLM‑as‑a‑judge evaluation pipeline, and adversarial rephrasing.

---

## Repository Structure

```
NLP-safety-benchmark/
│
├── src/
│   ├── generation/
│   │    ├── output/                        # generated prompts (output of TemplateFiller)
│   │    ├── templates/                     # templates for generation with variations for placeholders
│   │    │   
│   │    └── TemplateFiller.py              # Template-filling generator
│   │    
│   └──local_LLM/                           # Stores demo of capabilites of local LLMs
│       ├── safe_prompt.txt                 # LLM-as-a-judge safe prompt instruction
|       ├── unsafe_prompt.txt               # LLM-as-a-judge unsafe prompt instruction
│       ├── model_manager.py                # Functions for managing local LLMs
│       └── demo.py                         # Short demo same that were performed in POC
│
|── token.txt                               # You have to create this file and paste your huggingface login token  
└── README.md

```


## Prompt Generation (Template‑Filling)

The POC includes a simple template‑filling generator that produces **single‑turn social‑engineering prompts**. Templates define the structure of the attack scenario, and variable slots introduce controlled variation.


## Evaluation Pipeline (local_LLM)

For purpose of POC we prepared short demo of capabilities of local LLMs. You can use paraphrasing model, choose  easier or harder prompt or just try your own one. IF models are too big you can try smaller ones, but results could be unreasonable.

In the future adversarial prompts modification will be separated from evaluating pipeline.
## Important 
You need to login into huggingface with token in order to download model from website. Github doesn't track file with your token, but you need to create it by yourself. You need apply for permition for llama model, but it is matters of maximum hour. 

Remember models are heavy and requires noticeable resources. Importantly code supports CUDA library.


### Current Scoring Scale

Every answer is scored from 1-5. Details are described in safe and unsafe prompts for LLM as a judge.

## Usage

### Generate a Set of Prompts (n=50)

```bash
python src/generation/TemplateFiller.py
```

### Run demo of local LLM
```bash
python src\local_LLM\demo.py
```
## Roadmap

Planned next steps include:

- Adding categories **2) sensitive data extraction** and **3) physical and mental health**  
- Multi‑turn adversarial conversations   
- Multi-modal prompts
- Adversial mutations