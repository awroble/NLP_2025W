# Putin’s Talks — NLP Analysis of Political Discourse

This repository contains the code and analysis for **“Putin’s Talks”**, an NLP project investigating the evolution of Vladimir Putin’s public rhetoric. We implement a **hybrid pipeline** that combines classical NLP techniques (Regex, TF-IDF, Dependency Parsing) with modern Transformer-based models (BERT, BART) and Large Language Models (Gemini, Gemma) to track narrative shifts over time.

The project addresses quantitative, contextual, and diachronic questions related to geopolitical discourse, offering a comparative analysis of deterministic vs. generative approaches.

## Key Features

* **Hybrid NER Pipeline**: compares `dslim/bert-base-NER` against Generative LLMs and manual annotation for extracting geopolitical entities.
* **Semantic Framing Analysis**: Uses Dependency Parsing (SpaCy) for term extraction and Zero-Shot Classification (`facebook/bart-large-mnli`) to detect stance (e.g., "Partner" vs. "Enemy").
* **Methodological Comparison**: Benchmarks "Classical SOTA" (Regex, BERT) against "Modern SOTA" (LLMs) for tasks like frequency counting and entity extraction.
* **Diachronic Analysis**: Tracks the evolution of specific topics and keywords (e.g., "Democracy", "NATO", "Multipolar World") across varying timeframes.

## Project Structure

The project is organized into modular notebooks and source code:

```text
├── data/                                 # Dataset storage
│   ├── dates_tags.csv                    # Metadata index
│   ├── samples/                          # Validation sets for NER/Counting
│   └── sentences/                        # Processed country-specific sentence datasets
├── notebooks/                            # Analysis notebooks (run sequentially)
│   ├── 01_prepare_dataset.ipynb          # Data loading, cleaning, and date normalization
│   ├── 02_EDA.ipynb                      # Exploratory Data Analysis (speech length, density)
│   ├── 03a_NER.ipynb                     # NER pipeline (BERT vs Gemini)
│   ├── 03b_context_classification.ipynb  # Semantic framing & Zero-Shot classification
│   ├── 03c_plots.ipynb                   # Visualization of NER and Framing results
│   ├── 04a_counting.ipynb                # Frequency analysis (Regex vs LLM counting)
│   ├── 04b_annotating.ipynb              # Annotation helpers for ground truth generation
│   ├── 05_topics_modelling.ipynb         # Topic modeling implementation
│   └── 05a_topics_plots.ipynb            # Topic visualization
├── src/                                  # Shared source code
│   ├── nlp_models.py                     # Model wrappers (BERT, BART) and pipeline logic
│   └── utils.py                          # Utilities for seeding, logging, and plotting
├── reports/                              # Generated PDF reports and presentations
├── requirements.txt                      # Project dependencies
└── README.md                             # Project documentation



## Setup

Install the required dependencies:

```bash
pip install -r requirements.txt
```

All analyses are implemented in Jupyter notebooks and can be run sequentially.

## Data
The dataset consists of transcribed speeches by Vladimir Putin, along with metadata including dates and tags.
Dataset: https://drive.google.com/file/d/1Of3e_pz-01IqslaIk6pTlVtPn7p4zx9z/view
Questions: https://github.com/grzegorzZ1/masters_2025_nlp/tree/initial_steps
Downloaded data is stored in the `data/` directory.


## Reproducibility

* All preprocessing and analysis steps are documented in the notebooks.
* Data sources and assumptions are described in the accompanying report.
* No proprietary models or paid APIs are required for the core experiments.

## Authors

* **Łukasz Grabarski**
* **Łukasz Lepianka**
* **Marta Szuwarska**
