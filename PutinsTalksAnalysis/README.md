# Putin’s Talks — NLP Analysis of Political Discourse

This repository contains the code and analysis for **“Putin’s Talks”**, an NLP project investigating the evolution of Vladimir Putin’s public rhetoric using a hybrid pipeline that combines classical NLP techniques with modern Transformer-based models.

The project addresses quantitative, contextual, and diachronic questions related to geopolitical discourse, including word frequencies, named entity usage, semantic framing, and narrative shifts over time.

## Project Structure

```
├── data/                         
├── 01_prepare_dataset.ipynb      # Data loading, cleaning, and date normalization
├── 02_EDA.ipynb                  # Exploratory Data Analysis (speech length, density)
├── 03_NER_and_collocations.ipynb # Named Entity Recognition and contextual collocations
├── 04_statistics.ipynb           # Frequency analysis (democracy, NATO, threats, etc.)
├── NLP_Midterm_Report_*.pdf      
├── requirements.txt              
└── README.md                     
```



## Setup

Install the required dependencies:

```bash
pip install -r requirements.txt
```

All analyses are implemented in Jupyter notebooks and can be run sequentially.


## Reproducibility

* All preprocessing and analysis steps are documented in the notebooks.
* Data sources and assumptions are described in the accompanying report.
* No proprietary models or paid APIs are required for the core experiments.

## Authors

* **Łukasz Grabarski**
* **Łukasz Lepianka**
* **Marta Szuwarska**
