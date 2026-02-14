
# Topic discovery

## Module: `claim_extraction`

This folder contains all code related to claim extraction from text.

### 1. `bbc_dataset.py`

- Responsible for **loading and preparing the dataset**.

### 2. `claim_extraction.py`

- Implements the **main claim extraction pipeline**.
- Responsibilities:
  - Processing input text to detect claims
  - Applying the extraction model or logic
  - Measuring performance metrics

### 3. `EDA.ipynb`

- Performs **Exploratory Data Analysis** on the dataset.


## Module: `automatic_bias_sentiment_pipeline_for_news`

This research project presents an automated pipeline for analyzing news articles across dimensions of sentiment, bias, factuality, and framing to enhance topic discovery and classification.

### 1. `dataset`

- create_dataset_single_dataset.ipynb: Responsible for creating the dataset by augmenting the AG News benchmark (120,000 samples) with four nuance dimensions: **Sentiment, Bias, Subjectivity, and Framing**. Uses all-MiniLM-L6-v2 for semantic embeddings and twitter-roberta-base-sentiment for emotional characterization. Implements a Zero-Shot NLI approach using nli-deberta-v3-small to categorize articles into frames like Corporate, Social Impact, and Non-Economic.
requirements.txt: Contains libraries necessary for Hugging Face dataset creation and model inference.

### 2. `clustering`

- basic_clustering_comparison.ipynb is responsible for **visualising the newly created dataset**.
- requirements.txt contains libraries necessary for clustering

### 3. `ablation_studies`

This module evaluates the empirical utility of the machine-learning generated features using a Multi-Layer Perceptron (MLP) classifier.
- Objective: To determine the **marginal contribution of each nuance dimension** toward the accuracy of news category classification.
- Infrastructure: Experiments were conducted using PyTorch on Nvidia hardware (RTX 3090). You can follow the training setup by using vast.ai and selecting image https://hub.docker.com/r/vastai/pytorch/. 
- To train model, run: "python3 main.py". The default parameters are taken from config.py, which are exactly as we have used them.

### 4. `Dataset Access`

The resulting augmented dataset is hosted on Hugging Face: mkita/topic-discovery-for-news-articles-test


---
