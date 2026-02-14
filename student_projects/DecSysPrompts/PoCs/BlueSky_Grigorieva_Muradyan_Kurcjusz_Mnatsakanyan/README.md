# Bluesky Sentiment Analysis (LLMs + Tag Groupings)

This project studies sentiment/emotion modeling on **Bluesky** (AT Protocol) and explores how **hashtags/tags** can be grouped using label signals and graph structure. The work combines exploratory data analysis (EDA), tag-based analysis, and baseline ML sentiment experiments.

## Repository structure

- `code/`
  - `eda/`
    - `EDA_Bluesky_social.ipynb`  
      Feed-level EDA across multiple Bluesky feed samples (volume, users, language tags, engagement, replies, time activity).
    - `EDA_POLITISKY24.ipynb`  
      EDA for the POLITISKY24 dataset (stance distributions by target entity, confidence levels, text length, hashtag summaries).
    - `Labels_POLITISKY24.ipynb`  
      Tag-centric analysis using predicted emotion labels + hashtag co-occurrence graphs and tag–label summaries.
  - `ml_models/`
    - `Sentiment_Analysis_ML_Models.ipynb`  
      Transformer sentiment inference + classical TF–IDF baselines (LogReg / MultinomialNB / LinearSVC) trained on pseudo-labels.
- `presentation/`
  - `BlueSky_milestone_1_presentation.pdf`  
    Project milestone slides.
- `reports/`
  - `Project Proposal - Sentiment Analysis with Large Language Models on Bluesky` - First milestone report
  - `Updated for Milestone 2 - Sentiment Analysis with Large Language Models on Bluesky` - Second milestone report

## Models used (current stage)

- Sentiment inference: `distilbert-base-uncased-finetuned-sst-2-english`
- Emotion inference: `j-hartmann/emotion-english-distilroberta-base`
- Classical baselines: TF–IDF + Logistic Regression / Multinomial Naive Bayes / LinearSVC

## How to run

1. Create an environment (Python 3.10+ recommended).
2. Install typical dependencies:
   - `pandas`, `numpy`, `matplotlib`
   - `scikit-learn`
   - `transformers`, `torch`
   - (optional, if used in your notebook) `networkx`

3. Open notebooks from `code/` and run top-to-bottom.

> Note: Some notebooks expect dataset files to be present locally (Bluesky Social Dataset, POLITISKY24). If paths are hard-coded, update them near the top of each notebook.

## Expected outcomes

- EDA tables/plots comparing feeds and POLITISKY24 subsets
- Extracted hashtag statistics and hashtag co-occurrence graphs
- Transformer-predicted sentiment/emotion labels + confidence scores
- Baseline ML evaluation results for TF–IDF classifiers (trained on pseudo-labels)
