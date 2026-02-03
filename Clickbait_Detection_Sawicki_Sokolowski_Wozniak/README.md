# NLP Clickbait Detection

## Project Overview

This repository contains code and data for **clickbait detection in article headlines**. Three modeling pipelines are implemented:

1. **Random Forest**: Classical ML using TF-IDF features and custom hand-crafted features.  
2. **Feedforward Neural Network**: Dense embeddings with GloVe vectors and custom features.  
3. **DistilBERT**: Transformer-based sequence classification using Hugging Face models.

The project supports **training, evaluation, and saving predictions** and model outputs for reproducibility.  

---

## Directory Structure For Input Data

nlp2025/ \
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;data/ \
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;clickbait17-train-170331/\
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; instances.jsonl\
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; truth.jsonl\
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;    clickbait17-train-170630/\
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; instances.jsonl\
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; truth.jsonl\
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;    webis-clickbait-22/\
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; train.jsonl\
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; validation.jsonl\


---

## Data Sources

The datasets used in this project are publicly available:  

1. **Webis-Clickbait-22**:  
   - [https://zenodo.org/records/6362726](https://zenodo.org/records/6362726)  
   - Contains recent clickbait and non-clickbait headlines.  

2. **Clickbait17**:  
   - `clickbait17-train-170331.zip` and `clickbait17-train-170630.zip`  
   - [https://zenodo.org/records/5530410](https://zenodo.org/records/5530410)  
   - Contains historical annotated headlines for training classical models.  

---
