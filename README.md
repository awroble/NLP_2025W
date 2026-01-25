# NLP_2025W: Putin's Talks Analysis (1999–2025)

This repository contains a comprehensive Natural Language Processing (NLP) study of Vladimir Putin's political rhetoric. The project evaluates the evolution of topics, sentiment, and propaganda techniques over 25 years. The final report is in the file **`Report_Putins_Talks.pdf`**

## 📂 Repository Structure

* **`LLMs overview/`**: Evaluation of RAG pipelines and LLM performance (TinyLlama, Qwen, Phi, and GPT-4o) on analytical tasks.
* **`Sentiment_Analysis/`**: RoBERTa-based sentiment trajectories for Poland, USA, Ukraine, China, and the EU.
* **`Word frequency analysis/`**: Statistical trends for economic terms, military modernization, and WWII references.
* **`propaganda/`**: Detection of propaganda content and classification of specific rhetorical techniques using ModernBERT.
* **`putin_speeches_bertopic/`**: Modular topic modeling and Dynamic Topic Modeling (DTM) to track narrative shifts.
* **`Poland_Statements_Analysing.ipynb`**: Focused study of Poland's image using Zero-shot classification and "Enemy" image clustering (S-BERT + UMAP + DBSCAN).
* **`Report_Putins_Talks.pdf`**: Final comprehensive report detailing methodological findings and rhetorical insights.

## 🛠️ Key Methodologies

* **Topic Discovery**: BERTopic with c-TF-IDF for granular and temporal theme analysis.
* **Enemy Construction**: Dimensionality reduction (UMAP) and clustering (DBSCAN) on S-BERT embeddings to map hostile rhetoric.
* **Contextual Classification**: Zero-shot categorization of bilateral relations (Partner, Ally, Neighbor, Enemy).
* **LLM-as-a-Judge**: Utilizing high-tier models to evaluate the effectiveness and hallucinations of smaller open-source LLMs.

## 📊 Dataset
The analysis is based on the **Putin Corpus (1999–2025)**, primarily sourced from official transcripts at kremlin.ru.

---
*Deliverable for the NLP course at Warsaw University of Technology (WUT).*
