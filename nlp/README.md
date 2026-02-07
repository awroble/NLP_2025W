# Optimized Retrieval-Augmented Generation System for University Educational FAQ Retrieval

## Project Overview

This project implements a **Retrieval-Augmented Generation (RAG) chatbot** optimized for university educational FAQ retrieval. Through systematic empirical optimization of 216 parameter combinations, we demonstrate that domain-specific RAG configuration yields measurable performance improvements in accuracy, recall, and efficiency.

**Key Results:**
- **Accuracy@K:** 0.559 (55.9% retrieved documents correct)
- **Recall@K:** 0.553 (55.3% relevant documents retrieved)
- **Response Time:** 311.4ms (suitable for interactive systems)
- **Cost Efficiency:** 39.2% faster than 70B parameter models while maintaining 96.8% accuracy parity

---

## Project Structure

nlp/
├── README.md                               

├── data/
│   ├── chatbot_csv.csv                     # Educational FAQ dataset (500 Q&A pairs)
│   └── WUT_FAQ_Sources_Documentation.txt   # Detailet FAQ sources

├── code/
│   ├── requirements.txt                    # Python dependencies
│   ├── chatbot_interactive.py              # Interactive RAG chatbot CLI
│   ├── chatbot_parameter_analysis.py       # Grid search over 216 configurations
│   └── WUT_Chatbot_EDA_Advanced.ipynb      # Exploratory Data Analysis notebook

├── docs/
│   ├── Optimized_Retrieval_Augmented_Generation_Systemfor_University_Educational_FAQ_Retrieval.pdf  # Report PDF
│   ├── REPRODUCIBILITY_DETAILED.md         # Detailed reproducibility checklist
│   ├── CONTRIBUTIONS.md                    # Team member contributions
│   

├── results/
│   ├── chatbot_results_20260120_182117.csv 
│   ├── chatbot_ranking_20260120_182117.csv 
│   └── chatbot_report_20260120_182117.txt  


---

## Dataset

**File:** `chatbot_csv.csv`

**Structure:**
- **ID:** Unique identifier (1–500)
- **Question:** Student inquiry (avg. 23 characters)
- **Answer:** Authoritative response (avg. 117 characters)
- **Category:** Semantic classification (11 categories)
  - Admission & Application
  - Academic Policies
  - Tuition & Financial Aid
  - Accommodation & Housing
  - Visa & Immigration
  - Coursework & Grading
  - Facilities & Support
  - Student Life
  - Research & Internships
  - Technical Support
  - Other
- **Source_URLs:** Official university documentation links

**Data Quality:**
- 100% completeness (0 missing values)
- 0.4% Q&A pair duplicates (minimal)
- 99.4% unique questions
- Verified against official Warsaw University of Technology sources

Detailed descriptions of the original FAQ sources are provided in `WUT_FAQ_Sources_Documentation.txt`.

---

## Installation

### Prerequisites
- Python 3.9+
- pip or conda

### Setup

1. **Clone or download the project:**
   ```bash
   cd nlp
   ```

2. **Create a virtual environment (optional but recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set Groq API key:**
   ```bash
   export GROQ_API_KEY='your_groq_api_key_here'
   ```
   On Windows:
   ```bash
   set GROQ_API_KEY=your_groq_api_key_here
   ```

---

## Usage

### 1. Interactive RAG Chatbot

**Purpose:** Query the FAQ knowledge base in real time with the optimal configuration.

**Run:**
```bash
python chatbot_interactive.py
```

**Workflow:**
1. Embeddings are computed for all 500 FAQ pairs.
2. A FAISS index is built for fast retrieval.
3. The Groq API is initialized.
4. An interactive loop accepts user queries.

**Example:**
```text
You: What is the acceptance rate?
Answer:
The acceptance rate at Warsaw University of Technology varies by faculty. 
Typically ranges from 20-40% for undergraduate programs. 
For exact figures, refer to the official admissions portal.

Sources:
 - https://www.wut.edu.pl/admissions
```

**Exit:** Type `exit`, `quit`, or `q`.

---

### 2. Parameter Analysis (Grid Search)

**Purpose:** Evaluate all 216 parameter combinations across retrieval accuracy and latency metrics.

**Run:**
```bash
python chatbot_parameter_analysis.py
```

**Configuration Space:**
- **Embedding Models:** 2 options (`all-MiniLM-L6-v2`, `all-mpnet-base-v2`)
- **LLM Models:** 2 options (`llama-3.1-8b-instant`, `llama-3.3-70b-versatile`)
- **Temperature:** 3 values (0.2, 0.5, 0.8)
- **Top-K Retrieval:** 3 values (1, 3, 5)
- **Test Queries:** 6 diverse FAQ questions
- **Total Tests:** 2 × 2 × 3 × 3 × 6 = **216 configurations**

**Output Files:**
```text
chatbot_results_YYYYMMDD_HHMMSS.csv     # Raw results from all 216 tests
chatbot_ranking_YYYYMMDD_HHMMSS.csv     # Ranked configurations
[console output]                        # Printed analysis summary
```

---

### 3. Exploratory Data Analysis

**Purpose:** Understand dataset structure, quality and linguistic properties.

**Run:**
```bash
jupyter notebook WUT_Chatbot_EDA_Advanced.ipynb
```

**Coverage:**
- Dataset architecture and integrity
- Duplicate analysis
- Text length distributions
- Linguistic properties (word counts, readability)
- Category balance
- Source documentation validation
- Comparative analysis with SQuAD, MS MARCO, Natural Questions

---

## Optimal Configuration

Based on exhaustive grid search:

| Parameter        | Optimal Value            | Rationale                                                       |
|------------------|--------------------------|-----------------------------------------------------------------|
| Embedding Model  | `all-MiniLM-L6-v2`       | 8.5% improvement vs. larger model; lower latency               |
| LLM Model        | `llama-3.1-8b-instant`   | 96.8% accuracy of 70B model; 39.2% faster; cost-effective      |
| Temperature      | 0.5                      | Balance between determinism (0.2) and diversity (0.8)           |
| Top-K Retrieval  | 3                        | Good context coverage; Accuracy@K and Recall@K both at 0.553   |

**Performance Metrics (Optimal Configuration):**
- Accuracy@K: **0.559**
- Recall@K: **0.553**
- MRR: **0.67**
- Retrieval Latency: **24.3 ms**
- Generation Latency: **287.1 ms**
- **Total Response Time: 311.4 ms**

---

## Evaluation Metrics

### Retrieval Accuracy

- **Accuracy@K:** Fraction of retrieved documents that are relevant  
  `|{correct docs}| / K`

- **Recall@K:** Fraction of relevant documents successfully retrieved  
  `|{correct docs}| / |{all relevant docs}|`

- **Mean Reciprocal Rank (MRR):** Position of first relevant document  
  `1 / position_of_first_relevant`

### Performance

- **Retrieval Latency:** Time for embedding the query and FAISS search (ms)
- **Generation Latency:** Time for LLM inference (ms)
- **Total Response Time:** Sum of retrieval + generation (ms)

---

## Reproducibility

For detailed reproducibility information, see **`REPRODUCIBILITY_DETAILED.md`**.

**Quick Checklist:**
- Python 3.9+
- All dependencies in `requirements.txt`
- Dataset file: `chatbot_csv.csv`
- Groq API key for LLM inference
- FAISS for retrieval (included in requirements)
- All 216 configurations reproducible via `chatbot_parameter_analysis.py`

---


## Key References

- Lewis et al. (2020) – Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks.
- Karpukhin et al. (2020) – Dense Passage Retrieval for Open-Domain Question Answering (EMNLP).
- Reimers & Gurevych (2019) – Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks (EMNLP).
- Johnson et al. (2019) – Billion-Scale Similarity Search with GPUs (IEEE Transactions on Big Data).
- Thakur et al. (2021) – BEIR: A Heterogeneous Benchmark for Zero-Shot Evaluation of IR Models (NeurIPS).

---

## Contact & Support

**Project Authors:**
- Iñaki Gutiérrez-Mantilla López – 01205606@pw.edu.pl
- Hèctor Rodón Llabería – 01205604@pw.edu.pl

**Supervisor:**
- Anna Wróblewska – anna.wroblewska1@pw.edu.pl  
  Warsaw University of Technology

For further details, please refer to the project report and reproducibility documentation.

---

## License

This project is provided for academic purposes as part of the NLP Course at Warsaw University of Technology (Winter 2025).

---

**Last Updated:** January 25, 2026
