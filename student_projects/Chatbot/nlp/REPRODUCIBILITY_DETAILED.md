# Reproducibility Checklist: Detailed Parameters & Settings

Project: Optimized Retrieval-Augmented Generation System for University Educational FAQ Retrieval
Authors: Iñaki Gutiérrez-Mantilla López & Hèctor Rodon Llabería
Institution: Warsaw University of Technology, Winter 2025/2026

---

## 1. Experimental Parameters & Settings

### Embedding Models
- [ ] Model 1: all-MiniLM-L6-v2
  - Dimension: 384
  - Parameters: 22M
  - Pooling: mean
  - Quantization: None
  
- [ ] Model 2: all-mpnet-base-v2
  - Dimension: 768
  - Parameters: 110M
  - Pooling: mean
  - Quantization: None

### LLM Models
- [ ] Model 1: llama-3.1-8b-instant
  - Provider: Groq API
  - Context window: 8192 tokens
  - Quantization: Q4_K_M (server-side)
  - Max tokens: 1024
  
- [ ] Model 2: llama-3.3-70b-versatile
  - Provider: Groq API
  - Context window: 8192 tokens
  - Quantization: Q4_K_M (server-side)
  - Max tokens: 1024

### Temperature Settings
- [ ] T=0.2 (Deterministic mode)
  - Top_p: 1.0 (disabled)
  - Top_k: None (disabled)
  - Repetition penalty: 1.0
  
- [ ] T=0.5 (Balanced mode)
  - Top_p: 1.0 (disabled)
  - Top_k: None (disabled)
  - Repetition penalty: 1.0
  
- [ ] T=0.8 (Creative mode)
  - Top_p: 1.0 (disabled)
  - Top_k: None (disabled)
  - Repetition penalty: 1.0

### Retrieval Parameters
- [ ] Top-K=1: K value for FAISS retrieval
  - Similarity metric: L2 distance (Euclidean)
  - Similarity threshold: None
  - Re-ranking: None
  
- [ ] Top-K=3: K value for FAISS retrieval
  - Similarity metric: L2 distance (Euclidean)
  - Similarity threshold: None
  - Re-ranking: None
  
- [ ] Top-K=5: K value for FAISS retrieval
  - Similarity metric: L2 distance (Euclidean)
  - Similarity threshold: None
  - Re-ranking: None

### System Configuration
- [ ] RANDOM_SEED: 42
- [ ] GROQ_API_TIMEOUT: 30 seconds
- [ ] FAISS_INDEX_TYPE: Flat (exhaustive search)
- [ ] FAISS_METRIC: L2 (Euclidean distance)
- [ ] GPU_ENABLED: False (CPU-only)
- [ ] BATCH_SIZE: 1 (sequential queries)

### Prompt Template
- [ ] System prompt: Consistent across all configurations
- [ ] Query format: "Given these documents, answer: [query]"
- [ ] Document format: "[Doc 1] ... [Doc K]"
- [ ] Max input tokens: 2048
- [ ] Prompt length consistent: Yes

---

## 2. Data Specifications & Integrity

### FAQ Dataset Configuration
- [ ] Dataset file: WUT_FAQ_Dataset_500_Questions.csv
- [ ] Total records: 500 (verified: wc -l = 501 with header)
- [ ] Format: CSV (comma-separated values)
- [ ] Encoding: UTF-8 (verified with file command)
- [ ] Delimiter: Comma (,)
- [ ] Quote character: Double quote (")

### Column Structure
- [ ] Column 1: Question
  - Data type: String
  - Max length: 256 characters
  - Min length: 10 characters
  - Null values: 0
  
- [ ] Column 2: Answer
  - Data type: String
  - Max length: 512 characters
  - Min length: 20 characters
  - Null values: 0
  
- [ ] Column 3: Category
  - Data type: String (categorical)
  - Categories: 10 unique values
  - Null values: 0
  
- [ ] Column 4: SourceURL
  - Data type: String (URL)
  - Protocol: https://
  - Domain: wut.edu.pl or official sources
  - Null values: 0

### Data Distribution
- [ ] Admissions & Applications: 70 FAQs
- [ ] Tuition & Financial Aid: 80 FAQs
- [ ] Accommodation & Housing: 60 FAQs
- [ ] Visa & Immigration: 75 FAQs
- [ ] Academic Policies: 65 FAQs
- [ ] Language Requirements: 50 FAQs

### Dataset Integrity Checks
- [ ] File size: ~48 KB (verified: ls -la)
- [ ] MD5 checksum: [to be recorded: _______________]
- [ ] No duplicate rows: Verified (unique questions = 500)
- [ ] No missing values: Verified (null count = 0)
- [ ] Character encoding valid: Verified (iconv test passed)
- [ ] CSV format valid: Verified (Python pandas read_csv successful)

### Test Queries
- [ ] Query 1: "What is the acceptance rate?" (Admissions category)
- [ ] Query 2: "How much are the tuition fees?" (Financial category)
- [ ] Query 3: "Where can I find accommodation?" (Housing category)
- [ ] Query 4: "What are visa requirements?" (Immigration category)
- [ ] Query 5: "What is the academic calendar?" (Policies category)
- [ ] Query 6: "Do I need English certificate?" (Language category)

Test query properties:
- [ ] Fixed across all runs: Yes
- [ ] Not in FAQ ground truth: Verified manually
- [ ] Semantically diverse: Yes (covers 6 categories)
- [ ] Length range: 10-35 words

---

## 3. Experimental Execution Details

### Grid Search Configuration
- [ ] Total configurations: 216
- [ ] Calculation: 2 embeddings × 2 LLMs × 3 temperatures × 3 K-values × 6 queries
- [ ] Execution order: Sequential (no parallelization)
- [ ] Timeout per configuration: 30 seconds
- [ ] Max retries per config: 3

### Metric Computation Specifications

**Accuracy@K**
- [ ] Definition: P(retrieved_doc is relevant | top K retrieved)
- [ ] Relevance criteria: Document index matches ground truth
- [ ] Threshold: Exact match
- [ ] Range: [0, 1]
- [ ] Aggregation: Average across 6 queries

**Recall@K**
- [ ] Definition: P(relevant doc retrieved | exists K or fewer docs)
- [ ] Relevance criteria: Cosine similarity > 0.5 with ground truth
- [ ] Threshold: 0.5
- [ ] Range: [0, 1]
- [ ] Aggregation: Average across 6 queries

**Mean Reciprocal Rank (MRR)**
- [ ] Definition: Average(1 / rank of first relevant doc)
- [ ] First relevant rank: Minimum rank where similarity > 0.5
- [ ] Max rank considered: K
- [ ] Range: [0, 1]
- [ ] Aggregation: Average across 6 queries

**Retrieval Latency**
- [ ] Measurement: Time from query embedding to FAISS search completion
- [ ] Unit: Milliseconds (ms)
- [ ] Precision: 1 decimal place
- [ ] Excludes: Model loading, index initialization
- [ ] Aggregation: Average across 6 queries

**Generation Latency**
- [ ] Measurement: Time from prompt sending to full response received
- [ ] Unit: Milliseconds (ms)
- [ ] Precision: 1 decimal place
- [ ] Includes: API call overhead + model inference
- [ ] Excludes: Prompt construction time
- [ ] Aggregation: Average across 6 queries

**Total Response Time**
- [ ] Calculation: Retrieval Latency + Generation Latency
- [ ] Unit: Milliseconds (ms)
- [ ] Precision: 1 decimal place
- [ ] Target: < 500ms for interactive use

### Output File Specifications
- [ ] File format: CSV (comma-separated)
- [ ] Output path: results/chatbot_csv.csv
- [ ] Rows: 216 (one per configuration)
- [ ] Columns: 11 (config_id, embedding_model, llm_model, temperature, top_k, query_id, accuracy_k, recall_k, mrr, retrieval_latency_ms, generation_latency_ms)
- [ ] Header row: Yes (included)
- [ ] Encoding: UTF-8
- [ ] Decimal precision: 4 places for accuracy/recall/mrr, 1 place for latency
- [ ] Missing values: None (NaN not allowed)

### Expected Results Range
- [ ] Accuracy@K range: 0.42-0.57
- [ ] Recall@K range: 0.40-0.56
- [ ] MRR range: 0.55-0.75
- [ ] Retrieval latency: 15-40 ms
- [ ] Generation latency: 200-550 ms
- [ ] Total latency: 220-580 ms

### Validation Checks
- [ ] No NaN values in output
- [ ] No infinite values in output
- [ ] All configurations present (count = 216)
- [ ] All parameter combinations valid
- [ ] All metric values in valid range
- [ ] Latency values positive
- [ ] Accuracy/Recall/MRR values in [0, 1]

---

## Verification Checklist

### Parameter Verification
- [ ] Embedding model configuration matches spec
- [ ] LLM model configuration matches spec
- [ ] Temperature settings exactly: 0.2, 0.5, 0.8
- [ ] Top-K values exactly: 1, 3, 5
- [ ] Random seed set to 42
- [ ] API timeout set to 30 seconds
- [ ] Query set is fixed (not random)
- [ ] No parameter drift between runs

### Data Verification
- [ ] Dataset contains exactly 500 FAQs
- [ ] All 4 columns present and populated
- [ ] No null values in critical fields
- [ ] Character encoding is UTF-8
- [ ] File size matches expected (~48KB)
- [ ] No duplicate rows
- [ ] Categories correctly distributed
- [ ] SourceURLs valid format

### Metric Verification
- [ ] Metric formulas correct
- [ ] Aggregation method consistent
- [ ] Latency measurements include all components
- [ ] Results within expected ranges
- [ ] CSV output valid format
- [ ] All 216 configurations present
- [ ] No data corruption or loss

### Documentation
- [ ] All parameters documented
- [ ] Expected values documented
- [ ] Data structure documented
- [ ] Metric definitions documented
- [ ] Execution steps documented
- [ ] Troubleshooting section included


