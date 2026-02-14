# README - International Agreements Database Mining (Tasks 8 & 11)

## Overview

This notebook implements a **hybrid extraction pipeline** for mining legal agreements to extract:
- **Task 8**: Conditions for extending agreements (automatic vs mutual decision vs optional)
- **Task 11**: Evaluation of implementation obligations (review/audit/reporting requirements)

The pipeline combines rule-based retrieval, neural classification, and NLI verification for robust extraction from noisy OCR documents.

## Purpose

- Extract renewal/extension conditions from legal agreements
- Identify evaluation, monitoring, and reporting obligations
- Handle OCR noise and document variability
- Provide structured output with evidence and confidence scores
- Enable reproducible research with comprehensive profiling

## Code Structure

### Directory Layout
```
d:/NLP_Project_tasks_6_8_11/
├── tasks_8_11.ipynb                 # Main notebook
├── OCR_output/                      # Input documents by state
│   ├── Alabama/
│   ├── California/
│   └── ... (other states)
├── tables/                          # Output results
│   ├── tasks8_11_all_agreements.csv
│   ├── tasks8_11_all_agreements.xlsx
│   ├── tasks8_11_all_agreements_evidence_full.jsonl
│   ├── profiling_tasks8_11_full_dataset.csv
│   └── profiling_tasks8_11_full_dataset.xlsx
└── TreeTagger/                      # (Optional, not used in this pipeline)
```

### Notebook Sections

1. **Configuration & Model Loading**
   - Device detection (GPU/CPU)
   - Memory-safe model loading with fallbacks
   - Threshold and parameter settings

2. **Data I/O & Preprocessing**
   - OCR text loading (TXT and JSON formats)
   - Page-aware text segmentation
   - Text normalization and cleaning

3. **Candidate Retrieval**
   - High-recall regex patterns for renewal and evaluation
   - Neighbor expansion for context
   - Temporal expression extraction

4. **Neural Classification**
   - Zero-shot classification with MNLI models
   - NLI verification layer
   - Evidence scoring and ranking

5. **Processing Pipeline**
   - Document-level extraction
   - Evidence aggregation
   - Performance profiling

## Reproducibility Checklist

### System Requirements
- **Python**: 3.8+ with transformer support
- **Memory**: Minimum 8GB RAM (16GB recommended for large datasets)
- **Storage**: 2GB+ for model downloads and outputs
- **OS**: Windows/Linux/macOS (tested on Windows)

### Software Dependencies

#### Core Python Packages
```python
# Required packages
torch>=1.9.0
transformers>=4.20.0
pandas>=1.3.0
numpy>=1.21.0
tqdm>=4.60.0
psutil>=5.8.0
dateparser>=1.1.0
python-dateutil>=2.8.0
```

#### Model Dependencies
```python
# Automatically downloaded by transformers
typeform/distilbert-base-uncased-mnli  # Primary model
valhalla/distilbart-mnli-12-1          # Fallback model
```

### Configuration Parameters

#### Device & Model Settings
```python
DEVICE = -1                           # -1=CPU, 0=GPU
USE_FINETUNED_SEQCLS = False          # Custom fine-tuned model (if available)
USE_ZEROSHOT_FALLBACK = True          # Use MNLI as fallback
ZEROSHOT_MODEL = "typeform/distilbert-base-uncased-mnli"
USE_NLI_VERIFIER = True               # Enable NLI verification
NLI_MODEL = "typeform/distilbert-base-uncased-mnli"
```

#### Processing Parameters
```python
THRESH_RENEWAL = 0.60                 # Confidence threshold for renewal
THRESH_EVAL = 0.60                    # Confidence threshold for evaluation
NEIGHBOR_K = 2                        # Context expansion window
MAX_BLOCKS = 24                       # Maximum candidate blocks
MAX_SENTS = 80                         # Maximum candidate sentences
MAX_CANDIDATES = 50                    # Maximum candidates per document
EVIDENCE_MAX_CHARS = 240              # Evidence snippet length
```

#### Retrieval Patterns
```python
# Renewal patterns (high-recall)
RENEWAL_PATTERNS = [
    r"\brenew(al|ed|s|ing)?\b",
    r"\bextend(ed|s|ing)?\b",
    r"\bextension\b",
    r"\bautomatic(ally)?\s+renew\b",
    r"\bshall\s+be\s+renewed\b",
    r"\bmay\s+be\s+renewed\b",
    # ... additional patterns
]

# Evaluation patterns
EVAL_PATTERNS = [
    r"\bevaluat(e|ion|ing)\b",
    r"\breview(s|ed|ing)?\b",
    r"\bassess(ment|es|ed|ing)?\b",
    r"\bmonitor(ing|ed|s)?\b",
    r"\baudit(s|ed|ing)?\b",
    r"\breport(s|ed|ing)?\b",
    # ... additional patterns
]
```

### Data Requirements

#### Input Formats

##### TXT Files
- **Location**: `OCR_output/<STATE>/*.txt`
- **Encoding**: UTF-8
- **Content**: Plain text OCR output
- **Structure**: One agreement per file

##### JSON Files (Optional)
- **Format**: Structured OCR with page information
- **Schema**: `{"pages": [{"blocks": [{"lines": [{"words": [...]}]}]}]}`
- **Page Markers**: Automatically inserted as `[[PAGE=N]]`

#### Expected Dataset Structure
```
OCR_output/
├── Alabama/
│   ├── agreement1.txt
│   ├── agreement2.json
│   └── ...
├── California/
│   ├── ca_agreement1.txt
│   └── ...
└── [other states]/
```

### Execution Steps

#### 1. Environment Setup
```python
# Check device availability
try:
    import torch
    DEVICE = 0 if torch.cuda.is_available() else -1
except Exception:
    DEVICE = -1

# Load models with fallbacks
zsc = make_zsc(CLAUSE_MODEL, DEVICE)
if USE_NLI_VERIFIER:
    nli = make_zsc(NLI_MODEL, DEVICE)
```

#### 2. Document Processing
```python
# Load and normalize text
if path.lower().endswith(".txt"):
    raw_text = load_txt(path)
else:
    raw_text = load_ocr_json_with_pages(path)

# Process document
result = process_document_8_11(doc_id, path, raw_text)
```

#### 3. Full Dataset Processing
```python
# Process all documents with profiling
with RunProfile(label="Tasks8_11_full_dataset") as prof:
    final_df = run_poc_8_11(OCR_ROOT, n=None, strategy="all", profiler=prof)
```

### Output Specifications

#### Generated Files

1. **Main Results**: `tasks8_11_all_agreements.csv`
   - Document-level extractions
   - Renewal types and conditions
   - Evaluation obligations
   - Evidence snippets (shortened)

2. **Excel Workbook**: `tasks8_11_all_agreements.xlsx`
   - Same data as CSV in Excel format
   - Easier for manual review and filtering

3. **Full Evidence**: `tasks8_11_all_agreements_evidence_full.jsonl`
   - Complete evidence objects (JSONL format)
   - Full text spans and metadata
   - Detailed confidence scores

4. **Performance Metrics**: `profiling_tasks8_11_full_dataset.csv/xlsx`
   - Runtime statistics
   - Memory usage tracking
   - Processing efficiency metrics

#### Output Schema

##### Main Results Table
```python
{
    "doc_uid": "Alabama/Alabama_1",
    "state": "Alabama", 
    "doc_id": "Alabama_1",
    "source_path": "OCR_output/Alabama/Alabama_1.txt",
    
    # Task 8 - Renewal/Extension
    "renewal_type": "automatic|by_mutual_agreement|unilateral_option|absent|uncertain",
    "notice_period": "30 days|90 days|None",
    "renewal_status": "found|uncertain|absent",
    "renewal_evidence": "shortened evidence string...",
    
    # Task 11 - Evaluation
    "evaluation": "present|uncertain|absent",
    "evaluation_status": "found|uncertain|absent", 
    "evaluation_evidence": "shortened evidence string...",
    
    # Baselines
    "baseline_keyword_renewal": True/False,
    "baseline_keyword_eval": True/False
}
```

##### Full Evidence JSONL
```python
{
    "doc_uid": "Alabama/Alabama_1",
    "state": "Alabama",
    "doc_id": "Alabama_1", 
    "source_path": "OCR_output/Alabama/Alabama_1.txt",
    "renewal_evidence": [
        {
            "text": "full evidence text...",
            "page": 1,
            "sid": 15,
            "bid": null,
            "source": "sentence",
            "label": "Automatic renewal unless terminated...",
            "score": 0.85,
            "verified": true,
            "verify_score": 0.78
        }
    ],
    "evaluation_evidence": [...]
}
```

### Performance Benchmarks

#### Test Environment
- **Dataset**: 298 documents across 9 states
- **Total Processing Time**: ~35 minutes
- **Average Time/Document**: 6.96 seconds
- **Memory Usage**: Peak 856MB RSS, 2.9MB Python heap
- **Model Loading**: ~30 seconds initial load

#### Expected Results
- **Renewal Detection**: ~60% of agreements contain renewal clauses
- **Evaluation Obligations**: ~40% contain evaluation requirements
- **Confidence Scores**: Average 0.75 for verified evidence
- **Error Rate**: <2% processing failures

### Model Architecture

#### Pipeline Components

1. **Candidate Retrieval**
   - Regex-based high-recall patterns
   - Neighbor expansion (±K sentences)
   - Temporal expression extraction

2. **Neural Classification**
   - Zero-shot classification with MNLI
   - Multi-label prediction
   - Confidence scoring

3. **NLI Verification**
   - ContractNLI-style hypothesis testing
   - Entailment vs contradiction detection
   - Threshold-based filtering

4. **Evidence Aggregation**
   - Page-aware provenance tracking
   - Evidence ranking by confidence
   - Snippet generation for output

#### Classification Labels

##### Renewal Types
```python
RENEWAL_LABELS = [
    "Automatic renewal unless terminated or notice is given",
    "Renewal/extension requires mutual agreement", 
    "Unilateral renewal/extension option",
    "Other"
]
```

##### Evaluation Types
```python
EVAL_LABELS = [
    "Evaluation/monitoring/reporting/audit/review obligation",
    "Other"
]
```

### Troubleshooting

#### Common Issues

1. **Model Loading Errors**
   - **Symptom**: OSError or memory errors during model loading
   - **Solution**: Ensure sufficient RAM, try smaller models, use CPU mode

2. **CUDA Out of Memory**
   - **Symptom**: GPU memory errors during processing
   - **Solution**: Set DEVICE=-1 for CPU processing, reduce batch sizes

3. **Slow Processing**
   - **Symptom**: Very slow document processing
   - **Solution**: Use GPU if available, reduce MAX_SENTS/MAX_BLOCKS

4. **Encoding Issues**
   - **Symptom**: Unicode decode errors
   - **Solution**: Ensure UTF-8 encoding, check file integrity

#### Performance Optimization

1. **Memory Management**
   - Automatic garbage collection every 50 documents
   - Tracemalloc for Python heap monitoring
   - RSS memory tracking with psutil

2. **Processing Efficiency**
   - Batch processing where possible
   - Model caching to avoid reloading
   - Early termination for low-confidence results

### Validation & Quality Assurance

#### Quality Metrics
- **Precision**: High confidence thresholds (0.60)
- **Recall**: High-recall retrieval patterns
- **F1-Score**: Balanced precision/recall trade-off
- **Evidence Quality**: Page-level provenance tracking

#### Error Handling
- Graceful fallback for model loading failures
- Robust text preprocessing for OCR noise
- Comprehensive logging of processing errors
- Automatic recovery from malformed documents

### Extension & Customization

#### Adding New Tasks
1. Define new regex patterns for candidate retrieval
2. Add new classification labels
3. Create NLI hypotheses for verification
4. Extend the processing pipeline

#### Model Fine-Tuning
```python
# Replace zero-shot with fine-tuned model
USE_FINETUNED_SEQCLS = True
FINETUNED_SEQCLS_MODEL = "path/to/your/model"
```

#### Custom Patterns
```python
# Add domain-specific patterns
CUSTOM_PATTERNS = [
    r"\byour_custom_pattern\b",
    r"\banother_pattern\b"
]
```

## Citation & Attribution

When using this pipeline, please cite relevant works:
- Transformers library: Wolf et al. (2020). "Transformers: State-of-the-art Natural Language Processing"
- MNLI: Williams et al. (2018). "A Broad-Coverage Challenge Corpus for Sentence Understanding through Inference"
- ContractNLI: Koren & Roth (2021). "A Dataset for Natural Language Inference for Contracts"

## Contact & Support

For issues or questions:
1. Check model compatibility and system requirements
2. Verify input data format and encoding
3. Review configuration parameters
4. Examine notebook outputs for specific error messages
5. Consider reducing dataset size for initial testing
