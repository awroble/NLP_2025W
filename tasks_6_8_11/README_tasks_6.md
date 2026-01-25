# README - HeidelTime Standalone + TreeTagger (Task 6)

## Overview

This notebook implements a complete temporal information extraction pipeline using **HeidelTime Standalone** with and without **TreeTagger** POS tagging. It performs ablation studies comparing both approaches on OCR-processed legal agreements.

## Purpose

- Extract temporal expressions (TIMEX3) from legal agreement documents
- Compare HeidelTime performance with and without POS tagging
- Generate structured temporal data for downstream analysis
- Provide comprehensive performance metrics and reproducibility data

## Code Structure

### Directory Layout
```
d:/NLP_Project_tasks_6_8_11/
├── tasks_6.ipynb                    # Main notebook
├── heideltime-standalone/           # HeidelTime Java application
│   ├── de.unihd.dbs.heideltime.standalone.jar
│   ├── config.props
│   └── config.props.patched        # TreeTagger-enabled config
├── OCR_output/                      # Input documents by state
│   ├── Alabama/
│   ├── California/
│   └── ... (other states)
├── TreeTagger/                      # POS tagger (Windows)
│   ├── bin/tree-tagger.exe
│   ├── lib/english.par
│   └── lib/english-abbreviations
├── tables/                          # Output results
│   ├── full_dataset_treetagger_results.csv
│   ├── full_dataset_treetagger_time_memory_per_doc.csv
│   └── full_dataset_treetagger_ALL.xlsx
└── heideltime_cache/                # Optional caching directory
```

### Notebook Sections

1. **Environment Setup & Prerequisites**
   - Java validation
   - TreeTagger installation verification
   - Path configuration

2. **TreeTagger Configuration**
   - Config file patching for Windows paths
   - Validation of TreeTagger components

3. **HeidelTime Execution**
   - Single file testing
   - Robust TimeML parsing
   - Error handling for mixed output

4. **Full Dataset Processing**
   - Batch processing across all states
   - Memory and performance profiling
   - Validity extraction from temporal expressions

## Reproducibility Checklist

### System Requirements
- **Java**: OpenJDK 17+ (tested with Temurin 25.0.1)
- **Python**: 3.8+ with required packages
- **OS**: Windows (TreeTagger paths configured for Windows)
- **Memory**: Minimum 4GB RAM (8GB recommended for large datasets)

### Software Dependencies
```python
# Core Python packages
pandas>=1.3.0
xml.etree.ElementTree (built-in)
pathlib (built-in)
subprocess (built-in)
re (built-in)
datetime (built-in)
tqdm>=4.60.0
psutil>=5.8.0  # For memory profiling
```

### External Tools Setup

#### HeidelTime Standalone
1. Download HeidelTime Standalone JAR
2. Place in `heideltime-standalone/` directory
3. Ensure `config.props` is present

#### TreeTagger (Required for POS mode)
1. Download TreeTagger for Windows
2. Extract to `TreeTagger/` directory
3. Download English parameter file (PENN tagset)
4. Extract `english.par.gz` to `TreeTagger/lib/english.par`

### Configuration Parameters

#### HeidelTime Settings
```python
LANG = "ENGLISH"                    # Language setting
DOCTYPE = "NARRATIVES"              # Document type
OUTPUT = "TIMEML"                   # Output format
DCT = datetime.date.today()        # Document creation time
```

#### Processing Parameters
```python
STATE = "California"                # Target state for processing
USE_CACHE = False                   # Enable/disable caching
EVIDENCE_MAX_CHARS = 120           # Evidence snippet length
```

#### TreeTagger Path
```python
TREETAGGER_HOME = Path(r"D:\NLP_Project_tasks_6_8_11\TreeTagger")
```

### Data Requirements

#### Input Format
- **Location**: `OCR_output/<STATE>/*.txt`
- **Encoding**: UTF-8
- **Content**: OCR-processed legal agreements
- **Structure**: One agreement per text file

#### Expected Dataset Structure
```
OCR_output/
├── Alabama/
│   ├── agreement1.txt
│   └── agreement2.txt
├── California/
│   ├── ca_agreement1.txt
│   └── ca_agreement2.txt
└── ...
```

### Execution Steps

#### 1. Environment Validation
```python
# Check Java installation
ok_java, java_out = check(["java", "-version"])

# Validate TreeTagger components
validate_treetagger_install(TREETAGGER_HOME)
```

#### 2. Configuration Patching
```python
# Create TreeTagger-enabled config
patched_conf = patch_config_for_treetagger(
    HT_CONF, TREETAGGER_HOME, key="treeTaggerHome"
)
```

#### 3. Single File Test
```python
# Test HeidelTime on one document
example = txt_files[0]
out_no = run_heideltime(example, pos_mode="NO", conf_path=HT_CONF)
df_preview = parse_timeml_timex3(out_no)
```

#### 4. Full Dataset Processing
```python
# Process all documents with profiling
final_df = run_poc_6_full_dataset(OCR_ROOT)
```

### Output Specifications

#### Generated Files
1. **Results CSV**: `full_dataset_treetagger_results.csv`
   - Document-level temporal extractions
   - Validity information (effective_date, end_date, duration)
   - Evidence snippets

2. **Performance Metrics**: `full_dataset_treetagger_time_memory_per_doc.csv`
   - Processing time per document
   - Memory usage statistics
   - Error tracking

3. **State Aggregates**: `full_dataset_treetagger_time_memory_by_state.csv`
   - Performance metrics by state
   - Document counts and error rates

4. **Summary**: `full_dataset_treetagger_time_memory_summary.csv`
   - Overall dataset statistics
   - Performance benchmarks

5. **Excel Workbook**: `full_dataset_treetagger_ALL.xlsx`
   - All data in separate sheets
   - Easy for manual review and analysis

#### Output Schema
```python
{
    "doc_id": "California/agreement1",
    "state": "California",
    "effective_date": "2007-11-01",
    "end_date": "2032-01-01", 
    "duration": "three years",
    "validity_status": "found",
    "evidence_short": "Agreement effective November 1, 2007...",
    "time_sec": 2.75,
    "py_peak_mem_mb": 0.12
}
```

### Performance Benchmarks

#### Test Environment
- **Dataset**: 298 documents across 9 states
- **Total Processing Time**: ~15 minutes
- **Average Time/Document**: 3.0 seconds
- **Peak Memory Usage**: 0.55 MB per document
- **Error Rate**: <1%

#### Expected Results
- **California**: 116 documents, ~3.0s avg processing time
- **Other States**: 1-17 documents each, 2.3-3.2s avg processing time

### Troubleshooting

#### Common Issues
1. **Java not found**: Install JDK and restart IDE
2. **TreeTagger missing**: Download and configure TreeTagger paths
3. **Memory errors**: Reduce batch size or increase system RAM
4. **Encoding issues**: Ensure UTF-8 encoding for input files

#### Error Recovery
- Automatic fallback for missing TreeTagger components
- Robust XML parsing for mixed HeidelTime output
- Graceful handling of malformed documents

### Validation & Quality Assurance

#### Data Quality Checks
- XML sanitization for illegal characters
- TimeML block extraction with regex fallback
- Date normalization and validation
- Evidence snippet truncation for readability

#### Performance Monitoring
- Real-time memory tracking with psutil
- Processing time profiling
- Error logging and categorization
- Cache hit/miss statistics (when enabled)

## Citation & Attribution

When using this pipeline, please cite:
- HeidelTime: Strötgen, J. & Gertz, M. (2010). "HeidelTime: High quality rule-based extraction and normalization of temporal expressions."
- TreeTagger: Schmid, H. (1994). "Probabilistic Part-of-Speech Tagging Using Decision Trees."

## Contact & Support

For issues or questions regarding this implementation:
1. Check the troubleshooting section above
2. Verify all path configurations
3. Ensure all external tools are properly installed
4. Review the notebook cell outputs for specific error messages
