# Deconstructing the AI Constitution: System Prompt Analysis

This project analyzes leaked system prompts from major LLM providers using Institutional Grammar (IG). We decompose prompts into **constitutive statements** (what the model is) and **regulative statements** (what it should/may/must not do) to study how prompts evolve over time and differ across providers.

For the full methodology and findings, see [`report.pdf`](report.pdf).

## What's Here

- **`analysis/`** - Computes indicators, embeddings, and visualizations. See [`analysis/readme.md`](analysis/readme.md) for details.
- **`validation_app/`** - Streamlit web app for manually validating LLM-generated annotations. See [`validation_app/README.md`](validation_app/README.md).
- **`llm_processing_pipeline/`** - LLM-assisted pipeline for classifying statements and extracting IG components.
- **`CL4R1T4S/`** - Raw leaked system prompts from OpenAI, Anthropic, xAI, Google, Meta, and others.
- **`CL4R1T4S-no-code/`** - Same prompts with code blocks removed.
- **`CL4R1T4S-no-code-processed-gemini-flash-3_5/`** - Processed and classified statements in JSON format.
- **`extracted-system-prompts-18-12-2025/`** - Few incidentaly extracted system prompts parsed into atomic statements. PoC of using our parsing prompt for system prompt extraction.

## Quick Start

### Installation

```bash
# Clone the repo
git clone https://github.com/mytkom/SystemPromptDeconstruction.git
cd SystemPromptDeconstruction

# Install dependencies (conda recommended)
conda create --name systemprompts --file analysis/requirements.txt
conda activate systemprompts

# Or with pip
pip install -r analysis/requirements.txt
```

### Running the Analysis

```bash
cd analysis
python main.py
```

This computes indicators (length, readability, statement composition, deontic types), generates plots in `analysis/analysis_plots/`, and performs Sentence-BERT embeddings analysis with interactive visualizations.

### Using the Validation App
```bash
cd validation_app
pip install -r requirements.txt
streamlit run src/app.py
```

Then open `http://localhost:8501` to view and edit statement classifications and IG components.

## What We Analyze

- **Prompt size**: Character/word counts, Non-Code-to-Whole ratio
- **Readability**: Flesch Reading Ease scores
- **Statement composition**: Constitutive vs. regulative ratios
- **Deontic types**: Commands, permissions, prohibitions, strategies
- **Semantic clustering**: Sentence-BERT embeddings with UMAP and hierarchical clustering
- **Provider differences**: Cross-provider comparison of design patterns

## Research Questions

1. What preprocessing and annotation steps are needed to reliably decompose system prompts using Institutional Grammar?
2. Are there trends in prompt size, structure, and complexity over time?
3. Do providers differ systematically in how they formulate prompts (readability, normative structure)?

## Methodology

We use a simplified Institutional Grammar framework:
- **Constitutive statements**: Define what the system is (identity, role, capabilities)
- **Regulative statements**: Define what the system should, may, or must not do

The pipeline combines LLM-assisted preprocessing, expert validation via the Streamlit app, and systematic analysis. See the report for details.

## Key Findings

- Prompts consistently grow in size over time
- Relative decline of non-code instructional content
- Systematic differences in prompt complexity and normative style across providers
- Provider-specific signatures identifiable through embeddings analysis

## Authors

Marek Mytkowski, Weronika Gozdera, Julia Dudzińska, Wojciech Michaluk  
Supervisor: Anna Wróblewska  
Initiator: Bartosz Pieliński

## Acknowledgments

System prompts collected from the [CL4R1T4S repository](https://github.com/cl4r1t4s) and cross-referenced with independent repositories (AASP, SPL, SPaMoAT, GP) for verification.
