# System Prompt Deconstruction - Analysis

This module analyzes the preprocessed system prompts. It computes a range of indicators to quantify aspects such as size, readability, statement composition, and regulative components. Additionally, it performs embeddings analysis using Sentence-BERT (SBERT) to cluster prompts, reduce dimensionality, and generate visualizations including interactive 2D/3D plots and dendrograms.

## Module Structure

- `main.py`: Entry point to run the full analysis pipeline.
- `analysis.py`: Functions to analyze individual prompts and compute indicators.
- `config.py`: Configuration file with dataset paths.
- `embeddings.py`: Handles embeddings, dimensionality reduction, clustering, and plotting.
- `indicators.py`: Definitions of all prompt indicators.
- `metadata.py`: Metadata (provider, model, version, and release date) dictionary for system prompts of selected subset of main providers.
- `plots.py`: Plotting functions for visualizations.
- `statements_type.py`: Classification functions for statement types (constitutive, regulative, etc.).
- `utils.py`: Utility functions for reading files and loading data.
- `requirements.txt`: List of dependencies of this module.

## Installation

1. Clone the repository:

   ```
   git clone https://github.com/mytkom/SystemPromptDeconstruction.git
   cd SystemPromptDeconstruction
   ```

2. Install the requirements:
   ```
   conda create --name systemprompts --file analysis/requirements.txt
   conda activate systemprompts
   ```

## Usage

1. Ensure the dataset directories (`CL4R1T4S/`, `CL4R1T4S-no-code/`, `CL4R1T4S-no-code-processed-gemini-flash-3_5/classification`) are present and contain the required files.

2. Run the main analysis script:

   ```
   python main.py
   ```

   This will:
   - Analyze all prompts and compute indicators.
   - Generate plots saved to `analysis/analysis_plots/`.
   - Perform embeddings analysis and show interactive visualisations.

3. View results: Check the generated plots in `analysis/analysis_plots/` and console output for summary statistics.
