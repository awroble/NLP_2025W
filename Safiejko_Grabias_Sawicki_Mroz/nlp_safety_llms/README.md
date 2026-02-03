# NLP Safety LLMs

## Description
This project implements safety evaluation of Large Language Models using jailbreak prompts and multi-judge voting system.

## Installation

1. Clone the repository:
```bash
git clone https://github.com/ssafiejko/nlp_safety_llms.git
cd nlp_safety_llms
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download models from Google Drive:
[**Google Drive Link**](https://l.messenger.com/l.php?u=https%3A%2F%2Fdrive.google.com%2Fdrive%2Ffolders%2F1VJ2CuM0ir_tt1h-6tnF2r_JDN6rl5T0R%3Fusp%3Dsharing&h=AT1jXXY8l0aw9TXdMnYJmj_tkjwyHTE3URcUGo3ij4uW_KT6qQw0lXaUriQoP3ChpjmckIptAvFGb-hVoe0MXuhcQfHqyZrZewRKikIbZGcExJ2LHs8x2ulJszYOBx0MFUjubUG5B3w)


Place the downloaded models in the `models/` directory.

## Usage

1. In `benchmark_processor.py`, configure the following variables:
   - `INPUT_DATASET` - Path to your input dataset (e.g., `"dataset_poc.json"` for the example dataset, or your own custom dataset)
   - `OUTPUT_FILE` - Path where the model responses will be saved

2. Run the evaluation:
```bash
python benchmark_processor.py
```

For detailed analysis, check the Jupyter notebooks in the repository.

## Project Structure
- `benchmark_processor.py` - Main evaluation script
- `EDA.ipynb` - Exploratory data analysis
- `requirements.txt` - Python dependencies
- `models/` - Directory for model files

## Folder structure
- `input_folder` - consist of prompt data stored in benchmark_input.json
-  folder input_imgs consists of images that were used for dataset_multimodal
- `safety_benchamrk_folder` - used to store data for input and output of Hallulens and Safetybench benchmarks

## Requirements
- Python 3.8+
- all libraries from `requirements.txt` files
