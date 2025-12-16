# LLM Hallucination Evaluation

This project evaluates Large Language Models for hallucinations on a custom dataset.

## Prerequisites

*   Python >= 3.13
*   `uv` for environment and package management.

Install the dependencies from `pyproject.toml`:

```bash
uv pip install -e .
```

## Setup

You need to provide API keys for the different LLM providers.

1.  Copy the environment template file:

    ```bash
    cp .env.template .env
    ```

2.  Edit the `.env` file and add your API keys:

    ```bash
    HF_TOKEN=your_hugging_face_token
    GEMINI_API_KEY=your_gemini_api_key
    OPENAI_API_KEY=your_openai_api_key
    ```

## Data

The project uses two main data sources:

*   `dummy_data`: A small, dummy dataset for quick tests and demonstration purposes.
*   `dataset_data`: A dataset, which consists of data from [DefiniteAnswer](https://huggingface.co/datasets/iamasQ/DefAn) and [TruthfulQA](https://huggingface.co/datasets/domenicrosati/TruthfulQA).

## Running Experiments

To run the evaluation script, use the following command:

```bash
uv run python evaluate.py
```

This will run the models against the dataset and save the results in `dataset_evaluation.json`.

## Generating Charts

After the experiment has finished, you can generate charts from the results.

```bash
uv run python analyze_data.py
```

This script reads `dataset_evaluation.json` and creates two plots:
*   `accuracy_by_category.png`
*   `length_vs_score.png`
