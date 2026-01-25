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

## Generating Prompts

To generate the benchmark dataset, run the following scripts. Each script generates questions for a specific category.

```bash
# Generate factual questions
uv run python test_wiki_factual.py

# Generate questions that require context
uv run python test_wiki_insufficient.py

# Generate tricky/adversarial questions
uv run python test_wiki_tricky.py

# Generate safety-related prompts
uv run python test_wiki_safety.py
```

After generation, run the transformation script to format the data into the final `.jsonl` files.

```bash
uv run python transform_wiki_data.py
```

## Running Experiments

To evaluate models against the generated dataset:

1.  **For API-based models (e.g., GPT, Gemini):**

    Configure the `MODELS`, `DATA_CONFIG`, and `OUTPUT_FILE` variables inside `evaluate.py` and then run:

    ```bash
    uv run python evaluate.py
    ```

2.  **For local models (e.g., Llama, Mistral via Ollama):**

    Use the `evaluate_llms.ipynb` notebook to run the models and generate responses.

3.  **To judge pre-existing results:**

    If you have model outputs in `.jsonl` files, you can score them using a judge. Configure `evaluate_results.py` and run:
    
    ```bash
    uv run python evaluate_results.py
    ```

## Visualizing Results

After generating evaluation files (e.g., `gpt-5-mini-wikipedia_dataset2_evaluation.json`), you can create visualizations. Configure the `INPUT_FILE` in `analyze_data.py` and run:

```bash
uv run python analyze_data.py
```

This script creates two plots:
*   `accuracy_by_category.png`
*   `length_vs_score.png`

# Data
Dataset data or intermediate result data(including final benchmark dataset) is available on: <https://drive.google.com/drive/folders/1JlkvVn-oX4DCk52LMqPMNezpC498kB_u?usp=sharing>