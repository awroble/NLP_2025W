# nlp

NLP

## Repository structure

```bash
-- nlp
   |
   |-- POC/                   # Proof of concept code and results
   |-- safety-bench/          # Safety benchmark code and data
   |-- safety-bench/results-analysis/  # Full analysis of safety benchmark results
   |-- safety-bench/images    # Disinformation/offensive images part of the benchmark
   |-- safety-bench/nlp.xlsx  # Safety benchmark prompts for all 3 categories
   |-- safety-bench/results   # Safety benchmark results - all prompts/responses/scores and judge reasons
   |-- safety-bench/results/hardware # Hardware-specific metrics collected during benchmark runs
```

## Reproducibility of POC

Assuming there is local ollama and garak installations:

```bash
ollama pull qwen3:0.6b
ollama serve
```

```bash
garak --config safety_3areas.yml
```

## Reproducibility of safety bench: cultural, disinformative, offensive

```bash
cd safety-bench
cp .env.example .env
# edit .env to set OPENAI_API_KEY and OLLAMA_MODEL if needed
uv run score_prompts.py --input nlp.xlsx --output scored_prompts.xlsx --subset disinformation
```
