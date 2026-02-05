# Test Suite for LSB-LLM-Safety-Benchmark

This directory contains unit tests for the evaluation framework.

## Running Tests

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run All Tests

```bash
pytest
```

### Run Specific Test File

```bash
pytest tests/test_evaluator.py
```

### Run with Verbose Output

```bash
pytest -v
```

### Run with Coverage

```bash
pytest --cov=evaluate --cov-report=html
```

## Test Coverage

The test suite covers:

1. **Refusal Detection** (`test_detect_refusal_*`)
   - Keyword matching
   - Case insensitivity
   - Edge cases (empty responses)

2. **Harmful Content Detection** (`test_detect_harmful_content_*`)
   - Health domain keywords
   - Misinformation domain keywords
   - Disinformation domain keywords
   - Invalid domain handling

3. **Threat Assessment Extraction** (`test_extract_threat_assessment_*`)
   - JSON format parsing
   - Regex format parsing
   - Partial format parsing
   - No match scenarios

4. **Ground Truth Alignment** (`test_check_ground_truth_alignment_*`)
   - Dict ground truth handling
   - Empty ground truth
   - Short responses
   - Type validation

5. **Helper Methods** (`test_validate_score_*`, `test_flatten_dict_*`)
   - Score validation
   - Dictionary flattening for CSV
   - Edge cases

6. **Initialization** (`test_evaluator_initialization_*`)
   - Config-based initialization
   - Backward compatibility
   - Error handling

## Test Structure

Tests use pytest fixtures to mock the model and tokenizer, avoiding the need to load actual models during testing. This makes tests fast and reliable.

## Adding New Tests

When adding new functionality, add corresponding tests:

1. Create test methods with `test_` prefix
2. Use descriptive names: `test_<method>_<scenario>`
3. Use fixtures for common setup
4. Test both success and failure cases
5. Include edge cases (empty inputs, None values, etc.)
