import json
import os

def transform_json_to_jsonl(input_file, output_file, category_map):
    """
    Reads a JSON file, transforms its content, and writes to a JSONL file.
    """
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Warning: Input file not found: {input_file}. Skipping.")
        return 0

    count = 0
    with open(output_file, 'w', encoding='utf-8') as f_out:
        for item in data.get("results", []):
            if "Error" in item.get("prompt", "") or "Error" in item.get("expected_response", ""):
                continue
            if not item.get("prompt") or not item.get("expected_response", ""):
                if item.get("category") != "neutral_task" and not item.get("expected_response"):
                    continue

            # Map the category name
            original_category = item.get("category", "unknown")
            new_category = category_map.get(original_category, original_category)

            new_record = {
                "id": item.get("id", ""),
                "category": new_category,
                "prompt": item.get("prompt", ""),
                "expected_response": item.get("expected_response", "")
            }
            
            f_out.write(json.dumps(new_record, ensure_ascii=False) + '\n')
            count += 1
    
    return count

def main():
    """
    Main function to orchestrate the transformation of all generated files.
    """
    source_dir = os.path.dirname(__file__)
    output_dir = os.path.join(source_dir, "wikipedia_data2")
    
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory '{output_dir}' ensured.")

    category_map = {
        "wiki_factual": "factual_accuracy",
        "adversarial_fact": "tricky",
        "insufficient_context": "insufficient_information",
        "safety_contextual": "safety",
        "neutral_task": "neutral"
    }

    transformations = {
        "wiki_questions_factual.json": "factual_accuracy.jsonl",
        "wiki_questions_tricky.json": "tricky.jsonl",
        "wiki_questions_insufficient.json": "insufficient_information.jsonl",
        "wiki_questions_safety.json": "safety.jsonl",
        "wiki_prompts_neutral.json": "neutral.jsonl"
    }

    total_records = 0
    for source_file, dest_file in transformations.items():
        input_path = os.path.join(source_dir, source_file)
        output_path = os.path.join(output_dir, dest_file)
        
        print(f"Transforming '{source_file}' to '{dest_file}'...")
        num_written = transform_json_to_jsonl(input_path, output_path, category_map)
        if num_written > 0:
            print(f"  -> Successfully wrote {num_written} records.")
            total_records += num_written

    print(f"\nTransformation complete. Total records written: {total_records}")

if __name__ == "__main__":
    main()
