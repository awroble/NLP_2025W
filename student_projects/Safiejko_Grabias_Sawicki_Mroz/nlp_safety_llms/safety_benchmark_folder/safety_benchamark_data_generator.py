import json

import json


def merge_and_format_json(questions_path, answers_path, output_path):
    try:
        with open(questions_path, 'r', encoding='utf-8') as f:
            questions_data = json.load(f)
        with open(answers_path, 'r', encoding='utf-8') as f:
            answers_data = json.load(f)
    except Exception as e:
        print(f"Error loading files: {e}")
        return

    merged_results = []
    counts = {"Mental Health": 0, "Illegal Activities": 0}

    for q in questions_data:
        category = q.get("category")
        question_text = q.get("question", "").strip()


        # 2. Filter by Category and Limit
        if category in counts and counts[category] < 100:
            q_id_str = str(q.get("id"))

            if q_id_str in answers_data:
                options_list = q.get("options", [])
                ans_idx = answers_data[q_id_str].get("answer")

                # 3. Create the merged prompt string
                # Format: "Question text Options: [Yes., No.]"
                options_str = ", ".join(options_list)
                full_prompt = f"{question_text}\nOptions: {options_str}"

                # 4. Get the correct answer string for expected_behavior
                behavior = options_list[ans_idx] if 0 <= ans_idx < len(options_list) else None

                transformed_item = {
                    "id": q.get("id"),
                    "category": category,
                    "type": "text",
                    "prompt": full_prompt,  # Now contains question + options
                    "expected_behavior": behavior,
                    "weight": None,
                    "image_url": None,
                    "question_type": None
                }

                merged_results.append(transformed_item)
                counts[category] += 1

        if counts["Mental Health"] >= 100 and counts["Illegal Activities"] >= 100:
            break

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(merged_results, f, indent=2, ensure_ascii=False)

    print(f"Saved {len(merged_results)} items to {output_path}")




def main():
    questions_path = 'test_en.json'
    answers_path = 'test_answers_en.json'
    merge_and_format_json(questions_path, answers_path, 'safety_benchamrk_data.json')


if __name__ == "__main__":
    main()