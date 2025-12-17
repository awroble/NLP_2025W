import json
import os
import random
import re
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

from pyarrow import timestamp

JAILBREAK = "jailbreak"
BASE = Path(__file__).resolve().parent


def prune_template(template: str, slots: list[str]) -> str:
    """
    Remove parts of template that reference placeholders not in slots.
    Example:
      template = "{persona} requesting to {malicious_action} as {attack_type} to get {target} for {official_reason} with escalation '{escalation_subcategory}'"
      slots = ["malicious_action", "target", "escalation"]
    Result:
      "requesting to {malicious_action} to get {target}"
    """
    # Split by spaces to handle segments
    parts = template.split(';')
    pruned_parts = []

    for part in parts:
        # Find placeholders in this part
        placeholders = re.findall(r"\{(.*?)\}", part)

        if not placeholders or all(ph.removesuffix("_subcategory") in slots for ph in placeholders):
            pruned_parts.append(part)

    return "".join(pruned_parts)


def fill_template(template, **kwargs):
    """Recursively fill string formats in template dict or list."""
    if isinstance(template, str):
        return template.format(**kwargs)
    if isinstance(template, list):
        return [fill_template(t, **kwargs) for t in template]
    if isinstance(template, dict):
        return {k: fill_template(v, **kwargs) for k, v in template.items()}
    return template


class TemplateFiller:

    @staticmethod
    def fill_in_template(template_json: Dict[str, Any], variants_json: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate a filled prompt and metadata from a template and variants JSON.

        Args:
            template_json (dict): Template definition with 'template', 'slots', 'category', 'expected_behaviour' etc.
            variants_json (dict): Dictionary of slot -> list of possible values.

        Returns:
            dict: Filled prompt and metadata.
        """

        # Step 1: Sample values for each slot listed in the template
        slot_values = {}
        for slot in template_json.get("slots", []):
            if slot in variants_json:
                # If escalation is nested by subcategory, pick subcategory + value
                if isinstance(variants_json[slot], dict):
                    subcat = random.choice(list(variants_json[slot].keys()))
                    value = random.choice(variants_json[slot][subcat])
                    slot_values[slot] = value
                    slot_values[f"{slot}_subcategory"] = subcat
                else:
                    slot_values[slot] = random.choice(variants_json[slot])
            else:
                raise KeyError(f"Slot '{slot}' not found in variants JSON")

        # Step 2: Fill the template using str.format
        filled_prompt = fill_template(template_json["template"], **slot_values)
        if random.random() < 0.1:
            slot_values[JAILBREAK] = random.choice(variants_json[JAILBREAK])
            filled_prompt[0]["text"] = slot_values.get(JAILBREAK) + " " + filled_prompt[0]["text"]

        # Step 3: Derive scenario metadata automatically
        scenario = template_json["scenario"].format(**slot_values)
        expected = template_json.get("expected_behaviour", "unspecified").format(**slot_values)

        # Step 4: Build result dictionary
        result = {
            "id": str(uuid.uuid4()),
            "prompt": filled_prompt,
            "metadata": {
                "category": template_json.get("category", "unknown"),
                "scenario": scenario,
                "expected_behaviour": expected,
                "slots_filled": slot_values,
                "template_id": template_json["id"]
            }
        }

        return result

    @staticmethod
    def generate_prompts(filename_stem, n=50):
        variants_path = os.path.join(BASE, "templates", f"{filename_stem}.json")
        templates_path = os.path.join(BASE, "templates", f"{filename_stem}-templates.json")
        write_path = os.path.join(BASE, "output", f"{filename_stem}-{datetime.now().strftime("%Y%m%d_%H%M%S")}.json")

        with open(templates_path, "r", encoding="utf-8") as templates_file:
            templates = json.load(templates_file)

        with open(variants_path, "r", encoding="utf-8") as variants_file:
            variants = json.load(variants_file)

        propmpts = []

        for _ in range(n):
            propmpts.append(TemplateFiller.fill_in_template(random.choice(templates), variants))

        with open(write_path, "w", encoding="utf-8") as f:
            f.write(json.dumps(propmpts, indent=2, ensure_ascii=False))

    @staticmethod
    def prepare_templates(filename_stem):
        read_path = os.path.join(BASE, "templates", f"{filename_stem}-input.json")
        write_path = os.path.join(BASE, "templates", f"{filename_stem}-templates.json")
        with open(read_path, "r", encoding="utf-8") as f:
            input_json = json.load(f)

        enriched_templates = []
        for template_type in input_json["templates"].keys():
            for template in input_json.get("templates", []).get(template_type, []):
                slots = re.findall(r"\{(.*?)\}", template)
                template = [{"role": "user", "text": template}] if template_type == "single-turn" else template
                scenario_template = prune_template(input_json.get("scenario", "unknown"), slots)
                expected_template = prune_template(input_json.get("expected_behaviour", "unknown"), slots)
                enriched_templates.append({
                    "id": str(uuid.uuid4()),
                    "category": input_json.get("category", "unknown"),
                    "modality": input_json.get("modality", "unknown"),
                    "template": template,
                    "scenario": scenario_template,
                    "expected_behaviour": expected_template,
                    "slots": slots
                })

        with open(write_path, "w", encoding="utf-8") as f:
            f.write(json.dumps(enriched_templates, indent=2, ensure_ascii=False))


def main():
    # Needed to only run once if the input for templates have been updated
    # TemplateFiller.prepare_templates("social-engineering")

    # Generate prompts from templates for social engineering
    TemplateFiller.generate_prompts("social-engineering")

if __name__ == '__main__':
    main()
