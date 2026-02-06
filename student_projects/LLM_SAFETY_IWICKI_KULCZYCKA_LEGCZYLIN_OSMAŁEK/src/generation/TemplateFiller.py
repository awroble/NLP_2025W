import json
import os
import random
import re
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

random.seed(4242)
JAILBREAK = "jailbreak"
BASE = Path(__file__).resolve().parent


def prune_template(template: str, slots: list[str]) -> str:
    """
    Remove parts of template that reference placeholders not in slots. Split by ';'.
    Example:
      template = "{persona} ;requesting to {malicious_action}; as {attack_type}; to get {target}; with escalation '{escalation_subcategory}'."
      slots = ["malicious_action", "target", "escalation"]
    Result:
      "requesting to {malicious_action} to get {target} with escalation '{escalation_subcategory}'."
    """
    # Split by ';' to handle segments
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
            elif slot.removesuffix("_subcategory") in variants_json:
                continue
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
                "variant": template_json.get("variant", "unknown"),
                "scenario": scenario,
                "expected_behaviour": expected,
                "slots_filled": slot_values,
                "template_id": template_json["id"]
            }
        }

        return result

    @staticmethod
    def generate_prompts(filename_stem, n=60, neutral_frac=0.2, target_filename=None):
        """
        Generate single-turn, text prompts and metadata from templates and variants JSONs for given number of n. Save the prompts to file.
        :param filename_stem: category of prompts
        :param n: number of prompts to generate
        :param neutral_frac: fraction of safe prompts to generate
        :param target_filename: filename to save the generated prompts to
        :return: none
        """
        variants_path = os.path.join(BASE, "templates", f"{filename_stem}-variants.json")
        templates_path = os.path.join(BASE, "templates", f"{filename_stem}-templates.json")
        write_path = os.path.join(BASE, "output",
                                  f"{filename_stem}-{datetime.now().strftime('%Y%m%d_%H%M%S')}.json") if target_filename is None else target_filename

        with open(templates_path, "r", encoding="utf-8") as templates_file:
            templates = json.load(templates_file)

        with open(variants_path, "r", encoding="utf-8") as variants_file:
            variants = json.load(variants_file)

        prompts = []
        safe_templates = [t for t in templates if t["variant"] == "safe"]
        unsafe_templates = [t for t in templates if t["variant"] == "unsafe"]

        for i in range(n):
            if i < n * neutral_frac:
                prompts.append(TemplateFiller.fill_in_template(random.choice(safe_templates), variants))
            else:
                prompts.append(TemplateFiller.fill_in_template(random.choice(unsafe_templates), variants))

        with open(write_path, "w", encoding="utf-8") as f:
            f.write(json.dumps(prompts, indent=2, ensure_ascii=False))

    @staticmethod
    def prepare_templates(filename_stem):
        """
        Read file of simplified template inputs and enrich each entry there into a full template for prompt generation later. Save enriched templates to file.
        Extract multi-turn escalations from template inputs and safe to separate file for multi-turn generation later.
        :param filename_stem: category of templates
        :return: none
        """
        read_path = os.path.join(BASE, "templates", f"{filename_stem}-input.json")
        write_path = os.path.join(BASE, "templates", f"{filename_stem}-templates.json")
        write_path_escalations = os.path.join(BASE, "templates", f"{filename_stem}-escalations.json")
        with open(read_path, "r", encoding="utf-8") as f:
            input_json = json.load(f)

        enriched_templates = []
        multiturn_escalations = {}
        for subcategory, subcategory_dict in input_json.get("subcategories").items():
            for template in subcategory_dict.get("templates", []).get("single-turn", []):
                slots = re.findall(r"\{(.*?)\}", template)
                template = [{"role": "user", "text": template}]
                scenario_template = prune_template(subcategory_dict.get("scenario", "unknown"), slots)
                expected_template = prune_template(subcategory_dict.get("expected_behaviour", "unknown"), slots)
                enriched_templates.append({
                    "id": str(uuid.uuid4()),
                    "category": input_json.get("category", "unknown"),
                    "subcategory": subcategory,
                    "variant": "safe" if subcategory == "neutral" else "unsafe",
                    "modality": subcategory_dict.get("modality", "unknown"),
                    "template": template,
                    "scenario": scenario_template,
                    "expected_behaviour": expected_template,
                    "slots": slots
                })
            multiturn_escalations[subcategory] = subcategory_dict.get("templates", []).get("multi-turn-escalation", [])

        with open(write_path, "w", encoding="utf-8") as f:
            f.write(json.dumps(enriched_templates, indent=2, ensure_ascii=False))

        with open(write_path_escalations, "w", encoding="utf-8") as f:
            f.write(json.dumps(multiturn_escalations, indent=2, ensure_ascii=False))


def main():
    for category in ["sensitive-data-extraction", "social-engineering", "mental-physical-health"]:
        # Needed to only run once if the input for templates have been updated
        TemplateFiller.prepare_templates(category)

        # Generate prompts from templates for given category
        TemplateFiller.generate_prompts(category)


if __name__ == '__main__':
    main()
