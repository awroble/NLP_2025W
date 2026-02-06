import json
import os
import random
from datetime import datetime
from pathlib import Path

from src.generation.TemplateFiller import TemplateFiller

random.seed(4242)
BASE = Path(__file__).resolve().parent


def generate_multiturn(filename_stem, n=20, vmin=1, vmax=5, target_filename=None):
    """
    Generate multi-turn prompts based on single-turn, unsafe templates and escalations. Save multi-turn prompts to file.
    :param filename_stem: category of prompts
    :param n: number of multi-turn prompts to generate
    :param vmin: min. escalation turns (counting from the 2nd turn)
    :param vmax: max. escalation turns (counting from the 2nd turn)
    :param target_filename: filename to save the generated prompts to
    :return: none
    """
    assert isinstance(vmin, int) and vmin >= 1
    assert isinstance(vmax, int) and vmin <= vmax

    variants_path = os.path.join(BASE, "templates", f"{filename_stem}-variants.json")
    templates_path = os.path.join(BASE, "templates", f"{filename_stem}-templates.json")
    escalations_path = os.path.join(BASE, "templates", f"{filename_stem}-escalations.json")
    write_path = os.path.join(
        BASE, "output", f"{filename_stem}-multiturn.json"
    ) if target_filename is None else target_filename

    with open(templates_path, "r", encoding="utf-8") as templates_file:
        templates = json.load(templates_file)

    with open(variants_path, "r", encoding="utf-8") as variants_file:
        variants = json.load(variants_file)

    with open(escalations_path, "r", encoding="utf-8") as escalations_file:
        escalations = json.load(escalations_file)

    prompts = []

    unsafe_templates = [t for t in templates if t["variant"] == "unsafe"]

    for i in range(n):
        selected_template = random.choice(unsafe_templates)

        num_turns = random.randint(vmin, vmax)
        prompt = TemplateFiller.fill_in_template(selected_template, variants)
        multiturn = [{
            "role": "user",
            "text": prompt["prompt"][0]["text"],
        }]
        turns = random.sample(escalations.get(selected_template["subcategory"]), num_turns)
        for turn in turns:
            multiturn.append({
                "role": "user",
                "text": turn,
            })

        prompts.append(
            {
                "id": prompt["id"],
                "prompt": multiturn,
                "metadata": {
                    "category": prompt["metadata"]["category"],
                    "variant": prompt["metadata"]["variant"],
                    "scenario": prompt["metadata"]["scenario"],
                    "expected_behaviour": prompt["metadata"]["expected_behaviour"],
                    "slots_filled": prompt["metadata"]["slots_filled"],
                    "template_id": prompt["metadata"]["template_id"],
                },
            }
        )

    with open(write_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(prompts, indent=2, ensure_ascii=False))


def main():
    generate_multiturn("mental-physical-health")
    generate_multiturn("sensitive-data-extraction")
    generate_multiturn("social-engineering")

    return


if __name__ == "__main__":
    main()
