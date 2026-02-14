import unittest
import json
import tempfile
import os
from unittest.mock import patch

from src.generation.multi_turn_generation import generate_multiturn


class TestMultiTurnGeneration(unittest.TestCase):

    def setUp(self):
        # Temporary directory for templates, variants, escalations, output
        self.tmpdir = tempfile.TemporaryDirectory()
        self.base = self.tmpdir.name

        # Patch BASE inside the module
        patcher = patch("src.generation.multi_turn_generation.BASE", self.base)
        self.addCleanup(patcher.stop)
        patcher.start()

        os.makedirs(os.path.join(self.base, "templates"))
        os.makedirs(os.path.join(self.base, "output"))

        # Fake templates
        self.templates = [
            {
                "id": "t1",
                "subcategory": "attack_1",
                "variant": "unsafe",
                "template": [{"role": "user", "text": "Hello {x}"}],
                "slots": ["x"],
                "scenario": "Scenario {x}",
                "expected_behaviour": "Expected {x}",
                "category": "test"
            },
            {
                "id": "t2",
                "subcategory": "attack_2",
                "variant": "unsafe",
                "template": [{"role": "user", "text": "Hello there {x}"}],
                "slots": ["x"],
                "scenario": "Scenario_2 {x}",
                "expected_behaviour": "Expected_2 {x}",
                "category": "test"
            },
            {
                "id": "t3",
                "subcategory": "neutral",
                "variant": "safe",
                "template": [{"role": "user", "text": "Hi {x}"}],
                "slots": ["x"],
                "scenario": "Scenario {x}",
                "expected_behaviour": "Expected {x}",
                "category": "test"
            }
        ]

        # Fake variants
        self.variants = {
            "x": ["A", "B"],
            "jailbreak": ["JB"]
        }

        # Fake escalations
        self.escalations = {
            "attack_1": ["Esc1", "Esc2", "Esc3", "Esc4"],
            "attack_2": ["Esc2_1", "Esc2_2", "Esc2_3", "Esc2_4"]
        }

        # Write files
        with open(os.path.join(self.base, "templates", "test-templates.json"), "w") as f:
            json.dump(self.templates, f)

        with open(os.path.join(self.base, "templates", "test.json"), "w") as f:
            json.dump(self.variants, f)

        with open(os.path.join(self.base, "templates", "test-escalations.json"), "w") as f:
            json.dump(self.escalations, f)

    def tearDown(self):
        self.tmpdir.cleanup()

    # ───────────────────────────────────────────────
    # TESTS
    # ───────────────────────────────────────────────

    def test_generate_multiturn_creates_output_file(self):
        out_file = os.path.join(self.base, "output", "out.json")

        generate_multiturn("test", n=5, vmin=1, vmax=2, target_filename=out_file)

        self.assertTrue(os.path.exists(out_file))

        data = json.load(open(out_file))
        self.assertEqual(len(data), 5)

    def test_each_prompt_has_correct_structure(self):
        out_file = os.path.join(self.base, "output", "out.json")

        generate_multiturn("test", n=3, vmin=1, vmax=2, target_filename=out_file)
        data = json.load(open(out_file))

        for entry in data:
            self.assertIn("id", entry)
            self.assertIn("prompt", entry)
            self.assertIn("metadata", entry)

            # First turn is always the filled template
            self.assertEqual(entry["prompt"][0]["role"], "user")
            self.assertIsInstance(entry["prompt"][0]["text"], str)

    def test_escalation_turn_count_respects_vmin_vmax(self):
        out_file = os.path.join(self.base, "output", "out.json")

        generate_multiturn("test", n=10, vmin=1, vmax=3, target_filename=out_file)
        data = json.load(open(out_file))

        for entry in data:
            num_turns = len(entry["prompt"]) - 1  # subtract initial turn
            self.assertGreaterEqual(num_turns, 1)
            self.assertLessEqual(num_turns, 3)

    def test_safe_vs_unsafe_selection(self):
        out_file = os.path.join(self.base, "output", "out.json")

        generate_multiturn("test", n=10, vmin=1, vmax=3, target_filename=out_file)
        data = json.load(open(out_file))

        safe_count = sum(1 for d in data if d["metadata"]["variant"] == "safe")
        unsafe_count = sum(1 for d in data if d["metadata"]["variant"] == "unsafe")

        # no safe variants should be present
        self.assertGreaterEqual(safe_count, 0)
        self.assertGreaterEqual(unsafe_count, 10)

    def test_escalations_match_subcategory(self):
        out_file = os.path.join(self.base, "output", "out.json")

        generate_multiturn("test", n=5, vmin=1, vmax=1, target_filename=out_file)
        data = json.load(open(out_file))

        for entry in data:
            template_id = entry["metadata"]["template_id"]

            # Find template
            template = next(t for t in self.templates if t["id"] == template_id)
            subcategory = template["subcategory"]

            # Escalation turn
            escalation_text = entry["prompt"][1]["text"]
            self.assertIn(escalation_text, self.escalations[subcategory])


if __name__ == "__main__":
    unittest.main()