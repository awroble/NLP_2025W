from src.utils import set_seed
from .base_runner import BaseRunner

class MultiTurnRunner(BaseRunner):
    def __init__(self, models, conversations, output_dir="results"):
        """
        conversations: list of multi-turn conversations
        Each conversation is a list of prompts (dicts as per HFModel)
        """
        super().__init__(models, prompts=[], output_dir=output_dir)
        self.conversations = conversations

    def run_conversation(self, model_name, conversation, generation_params, seed):
        set_seed(seed)

        model = self.models[model_name]
        all_turns = []
        prompt_id = conversation[0].get("id", "conversation")
        conversation_id = prompt_id.rsplit("-", 1)[0]

        for prompt in conversation:
            output = model.generate(prompt, **generation_params)
            all_turns.append({
                'model_name': model_name,
                'prompt_id': prompt_id,
                'seed': seed,
                "output": output
            })

        self._save_result_to_file(all_turns, model_name, conversation_id, seed)

        model.clear_history()

        return all_turns

    def run_all(self, generation_params, seeds):
        all_results = []
        for model_name in self.models:
            for seed in seeds:
                for conversation in self.conversations:
                    result = self.run_conversation(model_name, conversation, generation_params, seed)
                    all_results.append({
                        "model_name": model_name,
                        "seed": seed,
                        "conversation": result
                    })
        return all_results
