import json
from pathlib import Path
from src.utils import set_seed

class BaseRunner:
    def __init__(self, models, prompts, output_dir="results"):
        self.models=models
        self.prompts=prompts
        self.output_dir=Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)

    def run_single_prompt(self, model_name, prompt, generation_params, seed, clear_history=True):
        set_seed(seed)

        model = self.models[model_name]
        output = model.generate(prompt, **generation_params)

        if clear_history:
            model.clear_history()

        prompt_id = prompt.get('id')

        result = {
            'model_name': model_name,
            'prompt_id': prompt_id,
            'seed': seed,
            'output': output
        }

        self._save_result_to_file(result, model_name, prompt_id, seed)

        return result

    def run_model(self, model_name, generation_params, seeds, multi_turn=False):
        results = []
        for seed in seeds:
            if not multi_turn:
                self.models[model_name].clear_history()

            for prompt in self.prompts:
                result = self.run_single_prompt(model_name, prompt, generation_params, seed, clear_history=not multi_turn)
                results.append(result)

        return results

    def run_all(self, generation_params, seeds, multi_turn=False):
        all_results = []
        for model_name in self.models:
            model_results = self.run_model(model_name, generation_params, seeds, multi_turn=multi_turn)
            all_results.extend(model_results)

        return all_results

    def _save_result_to_file(self, result, model_name, prompt_id, seed):

        model_dir = self.output_dir / model_name
        model_dir.mkdir(exist_ok=True, parents=True)

        filename = f'{prompt_id}_{seed}.json'
        path = model_dir / filename

        with open(path, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

