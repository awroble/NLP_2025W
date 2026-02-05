import json
from pathlib import Path
from src.utils import set_seed
from src.utils import profile_gpu

class BaseRunner:
    """
    Class to run models with single turn response generation.
    """
    def __init__(self, models, prompts, output_dir="results"):
        """
        :param models: Dictionary with HFModel class models with key as name from config file and value as HFModel.
        :param prompts: List of prompts.
        :param output_dir: Name of output directory. Defaults to 'results'.
        """
        self.models = models
        self.prompts = prompts
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)

    @profile_gpu
    def run_single_prompt(self, model_name, prompt, generation_params, seed, clear_history=True):
        """
        Method to run a single prompt on model.
        :param model_name: Model name as it is referenced in config/models.yaml
        :param prompt: Dictionary with prompt.
        :param generation_params: Dictionary with generation parameters.
        :param seed: Seed for reproducibility.
        :param clear_history: Whether to clear history. Defaults to True.
        :return: Model response.
        """
        set_seed(seed)

        model = self.models[model_name]
        output = model.generate(prompt, pad_token_id=model.tokenizer.eos_token_id, **generation_params)

        if clear_history:
            model.clear_history()

        prompt_id = prompt.get('id')

        result = {
            'model_name': model_name,
            'prompt_id': prompt_id,
            'seed': seed,
            'prompt': prompt['prompt'],
            'output': output,
            'generation_params': generation_params,
        }

        return result

    def run_model(self, model_name, generation_params, seeds):
        """
        Method to run a single turn response generation on all prompts. Saves responses to files.
        :param model_name: Model name as it is referenced in config/models.yaml
        :param generation_params: Dictionary with generation parameters.
        :param seeds: Seeds for which to generate responses.
        :return: List of model responses.
        """
        results = []
        for seed in seeds:

            self.models[model_name].clear_history()

            for prompt in self.prompts:
                result = self.run_single_prompt(model_name, prompt, generation_params, seed, clear_history=False)

                prompt_id = prompt.get('id')
                self._save_result_to_file(result, model_name, prompt_id, seed)
                results.append(result)

        return results

    def run_all(self, generation_params, seeds):
        """
        Method to run all prompts on all models. Saves responses to files.
        :param generation_params: Dictionary with generation parameters.
        :param seeds: Seeds for which to generate responses.
        :return: List of model responses.
        """
        all_results = []
        for model_name in self.models:
            model_results = self.run_model(model_name, generation_params, seeds)
            all_results.extend(model_results)

        return all_results

    def _save_result_to_file(self, result, model_name, prompt_id, seed):
        """
        Method to save results to files.
        :param result: Model response.
        :param model_name: Model name as it is referenced in config/models.yaml
        :param prompt_id: Prompt id as it is referenced in config/prompts.json
        :param seed: Random seed."""
        model_dir = self.output_dir / model_name
        model_dir.mkdir(exist_ok=True, parents=True)

        filename = f'{prompt_id}_{seed}.json'
        path = model_dir / filename

        with open(path, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

