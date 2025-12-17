from src.loaders.prompt_loader import PromptLoader
from src.loaders.model_manager import ModelManager
from src.runners import BaseRunner, MultiTurnRunner
import yaml

with open('config/settings.yaml', 'r') as f:
    settings = yaml.safe_load(f)

seeds = settings['reproducibility']['seeds']
generation_kwargs = settings['generation']

manager = ModelManager("config/models.yaml")
models = manager.all_models()

# Example use for single turn:
loader = PromptLoader("data/prompts/jailbreak")
prompts = loader.load_prompts()

runner = BaseRunner(models, prompts)
results = runner.run_all(generation_kwargs, seeds=seeds, multi_turn=False)

# Example use for multiturn
loader = PromptLoader("data/prompts/multiturn")
prompts = loader.load_prompts()

runner = MultiTurnRunner(models, prompts)
results_MT = runner.run_all(generation_kwargs, seeds=seeds)
