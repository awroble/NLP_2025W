from src.runners import BaseRunner, MultiTurnRunner
from src.loaders import PromptLoader
from src.judge import LLMJudge

def run_single_turn_response_generation(prompt_dir, output_dir, models, generation_params, seeds):
    loader = PromptLoader(prompt_dir)  # initialize prompt loader
    prompts = loader.load_prompts()

    runner = BaseRunner(models, prompts, output_dir)
    runner.run_all(generation_params, seeds=seeds)

def run_multi_turn_response_generation(prompt_dir, output_dir, models, generation_params, seeds):
    loader = PromptLoader(prompt_dir)  # initialize prompt loader
    prompts = loader.load_prompts()

    runner = MultiTurnRunner(models, prompts, output_dir)
    runner.run_all(generation_params, seeds=seeds)