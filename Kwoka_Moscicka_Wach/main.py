import yaml
from src.loaders import ModelManager
from src.judge import LLMJudge
from src.utils import run_single_turn_response_generation, run_multi_turn_response_generation

config_settings_path = 'config/settings.yaml'
config_models_path = 'config/models.yaml'

# load settings file
with open(config_settings_path, 'r') as f:
    settings = yaml.safe_load(f)

# load settings for model response generation
response_seeds = settings['model_response']['seeds'] #
response_generation = settings['model_response']['generation']

# load settings for judge model
judge_prompts_path = settings['judge']['prompt_filepath']
judge_seeds = settings['judge']['seeds']
judge_generation = settings['judge']['generation']

# load models from config file
manager = ModelManager(config_models_path)

# get all models used for response generation
models = manager.all_models()

# -----------------------
# # Example use for single turn:
prompt_dir = "data/prompts/bias"
output_dir = "results"
run_single_turn_response_generation(prompt_dir, output_dir, models, response_generation, response_seeds)

# # -----------------------
# # Example use for multiturn
prompt_dir = "data/prompts/multiturn"
output_dir = "results"
run_multi_turn_response_generation(prompt_dir, output_dir, models, response_generation, response_seeds)

# -----------------------
# Judge evaluation
judge_model = manager.get_judge()
llm_judge = LLMJudge(judge_model, judge_prompts_path)

prompt_dir = "results/llava"
eval_dir = "results/llava"
results_judge = llm_judge.judge_folder(prompt_dir, eval_dir, judge_generation, judge_seeds, save_results=True)


