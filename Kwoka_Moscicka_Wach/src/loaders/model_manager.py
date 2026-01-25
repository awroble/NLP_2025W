from pathlib import Path
import yaml
from src.loaders.model_loader import HFModel

class ModelManager:
    """
    Loads models defined in YAML file.
    """
    def __init__(self, yaml_path, multi_turn=False):
        """
        yaml_path: path to model configuration file
        """
        self.yaml_path = Path(yaml_path)
        if not self.yaml_path.exists():
            raise FileNotFoundError(f"File not found: {yaml_path}")

        with open(self.yaml_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        max_history = config["max_history"]
        if multi_turn:
            max_history_response = max_history["multi_turn"]
        else:
            max_history_response = max_history["single_turn"]

        self.models = {}
        for entry in config.get("models", []):
            # get data from configuration file
            name = entry["name"]
            checkpoint = entry["checkpoint"]
            model_type = entry.get("type", None)
            # get data format for multimodal models
            multimodal_format = entry.get('multimodal', {}).get('format', None)
            image_token = entry.get('multimodal', {}).get('image_token', None)

            print(f"Loading model: {name} ({checkpoint})")
            self.models[name] = HFModel(checkpoint=checkpoint, model_type=model_type, max_history=max_history_response, multimodal_format=multimodal_format, image_token=image_token)

        judge = config.get('judge', None)
        if judge is not None:
            # load judge model (always without storing history)
            self.judge = HFModel(checkpoint=judge['checkpoint'], model_type=judge['type'], max_history=1)

    def get_model(self, name: str):
        """
        Return a loaded HFModel by its name.
        """
        return self.models.get(name)

    def all_models(self):
        """
        Return all loaded models used for response generation.
        """
        return self.models

    def get_judge(self):
        """
        Return judge model.
        """
        return self.judge
