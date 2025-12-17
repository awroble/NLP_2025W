from pathlib import Path
import yaml
from src.loaders.model_loader import HFModel

class ModelManager:
    def __init__(self, yaml_path: str):
        """
        Load all models defined in a YAML file.
        yaml_path: path to models.yaml
        """
        self.yaml_path = Path(yaml_path)
        if not self.yaml_path.exists():
            raise FileNotFoundError(f"File not found: {yaml_path}")

        with open(self.yaml_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        self.models = {}
        for entry in config.get("models", []):
            name = entry["name"]
            checkpoint = entry["checkpoint"]
            print(f"Loading model: {name} ({checkpoint})")
            self.models[name] = HFModel(checkpoint=checkpoint, model_type=entry["type"])

    def get_model(self, name: str):
        """
        Return a loaded HFModel by its name.
        """
        return self.models.get(name)

    def all_models(self):
        """
        Return all loaded models as a dict.
        """
        return self.models