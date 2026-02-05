import json
from pathlib import Path

class PromptLoader:
    """
    Class to load prompts from files.
    """
    def __init__(self, directory: str):
        """
        :param directory: Directory to load prompts from.
        """
        self.directory = Path(directory)
        if not self.directory.exists() or not self.directory.is_dir():
            raise ValueError(f"Directory not found: {directory}")

    def load_prompts(self):
        """
        Method to load prompts from files.
        :return: List of prompts.
        """
        conversations = []

        #iterate over json files in the directory
        for json_file in self.directory.rglob("*.json"):
            with open(json_file, 'r', encoding='utf-8') as f:
                try:
                    data = json.load(f)
                except json.JSONDecodeError as e:
                    print(f"Error decoding JSON from file {json_file}: {e}")

                conversations.append(data)

        return conversations
