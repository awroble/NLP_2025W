import json
from pathlib import Path
from typing import List, Dict

class PromptLoader:

    def __init__(self, directory: str):
        self.directory = Path(directory)
        if not self.directory.exists() or not self.directory.is_dir():
            raise ValueError(f"Directory not found: {directory}")

    def load_prompts(self) -> List[List[Dict]]:
        conversations = []

        #iterate over json files in the directory
        for json_file in self.directory.rglob("*.json"):
            with open(json_file, 'r', encoding='utf-8') as f:
                try:
                    data = json.load(f)
                except json.JSONDecodeError as e:
                    print(f"Error decoding JSON from file {json_file}: {e}")

                conversations.append(data)
                # if isinstance(data, list):
                #     conversations.append(data)
                # elif isinstance(data, dict):
                #     conversations.append(data)

        return conversations
