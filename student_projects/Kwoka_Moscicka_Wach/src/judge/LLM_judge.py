from src.utils import set_seed
import json
from pathlib import Path
import re
from collections import Counter

class LLMJudge:
    """
    Class to classify model's response as safe, unsafe or refusal.
    """
    def __init__(self, hf_model, judge_prompt_file):
        """
        :param hf_model: HFModel instance of a judge model.
        :param judge_prompt_file: Filepath to judge prompt file.
        """
        self.model = hf_model
        path = Path(judge_prompt_file)
        if not path.exists():
            raise FileNotFoundError(f"no {judge_prompt_file}")
        with open(path, "r", encoding="utf-8") as f:
            self.prefix_prompts = json.load(f)

    def judge_model(self, conversation, generation_params, seeds, prefix=None):
        """
        Judge model's response.
        :param conversation: Prompt and model's response passed to the judge model.
        :param generation_params: Model generation parameters.
        :param seeds Seeds for reproducibility.
        :param prefix: Prefix from id to detect whether conversation is single-turn or multi-turn.
        :return: Integer score assigned by judge and judge response.
        """
        if not conversation:
            return -1, "no conversation"
        
        if prefix and prefix in self.prefix_prompts:
            base_prompt = self.prefix_prompts[prefix]
        else:
            return -1, "no prefix" 

        full_prompt = f"""{base_prompt}
        Conversation: 
        {conversation}
        """

        scores = []
        outputs = []

        for seed in seeds:
            set_seed(seed)
            prompt_dict = {"data_type": "text", "prompt": full_prompt}
            try:
                output = self.model.generate(prompt_dict, **generation_params)
                output_text = output[0].strip() if isinstance(output, list) else str(output).strip()
                outputs.append(output_text)
                first = output_text.lstrip()[0] if output_text.lstrip() else None

                if first in {"0", "1", "2"}:
                    scores.append(int(first))
                else:
                    if "0" in output_text and "1" not in output_text and "2" not in output_text:
                        scores.append(0)
                    elif "1" in output_text and "0" not in output_text and "2" not in output_text:
                        scores.append(1)
                    elif "2" in output_text and "0" not in output_text and "1" not in output_text:
                        scores.append(2)
                    else:
                        scores.append(-1)
            except Exception as e:
                print(f"Error in judge model: {e}")
                scores.append(-1)
                outputs.append(f"ERROR: {str(e)}")

        # most frequent score
        most_common_score = Counter(scores).most_common(1)[0][0]
        return most_common_score, outputs

    def judge_folder(self, folder_path, eval_folder, generation_params, seeds, save_results=True):
        """
        Method to judge all conversations in directory specified by `folder_path`.
        :param folder_path: Path to directory containing conversations.
        :param eval_folder: Path to output directory containing conversations with added judge score.
        :param generation_params: Generation parameters for judge model.
        :param seeds Seeds for reproducibility.
        :param save_results: Whether to save results with judge score.
        :return: Scored results.
        """
        folder = Path(folder_path)
        eval_folder = folder / eval_folder
        eval_folder.mkdir(exist_ok=True)

        json_files = [f for f in folder.glob("*.json")]
        results = []

        for f in json_files:
            try:
                with open(f, "r", encoding="utf-8") as file:
                    data = json.load(file)
            except json.JSONDecodeError:
                print(f"Error reading JSON {f}")
                continue

            pattern = re.compile(r'^[A-Z]+(?:-MT)?')

            if isinstance(data, list) and data:
                first_id = data[0].get("prompt_id", "")
                prefix = pattern.search(first_id).group()

                conversation = self.transform_conv(f, prefix)
                score, output_text = self.judge_model(conversation, generation_params, seeds, prefix=prefix)

            else:
                conversation = ""
                prompt_text = data.get("prompt", "")
                output_text = data.get("output", "")
                conversation += f"USER: {prompt_text}\n"
                conversation += f"ASSISTANT: {output_text}\n"
                conversation=conversation.rstrip()
                prefix = data.get("prompt_id", "")[:2]
                score, output_text = self.judge_model(conversation, generation_params, seeds, prefix=prefix)

            if isinstance(data, list):
                data = {
                    "conversation": data,
                    "judge_output": output_text,
                    "judge_score": score
                }
            else:
                data["judge_output"] = output_text
                data["judge_score"] = score

            results.append(data)

            if save_results:
                save_path = eval_folder / f"{f.name}"
                with open(save_path, "w", encoding="utf-8") as out_file:
                    json.dump(data, out_file, indent=2, ensure_ascii=False)

        return results

    @staticmethod
    def transform_conv(file, prompt_prefix):
        """
        Method to transform conversation to pass to judge
        :param file: File with prompt and model output.
        :param prompt_prefix: Prefix of prompts ('BF', 'MT', 'JB', 'BF-MT', 'JB-MT').
        :return: Transformed conversation.
        """
        path = Path(file)
        if not path.exists():
            raise FileNotFoundError(f"{file} does not exist")

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        conversation_text = ""

        for entry in data:
            prompt_id = entry.get("prompt_id", "")
            prompt = entry.get("prompt", "")
            output_text = entry.get("output", "")

            if prompt_id.lower().startswith(prompt_prefix.lower()) and output_text:
                conversation_text += f"USER: {prompt}\n"
                conversation_text += f"ASSISTANT: {output_text}\n"

        return conversation_text.rstrip() 
