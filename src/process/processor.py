import json
from pathlib import Path
from tqdm import tqdm

class QuestionGenerator:
    def __init__(self, process_type: str):
        self.process_type = process_type
        self.process_methods = {
            "current_question": self._current_question,
            "prefix_question": self._prefix_question,
            "prefix_question_answer": self._prefix_question_answer,
            "suffix_question": self._suffix_question
        }

    def run_process(self, original_data_path: str, output_path: str):
        method = self.process_methods.get(self.process_type)
        if not method:
            raise ValueError(f"Unsupported process type: {self.process_type}")

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(original_data_path, "r", encoding="utf-8") as f:
            data_list = json.load(f)

        method(data_list, output_path)

    def _current_question(self, data_list, file_path):
        for data in data_list:
            for conversation in data["conversation"]:
                conversation["question"] = {
                    "type": "current_question",
                    "content": conversation["user"],
                }
        self._save_output(data_list, file_path)

    def _prefix_question(self, data_list, file_path):
        for data in data_list:
            prefix_question = ""
            for conversation in data["conversation"]:
                prefix_question += conversation["user"]
                conversation["question"] = {
                    "type": "prefix_question",
                    "content": prefix_question,
                }
        self._save_output(data_list, file_path)

    def _prefix_question_answer(self, data_list, file_path):
        for data in data_list:
            prefix_question_answer = ""
            for conversation in data["conversation"]:
                prefix_question_answer += f" {conversation['user']}\n\n"
                conversation["question"] = {
                    "type": "prefix_question_answer",
                    "content": prefix_question_answer,
                }
                prefix_question_answer += f"{conversation['assistant']}\n\n"
        self._save_output(data_list, file_path)

    def _suffix_question(self, data_list, file_path):
        for data in data_list:
            suffix_question = ""
            for idx in range(len(data["conversation"]) - 1, -1, -1):
                suffix_question += f"{data['conversation'][idx]['user']}\n\n"
                data["conversation"][idx]["question"] = {
                    "type": "suffix_question",
                    "content": suffix_question,
                }
        self._save_output(data_list, file_path)

    def _save_output(self, data_list, file_path):
        with open(file_path, "w", encoding="utf-8") as f:
            for data in data_list:
                f.write(json.dumps(data, ensure_ascii=False) + "\n")
