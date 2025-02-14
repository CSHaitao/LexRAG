# Data preprocessing of dataset(output generated_samples for each turn)
import json

class DataProcessor:
    @staticmethod
    def process_conversation_turns(raw_data_path):
        with open(raw_data_path, encoding="utf-8") as f:
            data = json.load(f)
        
        processed = {}
        for turn in range(1, 6):
            processed[f"turn_{turn}"] = []
        
        for item in data:
            conv = item["conversation"]
            for turn_num in range(1, 6):
                if turn_num > len(conv):
                    continue
                
                current_turn = conv[turn_num-1]
                
                clean_history = [
                    {
                        "user": h["user"],
                        "assistant": h["assistant"]
                    }
                    for h in conv[:turn_num-1]
                ]

                entry = {
                    "id": f"{item['id']}_turn{turn_num}",
                    "history": clean_history,
                    "current_question": current_turn["user"],
                    "reference": current_turn["assistant"],
                    "keywords": current_turn.get("keyword", []),
                    "article": current_turn.get("article", []),
                    "article_context": current_turn.get("article_context", [])
                }
                processed[f"turn_{turn_num}"].append(entry)
        
        return processed