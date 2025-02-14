import os
import logging
import json
import argparse
from data_processor import DataProcessor
from client import Client

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def load_retrieval_results(data_path):
    llm_data = {}
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            entry = json.loads(line)
            llm_data[entry["id"]] = entry
    return llm_data

def get_top_articles(sample_id, retrieval_data, top_n):
    try:
        parts = sample_id.split("_")
        if len(parts) != 2 or not parts[1].startswith("turn"):
            raise ValueError(f"Invalid sample_id format: {sample_id}")
        
        dialogue_id = int(parts[0])
        turn_number = int(parts[1][4:])  
        turn_index = turn_number - 1   
        dialogue = retrieval_data.get(dialogue_id, {}).get("conversation", [])
        if not dialogue or turn_index >= len(dialogue):
            return []
    
        recall_list = dialogue[turn_index]["question"]["recall"]
        sorted_recall = sorted(recall_list, 
                             key=lambda x: x["score"], 
                             reverse=True)[:top_n]
        
        return [item["article"]["name"] for item in sorted_recall]
        
    except Exception as e:
        logging.error(f"Error processing sample {sample_id}: {str(e)}")
        return []

def run_evaluation():
    # [1/2] process data
    logging.info("[1/2] Processing raw data...")
    processor = DataProcessor()
    processed_data = processor.process_conversation_turns(args.raw_data_path)

    # output
    output_dir = "data/generated_samples"
    os.makedirs(output_dir, exist_ok=True)

    for turn_num, samples in processed_data.items():
        output_path = f"{output_dir}/{turn_num}.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(samples, f, indent=2, ensure_ascii=False)
    
    logging.info("Loading Retrieval data...")
    retrieval_data = load_retrieval_results(args.data_path)

    # [2/2] model client
    logging.info("[2/2] Initializing model client...")
    client = Client(
        api_key=args.api_key,
        base_url=args.base_url,
        model_name=args.model_name,
        max_retries=args.max_retries,
        max_parallel=args.max_parallel
    )

    for turn_num in processed_data.keys():
        input_path = f"{output_dir}/{turn_num}.json"
        with open(input_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        messages_list = []
        id_list = []
        questions_list = []
        articles_list = []
        for sample in data:
            messages = []
            for h in sample["history"]:
                messages.extend([
                    {"role": "user", "content": h["user"]},
                    {"role": "assistant", "content": h["assistant"]}
                ])
            messages.append({"role": "user", "content": sample["current_question"]})
            
            messages_list.append(messages)
            id_list.append(sample["id"])
            questions_list.append(sample["current_question"])
            articles = get_top_articles(sample["id"], retrieval_data, args.top_n)
            articles_list.append(articles)
        
        client.batch_call(messages_list, id_list, questions_list, articles_list)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_data_path", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--api_key", type=str, required=True)
    parser.add_argument("--base_url", type=str)
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--max_retries", type=int, required=True)
    parser.add_argument("--max_parallel", type=int, required=True)
    parser.add_argument("--top_n", type=int, required=True)

    args = parser.parse_args()
    run_evaluation(args)