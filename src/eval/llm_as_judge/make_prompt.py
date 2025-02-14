import json
import os
import argparse
from use_template import use_judge_template

def process_model(data_path, model_name, version):
    with open(data_path, 'r', encoding='utf-8') as f:
        original_data = json.load(f)
    
    #Path to the object for which the prompt needs to be generated
    gen_path = f"generate_data/{model_name}/generated_responses({model_name}_{version}).jsonl"
    with open(gen_path, 'r', encoding='utf-8') as f:
        generated_data = [json.loads(line) for line in f]
    
    for turn in range(5):
        #Path to the output prompts
        output_dir = f"eval/prompt/{model_name}/{version}/turn{turn+1}"
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "judge_prompt.jsonl")
        
        with open(output_path, 'w', encoding='utf-8') as out_file:
            for case in original_data:
                conv = case['conversation']
                if len(conv) <= turn:
                    continue
                
                gen_id = f"{case['id']}_turn{turn+1}"
                generated = next((g for g in generated_data if g['id'] == gen_id), None)
                if not generated:
                    continue
                
                prompt = use_judge_template(
                    conversation=conv,
                    reference_answer=conv[turn]['assistant'],
                    generated_answer=generated['response'],
                    current_turn=turn
                )
                
                out_file.write(json.dumps({
                    "id": gen_id,
                    "prompt": prompt
                }, ensure_ascii=False) + '\n')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate prompts for llm_as_judge")
    parser.add_argument("--data_path", help="Path to the original data JSON file")
    parser.add_argument("--model", help="Model name(related to gen_path)")
    parser.add_argument("--version", help="Generated response version(Related to gen_path)")
    args = parser.parse_args()
    process_model(args.data_path, args.model, args.version)