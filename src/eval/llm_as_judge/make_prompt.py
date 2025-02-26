import json
import os
from eval.llm_as_judge.use_template import use_judge_template

#data_path: original data path
#gen_path: (llm model)generated response path
def process_model(data_path, gen_path):
    with open(data_path, 'r', encoding='utf-8') as f:
        original_data = json.load(f)
    
    with open(gen_path, 'r', encoding='utf-8') as f:
        generated_data = [json.loads(line) for line in f]
    
    for turn in range(5):
        #Path to the output prompts
        output_dir = f"data/prompt/turn{turn+1}"
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
