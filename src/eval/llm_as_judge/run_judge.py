import os
import json
import yaml
import argparse
import httpx
from tqdm import tqdm
import concurrent.futures
from openai import OpenAI

class Judge:
    def __init__(self, base_url, api_key, model_name, max_retries, max_parallel):
        self.client = OpenAI(
            base_url=base_url,
            api_key=api_key
        )
        # using chatgpt to judge
        # self.client = OpenAI(
        #     base_url=self.base_url,
        #     api_key=self.api_key,
        #     http_client=httpx.Client(
        #         base_url=self.base_url,
        #         follow_redirects=True,
        #     ),
        # )
        self.model = model_name
        self.max_retries = max_retries
        self.max_workers = max_parallel

    def evaluate(self, prompt):
        for _ in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0
                )
                return response.choices[0].message.content
            except Exception as e:
                print(f"API Error: {str(e)}")
        return "Evaluation Fail"

def process_turn(evaluator, model, version, turn):
    #input(prompts made by make_prompt.py) and output path
    input_dir = f"eval/prompt/{model}/{version}/turn{turn}"
    output_dir = f"eval/results/{model}/{version}/turn{turn}"
    os.makedirs(output_dir, exist_ok=True)
    
    input_file = os.path.join(input_dir, "judge_prompt.jsonl")
    output_file = os.path.join(output_dir, "judge_results.jsonl")
    
    processed = set()
    if os.path.exists(output_file):
        with open(output_file, 'r', encoding="utf-8") as f:
            for line in f:
                try:
                    processed.add(json.loads(line)['id'])
                except:
                    continue
    
    with open(input_file, 'r', encoding="utf-8") as f:
        tasks = [json.loads(line) for line in f if json.loads(line)['id'] not in processed]
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=evaluator.max_workers) as executor:
        task_ids = [task['id'] for task in tasks]
        task_prompts = [task['prompt'] for task in tasks]
        results = []
        try:
            results = list(tqdm(
                executor.map(evaluator.evaluate, task_prompts),
                total=len(task_prompts),
                desc=f"{model}-{version} Turn{turn}"
            ))
        except Exception as e:
            print(f"Processing Error: {str(e)}")

        with open(output_file, 'a', encoding="utf-8") as f:
            for task_id, response in zip(task_ids, results):
                result = {
                    "id": task_id,
                    "response": response
                }
                f.write(json.dumps(result, ensure_ascii=False) + '\n')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='eval')
    parser.add_argument('--model', help='Model name(related to input_dir/output_dir)')
    parser.add_argument('--version', help='Generated response version(related to input_dir/output_dir)')
    parser.add_argument('--base_url', required=True, help='API Base URL')
    parser.add_argument('--api_key', required=True, help='API Key')
    parser.add_argument('--llm', required=True, help='Model used to judge')
    parser.add_argument('--max_retries', type=int, help='Maximum number of retries')
    parser.add_argument('--max_parallel', type=int, help='Maximum parallelism')
    args = parser.parse_args()
    
    evaluator = Judge(
        base_url=args.base_url,
        api_key=args.api_key,
        model_name=args.llm,
        max_retries=args.max_retries,
        max_parallel=args.max_parallel
    )
    
    for turn in range(1, 6):
        process_turn(evaluator, args.model, args.version, turn)