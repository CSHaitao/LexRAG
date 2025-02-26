import os
import json
import httpx
from tqdm import tqdm
import concurrent.futures
from openai import OpenAI
from zhipuai import ZhipuAI

class Judge:
    def __init__(self, config):
        self.config = config
        if self.config.get("model_type") == "openai":
            self.client = OpenAI(
                base_url=self.config.get("api_base"),
                api_key=self.config.get("api_key"),
                http_client=httpx.Client(
                    base_url=self.config.get("api_base"),
                    follow_redirects=True,
                ),
            )
        elif self.config.get("model_type") == "zhipu":
            self.client = ZhipuAI(api_key=self.config["api_key"])
        elif self.config.get("model_type") in ["qwen","llama"]:
            self.client = OpenAI(
                base_url=self.config.get("api_base"), 
                api_key=self.config.get("api_key")
            )
        self.model = self.config.get("model_name")
        self.max_retries = self.config.get("max_retries")
        self.max_workers = self.config.get("max_parallel")

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

def process_turn(config, turn):
    evaluator = Judge(config)
    #input(prompts made by make_prompt.py) and output path
    input_dir = f"data/prompt/turn{turn}"
    output_dir = f"data/results/turn{turn}"
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
                desc=f"{evaluator.model} Turn{turn}"
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
