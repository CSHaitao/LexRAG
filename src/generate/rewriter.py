import json
import time
import httpx
from tqdm import tqdm
from pathlib import Path
import argparse
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed
from zhipuai import ZhipuAI

class Rewriter:
    def __init__(self, model, base_url, api_key, max_retries, rpm_limit, max_parallel, original_data, output_path, batch_size):
        self.model = model
        self.base_url = base_url
        self.api_key = api_key
        self.max_retries = max_retries
        self.rpm_limit = rpm_limit
        self.max_parallel = max_parallel
        self.original_data = original_data
        self.output_path = output_path
        self.batch_size = batch_size
        
        self.client = OpenAI(
            base_url=self.base_url,
            api_key=self.api_key,
            http_client=httpx.Client(
                base_url=self.base_url,
                follow_redirects=True,
            ),
        )
        #self.client = ZhipuAI(api_key="your_zhipuai_api_key")
        self.results = {}
        self.failed_ids = set()

    def generate_prompt(self, history, question):
        if not history.strip() or history == "无历史对话":
            return question  
        return f"""给定以下对话(包括历史对话和当前问题)，请将用户的当前问题改写为一个独立的问题，使其无需依赖对话历史即可理解用户的意图。
在改写用户的当前问题时，请避免不必要的措辞修改或引入对话中未提及的新术语或概念。改写应尽可能接近用户当前问题的结构和含义。

历史对话：
{history}

当前问题：{question}

请输出改写结果："""

    def process_single(self, data_id, conv_idx, history, question):
        if not history.strip() or history == "无历史对话":
            return (data_id, conv_idx, question)
        unique_id = f"{data_id}_{conv_idx}"
        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{
                        "role": "user",
                        "content": self.generate_prompt(history, question)
                    }],
                    temperature=0.0,
                    stream=False
                )
                
                reworded = response.choices[0].message.content.strip()
                
                reworded = reworded.replace('"', '').replace("```", "").strip()
                return (data_id, conv_idx, reworded)
                
            except Exception as e:
                print(f"[{unique_id}] Attempt {attempt+1} failed: {str(e)}")
                time.sleep(2 ** attempt)
        
        self.failed_ids.add(unique_id)
        return (data_id, conv_idx, question)  

    def batch_process(self):
        with open(self.original_data, "r", encoding="utf-8") as f:
            original_data = json.load(f)
        
        tasks = []
        for data in original_data:
            data_id = data["id"]
            for conv_idx, conv in enumerate(data["conversation"]):
                history = "\n".join(
                    [f"用户：{c['user']}\n助理：{c['assistant']}" 
                     for c in data["conversation"][:conv_idx]]
                )
                if not history.strip():
                    history = "无历史对话"
                tasks.append((
                    data_id,
                    conv_idx,
                    history,
                    conv["user"]
                ))
        
        completed_count = 0

        with ThreadPoolExecutor(max_workers=self.max_parallel) as executor:
            futures = []
            
            for task in tasks:
                future = executor.submit(self.process_single, *task)
                futures.append(future)
            
            with tqdm(total=len(tasks), desc="Processing") as pbar:
                for future in as_completed(futures):
                    data_id, conv_idx, reworded = future.result()
                    unique_id = f"{data_id}_{conv_idx}"
                    self.results[unique_id] = reworded
                    completed_count += 1
                    pbar.update(1)
                    if completed_count % self.batch_size == 0:
                        self.generate_output_file(original_data, self.results)
        
        self.generate_output_file(original_data, self.results)
        print(f"Process completed. Failed: {len(self.failed_ids)}")

    def generate_output_file(self, original_data, results):
        output_lines = []
        for data in original_data:
            data_id = data["id"]
            for conv_idx, conv in enumerate(data["conversation"]):
                unique_id = f"{data_id}_{conv_idx}"
                conv["question"] = {
                    "type": "llm_question",
                    "content": results.get(unique_id, conv["user"])  
                }
            output_lines.append(json.dumps(data, ensure_ascii=False))
        
        with open(self.output_path, "w", encoding="utf-8") as f:
            f.write("\n".join(output_lines) + "\n")

def main():
    parser = argparse.ArgumentParser(description="using llm to rewrite query")
    
    parser.add_argument("--model", type=str, required=True, help="The model to use (e.g., gpt-4o-mini)")
    parser.add_argument("--base_url", type=str, required=True, help="base_url for API")
    parser.add_argument("--api_key", type=str, required=True, help="api_key")
    parser.add_argument("--max_retries", type=int, default=10, help="Maximum retries for each request")
    parser.add_argument("--rpm_limit", type=int, default=3000, help="RPM limit for API requests")
    parser.add_argument("--max_parallel", type=int, default=32, help="Maximum parallel processes")
    parser.add_argument("--original_data", type=str, required=True, help="Path to the original data file")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save the output file")
    parser.add_argument("--batch_size", type=int, default=20, help="Batch size for processing")
    
    args = parser.parse_args()

    rewriter = Rewriter(
        model=args.model,
        base_url=args.base_url,
        api_key=args.api_key,
        max_retries=args.max_retries,
        rpm_limit=args.rpm_limit,
        max_parallel=args.max_parallel,
        original_data=args.original_data,
        output_path=args.output_path,
        batch_size=args.batch_size
    )
    
    rewriter.batch_process()

if __name__ == "__main__":
    main()