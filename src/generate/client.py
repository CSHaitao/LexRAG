import time
from zhipuai import ZhipuAI
from openai import OpenAI
import httpx
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from tqdm import tqdm

class Client:
    def __init__(self, api_key, base_url, model_name, max_retries, max_parallel):
        #self.client = ZhipuAI(api_key=api_key)
        self.client = OpenAI(
            base_url=base_url, 
            api_key=api_key,
            http_client=httpx.Client(
                base_url=base_url,
                follow_redirects=True,
            ),
        )
        self.model = model_name
        self.max_retries = max_retries
        self.max_parallel = max_parallel
        self.failed_ids = set()
        
    def _call_api(self, messages, item_id, question, articles):
        """重试"""
        system_base = "你是一位精通法律知识的专家，致力于为用户提供准确、专业的法律咨询。你的回复应确保严谨、高效，并在风格上与前几轮的回答保持一致（如有）。若用户的问题涉及具体法律条文，应尽可能引用相关法条，以增强回答的权威性。同时，避免提供无关信息，确保回复简明、直接且切中要害。"
        
        if articles:
            system_base += "\n\n以下是你可以参考的法条：\n"
            system_base += "\n".join([f"{i+1}. {art}" for i, art in enumerate(articles)])
        
        system_prompt = [{
            "role": "system",
            "content": system_base
        }]

        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=system_prompt + messages,
                    temperature=0.0
                )
                return {
                    "id": item_id,
                    "question": question,
                    "response": response.choices[0].message.content
                }
            except Exception as e:
                print(f"API Error (Attempt {attempt+1}): {str(e)}")
                time.sleep(2 ** attempt)
        
        self.failed_ids.add(item_id)
        return {"id": item_id, "question": question, "response": ""}

    def batch_call(self, messages_list, id_list, questions_list, articles_list, batch_size=20):
        result_dict = {}
        with ThreadPoolExecutor(max_workers=self.max_parallel) as executor:
            futures = []
            
            with tqdm(total=len(messages_list), desc="Generating responses") as pbar:
                for messages, item_id, question, articles in zip(messages_list, id_list, questions_list, articles_list):
                    future = executor.submit(self._call_api, messages, item_id, question, articles)
                    future.add_done_callback(lambda _: pbar.update(1))
                    futures.append(future)
                    
                    if len(futures) >= batch_size:
                        current_batch = []
                        for future in as_completed(futures):
                            result = future.result()
                            result_dict[result["id"]] = result 
                            current_batch.append(result)
                        self._save_results({r["id"]: r for r in current_batch})
                        futures.clear()  
            
                if futures:
                    current_batch = []
                    for future in as_completed(futures):
                        result = future.result()
                        result_dict[result["id"]] = result
                        current_batch.append(result)
                self._save_results({r["id"]: r for r in current_batch})
        
        return [result_dict[id] for id in sorted(id_list, key=lambda x: int(x.split("_")[0]))]

    def _save_results(self, result_dict):
        with open("data/generated_responses.jsonl", "a", encoding="utf-8") as f:
            for item_id in sorted(result_dict.keys(), key=lambda x: int(x.split("_")[0])):  
                f.write(json.dumps(result_dict[item_id], ensure_ascii=False) + "\n")