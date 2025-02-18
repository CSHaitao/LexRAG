import os
import logging
import json
import json
import time
from typing import List
from tqdm import tqdm
from zhipuai import ZhipuAI
from openai import OpenAI
import httpx
from concurrent.futures import ThreadPoolExecutor, as_completed

class BaseGenerator:
    """Base class for all generators"""
    def __init__(self, config, max_retries, max_parallel, top_n, batch_size=20):
        self.config = config
        self.max_retries = max_retries
        self.max_parallel = max_parallel
        self.batch_size = batch_size
        self.top_n = top_n
        self.failed_ids = set()
        self.logger = logging.getLogger(self.__class__.__name__)
        
    def _save_results(self, result_dict):
        with open("data/generated_responses.jsonl", "a", encoding="utf-8") as f:
            for item_id in sorted(result_dict.keys(), key=lambda x: int(x.split("_")[0])):  
                f.write(json.dumps(result_dict[item_id], ensure_ascii=False) + "\n")
            
class OpenAIGenerator(BaseGenerator):
    """Generator for OpenAI compatible models"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = self.config.get("model_name", "gpt-3.5-turbo")
        if self.model and self.model.startswith("gpt"):
            self.client = OpenAI(
                base_url=self.config["api_base"],
                api_key=self.config["api_key"],
                http_client=httpx.Client(
                    base_url=self.config["api_base"],
                    follow_redirects=True,
                ),
            )
        else:
            self.client = OpenAI(
                base_url=self.config["api_base"], 
                api_key=self.config["key"]
            )

    def generate(self, processed_data, retrieval_data):
        for turn_num, samples in processed_data.items():
            messages_list, id_list, questions_list, articles_list = self._prepare_inputs(samples, retrieval_data)
            self._batch_call(messages_list, id_list, questions_list, articles_list)

    def _prepare_inputs(self, data, retrieval_data):
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
            articles_list.append(self._get_top_articles(sample["id"], retrieval_data, self.top_n))
        
        return messages_list, id_list, questions_list, articles_list
    

    def _get_top_articles(self, sample_id, retrieval_data, top_n):
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

    def _batch_call(self, messages_list, id_list, questions_list, articles_list):
        result_dict = {}
        with ThreadPoolExecutor(max_workers=self.max_parallel) as executor:
            futures = []
            
            with tqdm(total=len(messages_list), desc="Generating responses") as pbar:
                for messages, item_id, question, articles in zip(messages_list, id_list, questions_list, articles_list):
                    future = executor.submit(self._call_api, messages, item_id, question, articles)
                    future.add_done_callback(lambda _: pbar.update(1))
                    futures.append(future)
                    
                    if len(futures) >= self.batch_size:
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

class ZhipuGenerator(BaseGenerator):
    """Generator for ZhipuAI GLM models"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.client = ZhipuAI(api_key=self.config["api_key"])
        self.model = self.config.get("model_name", "glm-4-flash")

    def generate(self, processed_data, retrieval_data):
        for turn_num, samples in processed_data.items():
            messages_list, id_list, questions_list, articles_list = self._prepare_inputs(samples, retrieval_data)
            self._batch_call(messages_list, id_list, questions_list, articles_list)

    def _prepare_inputs(self, data, retrieval_data):
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
            articles_list.append(self._get_top_articles(sample["id"], retrieval_data, self.top_n))
        
        return messages_list, id_list, questions_list, articles_list
    

    def _get_top_articles(self, sample_id, retrieval_data, top_n):
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

    def _batch_call(self, messages_list, id_list, questions_list, articles_list):
        result_dict = {}
        with ThreadPoolExecutor(max_workers=self.max_parallel) as executor:
            futures = []
            
            with tqdm(total=len(messages_list), desc="Generating responses") as pbar:
                for messages, item_id, question, articles in zip(messages_list, id_list, questions_list, articles_list):
                    future = executor.submit(self._call_api, messages, item_id, question, articles)
                    future.add_done_callback(lambda _: pbar.update(1))
                    futures.append(future)
                    
                    if len(futures) >= self.batch_size:
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
    
class VLLMGenerator(BaseGenerator):
    """Generator for vLLM models"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        from vllm import LLM, SamplingParams
        self.llm = LLM(
            model=self.config["model_path"],
            tensor_parallel_size=self.config["gpu_num"],
            gpu_memory_utilization=0.85
        )
        self.sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=2048
        )

    def generate(self, processed_data, retrieval_data):
        for turn_num, samples in processed_data.items():
            prompts = self._prepare_prompts(samples, retrieval_data)
            responses = self.llm.generate(prompts, self.sampling_params)
            self._save_vllm_results(responses, samples)

    def _prepare_prompts(self, data, retrieval_data):
        prompts = []
        for sample in data:
            system_base = "你是一位精通法律知识的专家，致力于为用户提供准确、专业的法律咨询。你的回复应确保严谨、高效，并在风格上与前几轮的回答保持一致（如有）。若用户的问题涉及具体法律条文，应尽可能引用相关法条，以增强回答的权威性。同时，避免提供无关信息，确保回复简明、直接且切中要害。"
            articles = self._get_top_articles(sample["id"], retrieval_data, self.top_n)
            
            if articles:
                system_base += "\n\n以下是你可以参考的法条：\n" + "\n".join(articles)
            
            history = "\n".join(
                f"用户：{h['user']}\n助理：{h['assistant']}" 
                for h in sample["history"]
            )
            prompt = f"{system_base}\n\n{history}\n当前问题：{sample['current_question']}\n回答："
            prompts.append(prompt)
        return prompts

    def _get_top_articles(self, sample_id, retrieval_data, top_n):
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

    def _save_vllm_results(self, outputs, samples):
        results = {}
        for output, sample in zip(outputs, samples):
            results[sample["id"]] = {
                "id": sample["id"],
                "question": sample["current_question"],
                "response": output.outputs[0].text
            }
        self._save_results(results)
