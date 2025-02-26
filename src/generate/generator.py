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
from transformers import AutoModelForCausalLM, AutoTokenizer
from generate.prompt_builder import LegalPromptBuilder, CustomSystemPromptBuilder, FullCustomPromptBuilder

class BaseGenerator:
    """Base class for all generators"""
    def __init__(self, config, max_retries, max_parallel, top_n, batch_size = 20):
        self.config = config
        self.max_retries = max_retries
        self.max_parallel = max_parallel
        self.top_n = top_n
        self.batch_size = batch_size
        self.failed_ids = set()
        self.logger = logging.getLogger(self.__class__.__name__)
        
    def _save_results(self, result_dict):
        with open("data/generated_responses.jsonl", "a", encoding="utf-8") as f:
            for item_id in sorted(result_dict.keys(), key=lambda x: int(x.split("_")[0])):  
                f.write(json.dumps(result_dict[item_id], ensure_ascii=False) + "\n")
            
class OpenAIGenerator(BaseGenerator):
    """Generator for OpenAI API models"""
    def __init__(self, prompt_builder, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.prompt_builder = prompt_builder
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
                api_key=self.config["api_key"]
            )

    def generate(self, processed_data, retrieval_data):
        for turn_num, samples in processed_data.items():
            messages_list, id_list, questions_list, articles_list, system_prompt = self._prepare_inputs(samples, retrieval_data)
            self._batch_call(messages_list, id_list, questions_list, articles_list, system_prompt)

    def _prepare_inputs(self, data, retrieval_data):
        messages_list = []
        id_list = []
        questions_list = []
        articles_list = []
        system_prompt = []
        for sample in data:
            articles = self._get_top_articles(sample["id"], retrieval_data, self.top_n)
            messages = self.prompt_builder.build_messages(
                sample["history"],
                sample["current_question"],
                articles
            )
            
            system_msg = next((msg["content"] for msg in messages if msg["role"] == "system"), "")

            messages_list.append([msg for msg in messages if msg["role"] != "system"])
            id_list.append(sample["id"])
            questions_list.append(sample["current_question"])
            articles_list.append(articles)
            system_prompt.append(system_msg)
            
        
        return messages_list, id_list, questions_list, articles_list, system_prompt
    

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
        
    def _call_api(self, messages, item_id, question, articles, system_prompt):
        full_messages = []
        if system_prompt:
            full_messages.append({"role": "system", "content": system_prompt})
        full_messages.extend(messages)

        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=full_messages,
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

    def _batch_call(self, messages_list, id_list, questions_list, articles_list, system_prompts):
        result_dict = {}
        with ThreadPoolExecutor(max_workers=self.max_parallel) as executor:
            futures = []
            
            with tqdm(total=len(messages_list), desc="Generating responses") as pbar:
                for messages, item_id, question, articles, system_prompt in zip(messages_list, id_list, questions_list, articles_list, system_prompts):
                    future = executor.submit(self._call_api, messages, item_id, question, articles, system_prompt)
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
    """Generator for ZhipuAI API models"""
    def __init__(self, prompt_builder, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.prompt_builder = prompt_builder
        self.client = ZhipuAI(api_key=self.config["api_key"])
        self.model = self.config.get("model_name", "glm-4-flash")

    def generate(self, processed_data, retrieval_data):
        for turn_num, samples in processed_data.items():
            messages_list, id_list, questions_list, articles_list, system_prompt = self._prepare_inputs(samples, retrieval_data)
            self._batch_call(messages_list, id_list, questions_list, articles_list, system_prompt)

    def _prepare_inputs(self, data, retrieval_data):
        messages_list = []
        id_list = []
        questions_list = []
        articles_list = []
        system_prompt = []
        for sample in data:
            articles = self._get_top_articles(sample["id"], retrieval_data, self.top_n)
            messages = self.prompt_builder.build_messages(
                sample["history"],
                sample["current_question"],
                articles
            )
            
            system_msg = next((msg["content"] for msg in messages if msg["role"] == "system"), "")

            messages_list.append([msg for msg in messages if msg["role"] != "system"])
            id_list.append(sample["id"])
            questions_list.append(sample["current_question"])
            articles_list.append(articles)
            system_prompt.append(system_msg)
            
        
        return messages_list, id_list, questions_list, articles_list, system_prompt
    

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
        
    def _call_api(self, messages, item_id, question, articles, system_prompt):
        full_messages = []
        if system_prompt:
            full_messages.append({"role": "system", "content": system_prompt})
        full_messages.extend(messages)

        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=full_messages,
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

    def _batch_call(self, messages_list, id_list, questions_list, articles_list, system_prompts):
        result_dict = {}
        with ThreadPoolExecutor(max_workers=self.max_parallel) as executor:
            futures = []
            
            with tqdm(total=len(messages_list), desc="Generating responses") as pbar:
                for messages, item_id, question, articles, system_prompt in zip(messages_list, id_list, questions_list, articles_list, system_prompts):
                    future = executor.submit(self._call_api, messages, item_id, question, articles, system_prompt)
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
    def __init__(self, prompt_builder, *args, **kwargs):
        self.prompt_builder = prompt_builder
        kwargs.pop('prompt_builder', None)
        super().__init__(*args, **kwargs)
        from vllm import LLM, SamplingParams
        self.llm = LLM(
            model=self.config["model_path"],
            tensor_parallel_size=self.config["gpu_num"],
            gpu_memory_utilization=0.85
        )
        self.sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=4096
        )

    def generate(self, processed_data, retrieval_data):
        for turn_num, samples in processed_data.items():
            messages_list, id_list, questions_list, articles_list, system_prompts = self._prepare_inputs(samples, retrieval_data)
            prompts = self._build_vllm_prompts(messages_list, system_prompts)
            responses = self.llm.generate(prompts, self.sampling_params)
            self._save_vllm_results(responses, id_list, questions_list)

    def _prepare_inputs(self, data, retrieval_data):
        messages_list = []
        id_list = []
        questions_list = []
        articles_list = []
        system_prompt = []
        for sample in data:
            articles = self._get_top_articles(sample["id"], retrieval_data, self.top_n)
            messages = self.prompt_builder.build_messages(
                sample["history"],
                sample["current_question"],
                articles
            )
            
            system_msg = next((msg["content"] for msg in messages if msg["role"] == "system"), "")

            messages_list.append([msg for msg in messages if msg["role"] != "system"])
            id_list.append(sample["id"])
            questions_list.append(sample["current_question"])
            articles_list.append(articles)
            system_prompt.append(system_msg)
            
        
        return messages_list, id_list, questions_list, articles_list, system_prompt

    def _build_vllm_prompts(self, messages_list, system_prompts):
        prompts = []
        for messages, system_prompt in zip(messages_list, system_prompts):
            full_dialog = []
            if system_prompt:
                full_dialog.append(f"System: {system_prompt}")
            
            for msg in messages:
                if msg["role"] == "user":
                    full_dialog.append(f"User: {msg['content']}")
                elif msg["role"] == "assistant":
                    full_dialog.append(f"Assistant: {msg['content']}")
            
            last_question = messages[-1]["content"] if messages else ""
            full_dialog.append(f"User: {last_question}")
            
            prompt_text = "\n".join(full_dialog) + "\nAssistant:"
            prompts.append(prompt_text)
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

    def _save_vllm_results(self, outputs, id_list, questions_list):
        results = {}
        for output, item_id, question in zip(outputs, id_list, questions_list):
            results[item_id] = {
                "id": item_id,
                "question": question,
                "response": output.outputs[0].text.strip()
            }
        self._save_results(results)

class HuggingFaceGenerator(BaseGenerator):
    """Generator for HuggingFace models"""
    def __init__(self, config, prompt_builder, **kwargs):
        super().__init__(config, **kwargs)
        self.prompt_builder = prompt_builder
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            config["model_path"],
            trust_remote_code=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            config["model_path"],
            device_map="auto",
            trust_remote_code=True
        )

    def _call_api(self, messages, item_id, question, articles, system_prompt):
        try:
            full_messages = []
            if system_prompt:
                full_messages.append({"role": "system", "content": system_prompt})
            full_messages.extend(messages)
            
            inputs = self.tokenizer.apply_chat_template(
                full_messages,
                tokenize=False,
                add_generation_prompt=True
            )
            model_inputs = self.tokenizer(inputs, return_tensors="pt").to("cuda")
            
            outputs = self.model.generate(
                model_inputs.input_ids,
                do_sample=False
            )
            response = self.tokenizer.decode(
                outputs[0][len(model_inputs.input_ids[0]):], 
                skip_special_tokens=True
            )
            return {
                "id": item_id,
                "question": question,
                "response": response
            }
        except Exception as e:
            print(f"Generate Error：{str(e)}")
            return {"id": item_id, "question": question, "response": ""}
        
    def generate(self, processed_data, retrieval_data):
        for turn_num, samples in processed_data.items():
            messages_list, id_list, questions_list, articles_list, system_prompts = self._prepare_inputs(samples, retrieval_data)
            self._batch_call(messages_list, id_list, questions_list, articles_list, system_prompts)

    def _prepare_inputs(self, data, retrieval_data):
        messages_list = []
        id_list = []
        questions_list = []
        articles_list = []
        system_prompt = []
        for sample in data:
            articles = self._get_top_articles(sample["id"], retrieval_data, self.top_n)
            messages = self.prompt_builder.build_messages(
                sample["history"],
                sample["current_question"],
                articles
            )
            
            system_msg = next((msg["content"] for msg in messages if msg["role"] == "system"), "")

            messages_list.append([msg for msg in messages if msg["role"] != "system"])
            id_list.append(sample["id"])
            questions_list.append(sample["current_question"])
            articles_list.append(articles)
            system_prompt.append(system_msg)
            
        
        return messages_list, id_list, questions_list, articles_list, system_prompt
    

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

    def _batch_call(self, messages_list, id_list, questions_list, articles_list, system_prompts):
        result_dict = {}
        with ThreadPoolExecutor(max_workers=self.max_parallel) as executor:
            futures = []
            
            with tqdm(total=len(messages_list), desc="Generating responses") as pbar:
                for messages, item_id, question, articles, system_prompt in zip(messages_list, id_list, questions_list, articles_list, system_prompts):
                    future = executor.submit(self._call_api, messages, item_id, question, articles, system_prompt)
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
    
class LocalGenerator(HuggingFaceGenerator):
    def __init__(self, config, prompt_builder, **kwargs):
        super(HuggingFaceGenerator, self).__init__(config, **kwargs)
        self.prompt_builder = prompt_builder
        
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            config["model_path"],
            trust_remote_code=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            config["model_path"],
            device_map="cpu", 
            torch_dtype="auto", 
            trust_remote_code=True
        )
        
    def _call_api(self, messages, item_id, question, articles, system_prompt):
        try:
            full_messages = []
            if system_prompt:
                full_messages.append({"role": "system", "content": system_prompt})
            full_messages.extend(messages)
            
            inputs = self.tokenizer.apply_chat_template(
                full_messages,
                tokenize=False,
                add_generation_prompt=True
            )
            model_inputs = self.tokenizer(inputs, return_tensors="pt")
            
            outputs = self.model.generate(
                model_inputs.input_ids, 
                do_sample=False
            )
            response = self.tokenizer.decode(
                outputs[0][len(model_inputs.input_ids[0]):], 
                skip_special_tokens=True
            )
            return {
                "id": item_id,
                "question": question,
                "response": response
            }
        except Exception as e:
            print(f"Generate Error：{str(e)}")
            return {"id": item_id, "question": question, "response": ""}
        
class LocalGenerator(BaseGenerator):
    """Local Models"""
    def __init__(self, prompt_builder, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.prompt_builder = prompt_builder
        self._load_local_model()
        
    def _load_local_model(self):
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config["model_path"],
                trust_remote_code=True,
                local_files_only=True  
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config["model_path"],
                device_map="cuda",
                trust_remote_code=True,
                local_files_only=True  
            )
        except Exception as e:
            raise RuntimeError(f"Load Error！Check path：{self.config['model_path']}") from e

    def generate(self, processed_data, retrieval_data):
        for turn_num, samples in processed_data.items():
            messages_list, id_list, questions_list, articles_list, system_prompts = self._prepare_inputs(samples, retrieval_data)
            self._batch_call(messages_list, id_list, questions_list, articles_list, system_prompts)

    def _prepare_inputs(self, data, retrieval_data):
        messages_list = []
        id_list = []
        questions_list = []
        articles_list = []
        system_prompts = []
        for sample in data:
            articles = self._get_top_articles(sample["id"], retrieval_data, self.top_n)
            messages = self.prompt_builder.build_messages(
                sample["history"],
                sample["current_question"],
                articles
            )
            
            system_msg = next((msg["content"] for msg in messages if msg["role"] == "system"), "")
            
            messages_list.append([msg for msg in messages if msg["role"] != "system"])
            id_list.append(sample["id"])
            questions_list.append(sample["current_question"])
            articles_list.append(articles)
            system_prompts.append(system_msg)
            
        return messages_list, id_list, questions_list, articles_list, system_prompts

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
            self.logger.error(f"Error processing sample {sample_id}: {str(e)}")
            return []

    def _call_api(self, messages, item_id, question, articles, system_prompt):
        try:
            full_messages = []
            if system_prompt:
                full_messages.append({"role": "system", "content": system_prompt})
            full_messages.extend(messages)
            
            inputs = self.tokenizer.apply_chat_template(
                full_messages,
                tokenize=False,
                add_generation_prompt=True
            )
            model_inputs = self.tokenizer(inputs, return_tensors="pt").to(self.model.device)
            
            outputs = self.model.generate(
                model_inputs.input_ids, 
                do_sample=False
            )
            response = self.tokenizer.decode(
                outputs[0][len(model_inputs.input_ids[0]):], 
                skip_special_tokens=True
            )
            
            return {
                "id": item_id,
                "question": question,
                "response": response.strip()
            }
        except Exception as e:
            self.logger.error(f"Generate Error ID {item_id}: {str(e)}")
            return {"id": item_id, "question": question, "response": ""}

    def _batch_call(self, messages_list, id_list, questions_list, articles_list, system_prompts):
        result_dict = {}
        with ThreadPoolExecutor(max_workers=self.max_parallel) as executor:
            futures = []
            
            with tqdm(total=len(messages_list), desc="Local Model Generate") as pbar:
                for messages, item_id, question, articles, system_prompt in zip(
                    messages_list, id_list, questions_list, articles_list, system_prompts
                ):
                    future = executor.submit(
                        self._call_api, 
                        messages, 
                        item_id, 
                        question, 
                        articles, 
                        system_prompt
                    )
                    future.add_done_callback(lambda _: pbar.update(1))
                    futures.append(future)

                    if len(futures) >= self.batch_size:
                        self._process_batch(futures, result_dict)
                        futures.clear()
            
                if futures:
                    self._process_batch(futures, result_dict)
        
        return [result_dict[id] for id in sorted(id_list, key=lambda x: int(x.split("_")[0]))]

    def _process_batch(self, futures, result_dict):
        current_batch = []
        for future in as_completed(futures):
            result = future.result()
            result_dict[result["id"]] = result 
            current_batch.append(result)
        self._save_results({r["id"]: r for r in current_batch})