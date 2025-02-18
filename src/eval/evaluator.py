import os
import json
import logging
from typing import Dict, List
from .metrics import UnifiedEvaluator
from generate.data_processor import DataProcessor
from .retrieval_metrics import RetrievalMetrics

logging.basicConfig(
    level=logging.INFO, 
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(), 
        logging.FileHandler("log", mode='w', encoding='utf-8')  
    ]
)

class BaseEvaluator:
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)

class GenerationEvaluator(BaseEvaluator):
    def __init__(self, config):
        super().__init__(config)
        self.metric_calculator = UnifiedEvaluator()
    
    def evaluate(self, data_path, response_file, metrics):
        results = {}
        id_to_response = self._load_responses(response_file)
        
        # Segregated multi-stage data
        processor = DataProcessor()
        processed_data = processor.process_conversation_turns(data_path)
        output_dir = "data/generated_samples"
        os.makedirs(output_dir, exist_ok=True)
        for turn_num, samples in processed_data.items():
            output_path = f"{output_dir}/{turn_num}.json"
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(samples, f, indent=2, ensure_ascii=False)
        
        for turn_file in sorted(os.listdir(output_dir)):
            turn_path = os.path.join(output_dir, turn_file)
            with open(turn_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            preds, refs, keywords = self._prepare_data(data, id_to_response)

            metrics_result = {}
            if "rouge" in metrics:
                metrics_result.update(self.metric_calculator._get_rouge(preds, refs))
            if "bert-score" in metrics:
                metrics_result.update(self.metric_calculator._get_bert_score(preds, refs))
            if "bleu" in metrics:
                metrics_result.update(self.metric_calculator._get_bleu(preds, refs))
            if "keyword_accuracy" in metrics:
                metrics_result.update(self.metric_calculator._get_keyword_accuracy(keywords, preds))
            if "char-scores" in metrics:
                metrics_result.update(self.metric_calculator._get_char_f1(preds, refs))
            
            turn_num = os.path.splitext(turn_file)[0].split('_')[-1]
            results[turn_num] = {k:v for k,v in metrics_result.items() if k in metrics}
            logging.info(f"\nTurn{turn_num.upper()} Metrics:")
            for k, v in metrics_result.items():
                logging.info(f"{k.ljust(15)}: {v:.4f}")
        
        return results

    def _load_responses(self, response_file):
        id_to_response = {}
        with open(response_file, "r", encoding="utf-8") as f:
            for line in f:
                record = json.loads(line)
                id_to_response[record["id"]] = record["response"]
        return id_to_response

    def _prepare_data(self, data, id_to_response):
        preds, refs, keywords = [], [], []
        for sample in data:
            pred_response = id_to_response.get(sample["id"], "").strip()
            if pred_response:
                preds.append(pred_response)
                refs.append(sample["reference"])
                keywords.append(sample["keywords"])
        return preds, refs, keywords

class LLMJudge(BaseEvaluator):
    def __init__(self, config):
        super().__init__(config)
        from openai import OpenAI
        from zhipuai import ZhipuAI
        import httpx
        from eval.llm_as_judge.run_judge import Judge
        if self.config["model_type"] == "openai":
            self.client = OpenAI(
                base_url=self.config["api_base"],
                api_key=self.config["api_key"],
                http_client=httpx.Client(
                    base_url=self.config["api_base"],
                    follow_redirects=True,
                ),
            )
        elif self.config["model_type"] == "zhipu":
            self.client = ZhipuAI(api_key=self.config["api_key"])
        elif self.config["model_type"] in ["qwen","llama"]:
            self.client = OpenAI(
                base_url=self.config["api_base"], 
                api_key=self.config["key"]
            )
        self.model_name = self.config["model_name"]
        Judge(self.config)
    
    def evaluate(self, data_path, gen_path):
        from eval.llm_as_judge.make_prompt import process_model
        from eval.llm_as_judge.run_judge import process_turn
        
        process_model(
            data_path,
            gen_path
        )
        
        for turn in range(1, 6):
            process_turn(
                self.config,
                turn=turn
            )
    
class RetrievalEvaluator(BaseEvaluator):
    def evaluate(self, results_path: str, metrics: List[str], k_values: List[int], report_path="data/retrieval/report.jsonl") -> Dict:
        with open(results_path, "r", encoding="utf-8") as f:
            res_data = [json.loads(line) for line in f]

        res_list, res_score_list, label_list = [], [], []
        for data in res_data:
            for conv in data["conversation"]:
                res = [law["article"]["name"] for law in conv["question"]["recall"]]
                scores = [law["score"] for law in conv["question"]["recall"]]
                label = conv["article"]
                res_list.append(res)
                res_score_list.append(scores)
                label_list.append(label)

        report = {"results_path": results_path}
        metric_functions = {
            "recall": RetrievalMetrics.recall,
            "precision": RetrievalMetrics.precision,
            "f1": RetrievalMetrics.f1_score,
            "mrr": RetrievalMetrics.mrr,
            "ndcg": RetrievalMetrics.ndcg
        }

        for metric in metrics:
            if metric not in metric_functions:
                continue
            for k in k_values:
                if metric == "ndcg":
                    score = metric_functions[metric](res_list, res_score_list, label_list, k)
                else:
                    score = metric_functions[metric](res_list, label_list, k)
                report[f"{metric}@{k}"] = score
                self.logger.info(f"{metric.upper()}@{k}: {score:.4f}")
        report_dir = os.path.dirname(report_path)
        os.makedirs(report_dir, exist_ok=True)
        with open(report_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(report, ensure_ascii=False) + "\n")
        return report
