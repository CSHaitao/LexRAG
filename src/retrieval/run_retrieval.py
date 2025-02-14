import faiss
import json
from tqdm import tqdm
import numpy as np
from pathlib import Path
import os
import argparse
from dense_retriever import EmbeddingModel
from bm25_retriever import Metrics
from bm25_retriever import BM25


QUESTION_TYPE_LIST = [
    "current_question",
    "prefix_question",
    "prefix_question_answer",
    "suffix_question",
    "llm_question",
]

class Pipeline:
    def __init__(self, args):
        self.model_type = args.model
        self.api_key = args.api_key
        self.base_url = args.base_url
        self.data_path = Path(args.data_path)
        self.dataset_path = Path(args.dataset_path)
        self.question_dir = Path(args.question_dir)
        self.output_dir = Path(args.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def run(self):
        if self.model_type == "bm25":
            self.pipeline_bm25()
            self.pipeline_metrcs(model_list=["bm25"])
        else:
            self.pipeline_question(model_list=[self.model_type], batch_size=32)
            self.pipeline_search(model_list=[self.model_type])
            self.pipeline_metrcs(model_list=[self.model_type])

    def pipeline_bm25(self):
        question_type_list = QUESTION_TYPE_LIST
        question_dir = self.question_dir
        law_path = self.data_path
        res_dir = self.output_dir

        question_file_path_list = [
            f"{question_dir}/{question_type}.jsonl"
            for question_type in question_type_list
        ]
        res_file_path_list = [
            f"{res_dir}/{question_type}_bm25.jsonl"
            for question_type in question_type_list
        ]
        for question_file_path, res_file_path in zip(
            question_file_path_list, res_file_path_list
        ):
            with open(question_file_path, "r", encoding="utf-8") as f:
                data = [json.loads(line) for line in f]
            with open(law_path, "r", encoding="utf-8") as f:
                laws = [json.loads(line) for line in f]
            corpus = [law["name"] + law["content"] for law in laws]
            bm25_model = BM25()
            query_list = []
            for d in data:
                for conversation in tqdm(d["conversation"]):
                    query_list.append(conversation["question"]["content"])
            result_idx_list, scores = bm25_model.bm25(corpus, query_list, k=10)
            result_idx_list, scores = result_idx_list, scores

            idx = 0
            for d in data:
                for conversation in d["conversation"]:
                    tmp_laws = []
                    for result_idx, score in zip(
                        result_idx_list[idx][0], scores[idx][0]
                    ):
                        tmp_laws.append({"article": laws[result_idx], "score": score})
                    conversation["question"]["recall"] = tmp_laws
                    idx += 1

            with open(res_file_path, "w", encoding="utf-8") as f:
                for d in data:
                    f.write(json.dumps(d, ensure_ascii=False) + "\n")

    def pipeline_metrcs(self, model_list, report_path="report.jsonl"):
        res_template_list = [
            "current_question",
            "prefix_question",
            "prefix_question_answer",
            "suffix_question",
            "llm_question",
        ]
        res_dir = self.output_dir
        res_file_list = [
            f"{res_dir}/{res_template}_{model}.jsonl"
            for res_template in res_template_list
            for model in model_list
        ]
        for res_file in res_file_list:
            self.metrics(res_file, report_path)

    def metrics(self, res_file_path, report_path="report.jsonl"):
        with open(res_file_path, "r", encoding="utf-8") as f:
            res_data = [json.loads(line) for line in f]

        res_list = []
        res_score_list = []
        label_list = []
        for data in res_data:
            for conversation in data["conversation"]:
                res = [
                    law["article"]["name"] for law in conversation["question"]["recall"]
                ]
                score = [law["score"] for law in conversation["question"]["recall"]]
                label = conversation["article"]
                res_list.append(res)
                res_score_list.append(score)
                label_list.append(label)

        k_list = [1, 3, 5, 10]
        metrics = Metrics()
        report = {"file": res_file_path}
        for k in k_list:
            res_list_k = [res[:k] for res in res_list]
            recall = metrics.recall(res_list_k, label_list)
            print(f"Recall@{k}: {recall}")
            ndcg = metrics.nDCG(res_list, res_score_list, label_list, k=k)
            print(f"nDCG@{k}: {ndcg}")
            report[f"Recall@{k}"] = recall
            report[f"nDCG@{k}"] = ndcg
        with open(report_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(report, ensure_ascii=False) + "\n")

    def pipeline_law(self, model_list, batch_size=8):
        for model in model_list:
            self.embed_law(model, batch_size=batch_size)

    def pipeline_question(self, model_list, batch_size=8):
        question_type_list = QUESTION_TYPE_LIST
        question_dir = self.question_dir
        question_file_path_list = [
            f"{question_dir}/{question_type}.jsonl"
            for question_type in question_type_list
        ]
        for model in model_list:
            for question_file_path in question_file_path_list:
                self.embed_question(model, question_file_path, batch_size=batch_size)

    def pipeline_search(self, model_list):
        question_type_list = QUESTION_TYPE_LIST
        question_npy_dir = "retrieval/npy"
        question_dir = self.question_dir
        law_dir = "retrieval/data"
        law_path = self.data_path
        res_dir = self.output_dir

        for model in model_list:
            law_faiss_path = f"{law_dir}/law_index_{model}.faiss"
            question_npy_path_list = [
                f"{question_npy_dir}/{question_type}_embeddings_{model}.npy"
                for question_type in question_type_list
            ]
            question_file_path_list = [
                f"{question_dir}/{question_type}.jsonl"
                for question_type in question_type_list
            ]
            res_file_path_list = [
                f"{res_dir}/{question_type}_{model}.jsonl"
                for question_type in question_type_list
            ]
            for question_npy_path, question_file_path, res_file_path in zip(
                question_npy_path_list, question_file_path_list, res_file_path_list
            ):
                D, I = self.search(law_faiss_path, question_npy_path, top_k=10)
                self.incorporate_law(question_file_path, law_path, D, I, res_file_path)

    def embed_law(self, model_name, batch_size=8):
        law_path = self.data_path
        with open(law_path, "r", encoding="utf-8") as f:
            laws = [
                json.loads(line)["name"] + json.loads(line)["content"] for line in f
            ]

        embedding_model = EmbeddingModel(api_key=self.api_key, base_url=self.base_url)
        embeddings = embedding_model.embed(laws, model_name, batch_size)
        embedding_model.save_faiss(
            embeddings, f"retrieval/data/law_index_{model_name}.faiss"
        )

    def embed_question(self, model_name, question_path, batch_size=8):
        file_name = Path(question_path).stem
        with open(question_path, "r", encoding="utf-8") as f:
            data = [json.loads(line) for line in f]
        questions = []
        for d in data:
            for conversation in d["conversation"]:
                questions.append(conversation["question"]["content"])

        embedding_model = EmbeddingModel(api_key=self.api_key, base_url=self.base_url)
        embeddings = embedding_model.embed(questions, model_name, batch_size)

        with open(f"retrieval/npy/{file_name}_embeddings_{model_name}.npy", "wb") as f:
            np.save(f, embeddings)

    def search(self, law_faiss_path, question_npy_path, top_k=10):
        law_index = faiss.read_index(law_faiss_path)
        question_embeddings = np.load(question_npy_path)
        D, I = law_index.search(question_embeddings, top_k)
        return D, I

    def incorporate_law(self, question_path, law_path, D, I, res_path):
        with open(question_path, "r", encoding="utf-8") as f:
            data = [json.loads(line) for line in f]
        with open(law_path, "r", encoding="utf-8") as f:
            laws = [json.loads(line) for line in f]

        idx = 0
        for d in data:
            for conversation in d["conversation"]:
                tmp_laws = []
                for I_idx, D_score in zip(I[idx], D[idx]):
                    tmp_laws.append({"article": laws[I_idx], "score": float(D_score)})
                conversation["question"]["recall"] = tmp_laws
                idx += 1

        with open(res_path, "w", encoding="utf-8") as f:
            for d in data:
                f.write(json.dumps(d, ensure_ascii=False) + "\n")

    def generate_question(self):
        def _current_question(data_list, file_path):
            for data in data_list:
                for conversation in data["conversation"]:
                    conversation["question"] = {
                        "type": "current_question",
                        "content": conversation["user"],
                    }
            with open(file_path, "w", encoding="utf-8") as f:
                for data in data_list:
                    f.write(json.dumps(data, ensure_ascii=False) + "\n")

        def _prefix_question(data_list, file_path):
            for data in data_list:
                prefix_question = ""
                for conversation in data["conversation"]:
                    prefix_question += conversation["user"]
                    conversation["question"] = {
                        "type": "prefix_question",
                        "content": prefix_question,
                    }
            with open(file_path, "w", encoding="utf-8") as f:
                for data in data_list:
                    f.write(json.dumps(data, ensure_ascii=False) + "\n")

        def _prefix_question_answer(data_list, file_path):
            for data in data_list:
                prefix_question_answer = ""
                for conversation in data["conversation"]:
                    prefix_question_answer += f" {conversation['user']}\n\n"
                    conversation["question"] = {
                        "type": "prefix_question_answer",
                        "content": prefix_question_answer,
                    }
                    prefix_question_answer += f"{conversation['assistant']}\n\n"
            with open(file_path, "w", encoding="utf-8") as f:
                for data in data_list:
                    f.write(json.dumps(data, ensure_ascii=False) + "\n")

        def _suffix_question(data_list, file_path):
            for data in data_list:
                suffix_question = ""
                for idx in range(len(data["conversation"]) - 1, -1, -1):
                    suffix_question += f"{data['conversation'][idx]['user']}\n\n"
                    data["conversation"][idx]["question"] = {
                        "type": "suffix_question",
                        "content": suffix_question,
                    }
            with open(file_path, "w", encoding="utf-8") as f:
                for data in data_list:
                    f.write(json.dumps(data, ensure_ascii=False) + "\n")

        data_list = self.dataset_path
        current_file_path = "retrieval/question/current_question.jsonl"
        prefix_question_file_path = "retrieval/question/prefix_question.jsonl"
        prefix_question_answer_file_path = (
            "retrieval/question/prefix_question_answer.jsonl"
        )
        suffix_question_file_path = "retrieval/question/suffix_question.jsonl"

        with open(data_list, "r", encoding="utf-8") as f:
            data_list = json.load(f)

        _current_question(data_list, current_file_path)
        _prefix_question(data_list, prefix_question_file_path)
        _prefix_question_answer(data_list, prefix_question_answer_file_path)
        _suffix_question(data_list, suffix_question_file_path)


def init_dir():
    question_dir = Path(args.question_dir)
    law_dir = "retrieval/data"
    res_dir = Path(args.output_dir)
    npy_dir = "retrieval/npy"
    for dir_path in [question_dir, law_dir, res_dir, npy_dir]:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, 
                       choices=["BGE-base-zh", "Qwen2-1.5B", "bm25", "openai"],
                       help="Select retrieval model")
    parser.add_argument("--data-path", required=True, help="Path to law library")
    parser.add_argument("--question-dir", required=True, help="Question files directory")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument("--api-key", help="OpenAI API key (required for openai model)")
    parser.add_argument("--base-url", help="OpenAI API base URL (required for openai model)")
    parser.add_argument("--dataset-path", required=True, help="Path to dataset")
    args = parser.parse_args()
    
    if args.model == "openai" and not (args.api_key and args.base_url):
        raise ValueError("OpenAI model requires --api-key and --base-url")
    
    pipeline = Pipeline(args)
    pipeline.pipeline_law()
    pipeline.run()