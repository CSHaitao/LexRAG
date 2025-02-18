import faiss
import json
from tqdm import tqdm
import numpy as np
from pathlib import Path
import os
from retrieval.dense_retriever import EmbeddingModel
from retrieval.bm25_retriever import BM25

class Pipeline:
    def __init__(self):
        pass

    def run(self, model_type, question_file_path, law_path):
        self.init_dir()
        if model_type == "bm25":
            self.pipeline_bm25(question_file_path, law_path)
        else:
            self.pipeline_law(law_path, model_type)
            self.pipeline_question(question_file_path, model_type, batch_size=32)
            self.pipeline_search(question_file_path, law_path, model_type)

    def pipeline_bm25(self, question_file_path, law_path):
        res_dir = "data/retrieval/res"
        res_file_path = f"{res_dir}/retrieval_bm25.jsonl"

        with open(question_file_path, "r", encoding="utf-8") as f:
            data = [json.loads(line) for line in f]
        with open(law_path, "r", encoding="utf-8") as f:
            laws = [json.loads(line) for line in f]
        corpus = [law["name"] + law["content"] for law in laws]
        bm25_model = BM25()
        query_list = [conv["question"]["content"] for d in data for conv in d["conversation"]]
        result_idx_list, scores = bm25_model.bm25(corpus, query_list, k=10)

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

    def pipeline_law(self, law_path, model_name, batch_size=8):
            self.embed_law(law_path, model_name, batch_size=batch_size)

    def pipeline_question(self, question_file_path, model_name, batch_size=8):
        self.embed_question(model_name, question_file_path, batch_size=batch_size)

    def pipeline_search(self, question_file_path, law_path, model_name):
        law_dir = "data/retrieval"
        res_dir = "data/retrieval/res"
        law_faiss_path = f"{law_dir}/law_index_{model_name}.faiss"
        question_embeds = np.load(f"data/retrieval/npy/retrieval_{model_name}.npy")
        res_file_path = f"{res_dir}/retrieval_{model_name}.jsonl"
        D, I = self.search(law_faiss_path, question_embeds, top_k=10)
        self.incorporate_law(question_file_path, law_path, D, I, res_file_path)

    def embed_law(self, law_path, model_name, batch_size=8):
        with open(law_path, "r", encoding="utf-8") as f:
            laws = [
                json.loads(line)["name"] + json.loads(line)["content"] for line in f
            ]

        embedding_model = EmbeddingModel()
        embeddings = embedding_model.embed(laws, model_name, batch_size)
        embedding_model.save_faiss(
            embeddings, f"data/retrieval/law_index_{model_name}.faiss"
        )

    def embed_question(self, model_name, question_path, batch_size=8):
        with open(question_path, "r", encoding="utf-8") as f:
            data = [json.loads(line) for line in f]
        questions = []
        for d in data:
            for conversation in d["conversation"]:
                questions.append(conversation["question"]["content"])

        embedding_model = EmbeddingModel()
        embeddings = embedding_model.embed(questions, model_name, batch_size)

        with open(f"data/retrieval/npy/retrieval_{model_name}.npy", "wb") as f:
            np.save(f, embeddings)

    def search(self, law_faiss_path, question_npy_path, top_k=10):
        law_index = faiss.read_index(law_faiss_path)
        if isinstance(question_npy_path, np.ndarray):
            question_embeddings = question_npy_path
        else:
            question_embeddings = np.load(question_npy_path)
        if question_embeddings.dtype != np.float32:
            question_embeddings = question_embeddings.astype(np.float32)
        if len(question_embeddings.shape) == 1:
            question_embeddings = question_embeddings.reshape(1, -1)
        assert law_index.d == question_embeddings.shape[1], \
            f"Index dim {law_index.d} != Embedding dim {question_embeddings.shape[1]}"
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
        """
        generate some types of questions
        """
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

        data_list = "data/dataset.json"
        current_file_path = "data/retrieval/question/current_question.jsonl"
        prefix_question_file_path = "data/retrieval/question/prefix_question.jsonl"
        prefix_question_answer_file_path = (
            "data/retrieval/question/prefix_question_answer.jsonl"
        )
        suffix_question_file_path = "data/retrieval/question/suffix_question.jsonl"

        with open(data_list, "r", encoding="utf-8") as f:
            data_list = json.load(f)

        _current_question(data_list, current_file_path)
        _prefix_question(data_list, prefix_question_file_path)
        _prefix_question_answer(data_list, prefix_question_answer_file_path)
        _suffix_question(data_list, suffix_question_file_path)


    def init_dir(self):
        law_dir = "data/retrieval"
        res_dir = "data/retrieval/res"
        npy_dir = "data/retrieval/npy"
        for dir_path in [law_dir, res_dir, npy_dir]:
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)
