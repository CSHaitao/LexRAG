import faiss
import json
from tqdm import tqdm
import numpy as np
from pathlib import Path
import os
from retrieval.dense_retriever import DenseRetriever
from retrieval.lexical_matching import LexicalRetriever

class Pipeline:
    def __init__(self, config=None):
        self.openai_config = config or {}
        self.init_dir()

    def run_retriever(self, model_type, question_file_path, law_path, 
           bm25_backend="bm25s", faiss_type="FlatIP", model_name=None):
        if model_type == "bm25":
            self.pipeline_bm25(question_file_path, law_path, bm25_backend)
        elif model_type == "qld":
            self.pipeline_qld(question_file_path, law_path)
        else:
            self.pipeline_law(law_path, model_type, faiss_type, model_name)
            self.pipeline_question(question_file_path, model_type, model_name)
            self.pipeline_search(question_file_path, law_path, model_type, faiss_type)

    def pipeline_bm25(self, question_path, law_path, backend):
        res_path = f"data/retrieval/res/retrieval_bm25_{backend}.jsonl"
        
        with open(question_path, "r", encoding="utf-8") as f:
            data = [json.loads(line) for line in f]
        with open(law_path, "r", encoding="utf-8") as f:
            laws = [json.loads(line) for line in f]
        corpus = [law["name"] + law["content"] for law in laws]
        
        retriever = LexicalRetriever(bm25_backend=backend)
        queries = [conv["question"]["content"] for d in data for conv in d["conversation"]]
        
        if backend == "bm25s":
            result_idx_list, scores = retriever.search(corpus, law_path, queries, k=10)
            idx = 0
            for d in data:
                for conv in d["conversation"]:
                    tmp_laws = []
                    for result_idx, score in zip(result_idx_list[idx][0], scores[idx][0]):
                        tmp_laws.append({
                            "article": laws[result_idx],
                            "score": float(score)
                        })
                    conv["question"]["recall"] = tmp_laws
                    idx += 1
                    
        elif backend == "pyserini":
            results, scores = retriever.search(corpus, law_path, queries, k=10)
            idx = 0
            for d in data:
                for conv in d["conversation"]:
                    tmp_laws = []
                    for doc_id, score in zip(results[idx], scores[idx]):
                        tmp_laws.append({
                            "article": laws[int(doc_id)],
                            "score": float(score)
                        })
                    conv["question"]["recall"] = tmp_laws
                    idx += 1

        with open(res_path, "w", encoding="utf-8") as f:
            for d in data:
                f.write(json.dumps(d, ensure_ascii=False) + "\n")

    def pipeline_qld(self, question_path, law_path):
        res_path = "data/retrieval/res/retrieval_qld.jsonl"

        with open(question_path, "r", encoding="utf-8") as f:
            data = [json.loads(line) for line in f]
        with open(law_path, "r", encoding="utf-8") as f:
            laws = [json.loads(line) for line in f]
        corpus = [law["name"] + law["content"] for law in laws]
        
        retriever = LexicalRetriever()
        queries = [conv["question"]["content"] for d in data for conv in d["conversation"]]

        results, scores = retriever.search(corpus, law_path, queries, k=10, method="qld")
        idx = 0
        for d in data:
            for conv in d["conversation"]:
                tmp_laws = []
                for doc_id, score in zip(results[idx], scores[idx]):
                    tmp_laws.append({
                        "article": laws[int(doc_id)],
                        "score": float(score)
                    })
                conv["question"]["recall"] = tmp_laws
                idx += 1

        with open(res_path, "w", encoding="utf-8") as f:
            for d in data:
                f.write(json.dumps(d, ensure_ascii=False) + "\n")

    def pipeline_law(self, law_path, model_type, faiss_type, model_name):
        law_index_path = f"data/retrieval/law_index_{model_type}.faiss"
        if os.path.exists(law_index_path):
            return

        with open(law_path) as f:
            laws = [json.loads(line)["name"]+json.loads(line)["content"] for line in f]

        emb_model = DenseRetriever(**self.openai_config)
        embeddings = emb_model.embed(laws, model_type, model_name)
        emb_model.save_faiss(embeddings, faiss_type, law_index_path)

    def pipeline_question(self, question_path, model_type, model_name):
        question_emb_path = f"data/retrieval/npy/retrieval_{model_type}.npy"
        if os.path.exists(question_emb_path):
            return

        with open(question_path) as f:
            data = [json.loads(line) for line in f]
            questions = [q["question"]["content"] for d in data for q in d["conversation"]]

        emb_model = DenseRetriever(**self.openai_config)
        embeddings = emb_model.embed(questions, model_type, model_name)
        np.save(question_emb_path, embeddings)

    def pipeline_search(self, question_path, law_path, model_type, faiss_type):
        res_path = f"data/retrieval/res/retrieval_{model_type}.jsonl"
        law_index_path = f"data/retrieval/law_index_{model_type}.faiss"
        question_emb_path = f"data/retrieval/npy/retrieval_{model_type}.npy"

        index = faiss.read_index(law_index_path)
        question_embeds = np.load(question_emb_path)
        D, I = index.search(question_embeds.astype('float32'), 10)
        
        with open(law_path) as f:
            laws = [json.loads(line) for line in f]
        
        with open(question_path) as f:
            data = [json.loads(line) for line in f]
        
        self.incorporate_dense_results(data, laws, D, I, res_path)

    def incorporate_dense_results(self, data, laws, D, I, res_path):
        idx = 0
        for d in data:
            for conv in d["conversation"]:
                tmp_laws = []
                for i in range(len(I[idx])):
                    tmp_laws.append({
                        "article": laws[I[idx][i]], 
                        "score": float(D[idx][i])
                    })
                conv["question"]["recall"] = tmp_laws
                idx += 1
        
        with open(res_path, "w") as f:
            for d in data:
                f.write(json.dumps(d, ensure_ascii=False) + "\n")

    def init_dir(self):
        os.makedirs("data/retrieval/res", exist_ok=True)
        os.makedirs("data/retrieval/npy", exist_ok=True)
