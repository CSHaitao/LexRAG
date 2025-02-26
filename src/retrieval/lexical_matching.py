import os
import json
import jieba
import langid
import bm25s
import numpy as np
from tqdm import tqdm
from pyserini.search.lucene import LuceneSearcher
from bm25s.tokenization import Tokenized
import math
import subprocess

def judge_zh(text: str) -> bool:
    return langid.classify(text)[0] == 'zh'

class LexicalRetriever:
    def __init__(self, bm25_backend=None):
        self.bm25_backend = bm25_backend
        self.searcher = None
    
    def _bm25s_tokenize(
        self,
        texts,
        return_ids: bool = True,
        show_progress: bool = True,
        leave: bool = False,
    ):
        if isinstance(texts, str):
            texts = [texts]

        corpus_ids = []
        token_to_index = {}

        for text in tqdm(
            texts, desc="Split strings", leave=leave, disable=not show_progress
        ):

            splitted = jieba.lcut(text)
            doc_ids = []

            for token in splitted:
                if token not in token_to_index:
                    token_to_index[token] = len(token_to_index)

                token_id = token_to_index[token]
                doc_ids.append(token_id)

            corpus_ids.append(doc_ids)

        unique_tokens = list(token_to_index.keys())
        vocab_dict = token_to_index

        if return_ids:
            return Tokenized(ids=corpus_ids, vocab=vocab_dict)

        else:
            reverse_dict = unique_tokens
            for i, token_ids in enumerate(
                tqdm(
                    corpus_ids,
                    desc="Reconstructing token strings",
                    leave=leave,
                    disable=not show_progress,
                )
            ):
                corpus_ids[i] = [reverse_dict[token_id] for token_id in token_ids]

            return corpus_ids
    
    def _bm25s_search(self, corpus, query_list, k=10):
        bm25s.tokenize = self._bm25s_tokenize
        coupus_token = bm25s.tokenize(corpus)
        retriever = bm25s.BM25()
        retriever.index(coupus_token)

        query_token_list = [bm25s.tokenize(query) for query in query_list]
        scores = []
        result_idx_list = []
        for query_token in query_token_list:
            result, score = retriever.retrieve(query_token, k=k)
            scores.append(score.tolist())
            result_idx_list.append(result.tolist())
        print(np.array(result_idx_list).shape, np.array(scores).shape)
        return result_idx_list, scores

    def _build_pyserini_index(self, corpus_path, folder_path, index_dir):
        temp_path = "data/law_library/temp.jsonl"

        args = [
            "-collection", "JsonCollection",
            "-input", folder_path,
            "-index", index_dir,
            "-generator", "DefaultLuceneDocumentGenerator",
            "-threads", "1",
        ]
        
        # Detecting Chinese
        with open(temp_path) as f:
            sample_text = json.loads(next(f))['contents']
            lang = 'zh' if judge_zh(sample_text) else 'en'
        if lang == 'zh':
            args += ["-language", "zh"]
        
        os.makedirs(index_dir, exist_ok=True)
        subprocess.run(["python", "-m", "pyserini.index.lucene"] + args)
        self.searcher = LuceneSearcher(index_dir)
        if lang == 'zh':
            self.searcher.set_language('zh')

    def _bm25_search(self, queries, k=10):
        results = []
        scores = []
        for query in tqdm(queries, desc="Pyserini BM25 Searching"):
            hits = self.searcher.search(query, k=k)
            results.append([hit.docid for hit in hits])
            scores.append([hit.score for hit in hits])
        return results, scores

    def _qld_search(self, queries, k=10, index_dir="data/pyserini_index"):
        results = []
        scores = []
        self.searcher.set_qld()
        for query in tqdm(queries, desc="QLD Searching"):
            hits = self.searcher.search(query, k=k)
            results.append([hit.docid for hit in hits])
            scores.append([hit.score for hit in hits])
        return results, scores

    def search(self, corpus, law_path, queries, k=10, method="bm25"):
        output_directory = "data/law_library"
        os.makedirs(output_directory, exist_ok=True)    
        temp_file_path = os.path.join(output_directory, 'temp.jsonl')
        with open(law_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()
        with open(temp_file_path, 'w', encoding='utf-8') as temp_file:
            for line in lines:
                data = json.loads(line)
                if 'content' in data:
                    data['contents'] = data.pop('content')
                temp_file.write(json.dumps(data, ensure_ascii=False) + '\n')

        folder_path = output_directory
        if method == "bm25":
            if self.bm25_backend == "bm25s":
                return self._bm25s_search(corpus, queries, k)
            elif self.bm25_backend == "pyserini":
                self._build_pyserini_index(law_path, folder_path, "data/retrieval/pyserini_index")
                return self._bm25_search(queries, k)
        elif method == "qld":
            self._build_pyserini_index(law_path, folder_path, "data/retrieval/qld_index")
            return self._qld_search(queries, k)
        else:
            raise ValueError(f"Unsupported method: {method}")