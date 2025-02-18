from tqdm import tqdm
import numpy as np
import bm25s
import jieba
from bm25s.tokenization import Tokenized


def tokenize(
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


class BM25:
    def __init__(self):
        pass

    def bm25(self, corpus, query_list, k=1):
        bm25s.tokenize = tokenize
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