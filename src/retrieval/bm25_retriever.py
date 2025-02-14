from tqdm import tqdm
import numpy as np
from sklearn.metrics import ndcg_score
import bm25s
import jieba
from bm25s.tokenization import Tokenized

class Metrics:
    def __init__(self):
        pass

    def recall(self, res_list: list[list[str]], label_list: list[list[str]]) -> float:
        """
        Calculate Recall
        :param res_list: list of positive examples predicted by the model, each element is a list of strings
        :param label_list: list of actual positive examples, each element is a list of strings

        :return: Recall
        """
        if not label_list or not res_list:
            raise ValueError("input list cannot be empty")

        if len(label_list) != len(res_list):
            raise ValueError("label_list and res_list must be the same length")
        true_positives = 0
        false_negatives = 0
        for actual, predicted in zip(label_list, res_list):
            actual_set = set(actual)
            predicted_set = set(predicted)

            true_positives += len(actual_set & predicted_set)
            false_negatives += len(actual_set - predicted_set)

        if true_positives + false_negatives == 0:
            return 0.0

        recall = true_positives / (true_positives + false_negatives)
        return recall

    def nDCG(
        self,
        data_list: list[list[str]],
        score_list: list[list[float]],
        label_list: list[list[str]],
        k=10,
    ) -> float:
        """
        Calculate nDCG(support variable length recommendation lists)
        :param data_list: list of sorted items, each sub-list can be of different lengths
        :param score_list: list of scores for corresponding items, length must be the same as data_list
        :param label_list: list of real labels for each item
        :param k: length of the recommendation list
        :return: average nDCG
        """
        ndcg_scores = []
        for data, scores, labels in zip(data_list, score_list, label_list):
            true_scores = [1 if item in labels else 0 for item in data]
            y_true = np.array([true_scores])
            y_score = np.array([scores])

            try:
                score = ndcg_score(y_true, y_score, k=k)
            except ValueError:
                score = 0.0
            ndcg_scores.append(score)
        return np.mean(ndcg_scores)


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