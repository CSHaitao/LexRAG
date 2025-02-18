import numpy as np
from sklearn.metrics import ndcg_score

class RetrievalMetrics:
    @staticmethod
    def recall(res_list: list[list[str]], label_list: list[list[str]], k: int) -> float:
        true_positives = 0
        false_negatives = 0
        for actual, predicted in zip(label_list, res_list):
            actual_set = set(actual)
            predicted_set = set(predicted[:k])
            true_positives += len(actual_set & predicted_set)
            false_negatives += len(actual_set - predicted_set)
        return true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0

    @staticmethod
    def precision(res_list: list[list[str]], label_list: list[list[str]], k: int) -> float:
        true_positives = 0
        false_positives = 0
        for actual, predicted in zip(label_list, res_list):
            actual_set = set(actual)
            predicted_set = set(predicted[:k])
            true_positives += len(actual_set & predicted_set)
            false_positives += len(predicted_set - actual_set)
        return true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0

    @staticmethod
    def f1_score(res_list: list[list[str]], label_list: list[list[str]], k: int) -> float:
        prec = RetrievalMetrics.precision(res_list, label_list, k)
        rec = RetrievalMetrics.recall(res_list, label_list, k)
        return 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0

    @staticmethod
    def mrr(res_list: list[list[str]], label_list: list[list[str]], k: int) -> float:
        reciprocal_ranks = []
        for actual, predicted in zip(label_list, res_list):
            for i, item in enumerate(predicted[:k], 1):
                if item in actual:
                    reciprocal_ranks.append(1 / i)
                    break
            else:
                reciprocal_ranks.append(0)
        return np.mean(reciprocal_ranks)

    @staticmethod
    def ndcg(data_list: list[list[str]], score_list: list[list[float]], label_list: list[list[str]], k: int) -> float:
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
    