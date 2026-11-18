import numpy as np


def recall_at_k(ranked_ids, relevant_ids, k):
    relevant = set(relevant_ids)
    return len(set(ranked_ids[:k]) & relevant) / len(relevant) if relevant else None


def precision_at_k(ranked_ids, relevant_ids, k):
    if k <= 0:
        return 0.0
    relevant = set(relevant_ids)
    if not relevant:
        return 0.0
    return len(set(ranked_ids[:k]) & relevant) / k


def reciprocal_rank(ranked_ids, relevant_ids):
    relevant = set(relevant_ids)
    for i, did in enumerate(ranked_ids, start=1):
        if did in relevant:
            return 1.0 / i
    return 0.0


def ndcg_at_k(ranked_ids, relevant_ids, k):
    rel = set(relevant_ids)
    if not rel or k <= 0:
        return 0.0
    gains = np.asarray([1.0 if x in rel else 0.0 for x in ranked_ids[:k]])
    dcg = np.sum(gains / np.log2(np.arange(2, len(gains) + 2)))
    ideal_n = min(len(rel), k)
    idcg = np.sum(1.0 / np.log2(np.arange(2, ideal_n + 2)))
    return float(dcg / idcg) if idcg else 0.0


def compute_all_metrics(ranked_ids, relevant_ids, k):
    relevant = set(relevant_ids)
    if not relevant:
        return None
    return {
        "recall_at_k": recall_at_k(ranked_ids, relevant, k),
        "precision_at_k": precision_at_k(ranked_ids, relevant, k),
        "mrr": reciprocal_rank(ranked_ids, relevant),
        "ndcg_at_k": ndcg_at_k(ranked_ids, relevant, k),
    }
