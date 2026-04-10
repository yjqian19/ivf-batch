import numpy as np


def recall_at_k(predicted_ids: np.ndarray, ground_truth: np.ndarray, k: int) -> float:
    """Fraction of queries where the true nearest neighbor is in predicted top-k."""
    hits = sum(gt[0] in pred[:k] for pred, gt in zip(predicted_ids, ground_truth))
    return hits / len(predicted_ids)
