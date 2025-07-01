# interpretability/utils.py

import numpy as np
from scipy.special import softmax
from typing import Optional


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    Computes cosine similarity between two vectors.

    Parameters:
        vec1 (np.ndarray): First vector.
        vec2 (np.ndarray): Second vector.

    Returns:
        float: Cosine similarity between the two vectors.
    """
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return float(np.dot(vec1, vec2) / (norm1 * norm2))


def entropy_from_logits(logits: np.ndarray, base: Optional[float] = None) -> float:
    """
    Computes the entropy from raw logits.

    Parameters:
        logits (np.ndarray): The raw logits from the model output.
        base (Optional[float]): Logarithm base (e.g. 2 for bits). Defaults to natural log.

    Returns:
        float: Entropy of the predicted distribution.
    """
    probs = softmax(logits)
    log_probs = np.log(probs)
    if base:
        log_probs /= np.log(base)
    entropy = -np.sum(probs * log_probs)
    return float(entropy)


def top_k_from_logits(logits: np.ndarray, k: int = 5) -> list:
    """
    Returns the indices and values of the top-k logits.

    Parameters:
        logits (np.ndarray): Raw model logits.
        k (int): Number of top elements to return.

    Returns:
        list: List of (index, logit value) tuples.
    """
    top_indices = np.argpartition(-logits, k)[:k]
    top_logits = logits[top_indices]
    sorted_pairs = sorted(zip(top_indices, top_logits), key=lambda x: -x[1])
    return [(int(idx), float(val)) for idx, val in sorted_pairs]
