# interpretability/divergence_tracker.py

import numpy as np
from typing import List, Dict


def compute_semantic_drift(projected_coords: np.ndarray) -> Dict[str, object]:
    """
    Given a sequence of reduced latent vectors (e.g., via PCA or UMAP),
    compute the semantic drift over time:
    - Cosine distance between consecutive points (local drift)
    - Arc length of the trajectory
    - Distance from the initial point at each step (conceptual coherence over time)
    """
    num_points = projected_coords.shape[0]
    cosine_drift = []
    distance_from_origin = []
    arc_length = 0.0

    for i in range(1, num_points):
        v_prev = projected_coords[i - 1]
        v_curr = projected_coords[i]

        # Local cosine drift
        cos_sim = np.dot(v_prev, v_curr) / (
            np.linalg.norm(v_prev) * np.linalg.norm(v_curr) + 1e-8
        )
        cosine_distance = 1.0 - cos_sim
        cosine_drift.append(cosine_distance)

        # Arc length
        step_distance = np.linalg.norm(v_curr - v_prev)
        arc_length += step_distance

        # Distance from origin
        origin_distance = np.linalg.norm(v_curr - projected_coords[0])
        distance_from_origin.append(origin_distance)

    return {
        "arc_length": arc_length,
        "cosine_drift": cosine_drift,
        "distance_from_origin": distance_from_origin,
    }


def measure_half_life(influence_curve: List[float], threshold: float = 0.5) -> int:
    """
    Given a list of values representing conceptual influence decay
    (e.g. distance from initial state), find the point where influence
    drops below a given threshold (defaults to 50% of max).
    
    Returns the token index where this decay threshold is crossed.
    """
    if not influence_curve:
        return -1

    peak = influence_curve[0]
    for i, val in enumerate(influence_curve):
        if val >= peak * threshold:
            continue
        return i

    return len(influence_curve) - 1
