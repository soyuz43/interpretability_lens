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


def measure_half_life(
    influence_curve, threshold=0.5, relative_to="first", smooth=None, require_k=1
):
    """
    Half-life = first index where curve < threshold * reference.
    reference: 'first' or 'max'
    smooth: None, or ('ema', alpha) where alpha in (0,1]
    require_k: require k consecutive points below threshold
    """
    if not influence_curve:
        return -1
    x = list(influence_curve)
    if smooth and smooth[0] == "ema":
        alpha = float(smooth[1])
        y = [x[0]]
        for v in x[1:]:
            y.append(alpha*v + (1-alpha)*y[-1])
        x = y
    ref = x[0] if relative_to == "first" else max(x)
    consec = 0
    for i, v in enumerate(x):
        if v < ref * threshold:
            consec += 1
            if consec >= require_k:
                return i
        else:
            consec = 0
    return len(x) - 1

