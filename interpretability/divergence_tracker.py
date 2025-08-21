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
    influence_curve: List[float],
    threshold: float = 0.5,
    smooth_ema_alpha: float = 0.2,   # None or 0 to disable smoothing
    require_k: int = 2,              # consecutive points below threshold
    post_peak: bool = True           # start search after argmax
) -> int:
    """
    Half-life = first index (optionally post-peak) where the curve stays below
    threshold * reference for k consecutive points.
    - reference is max(curve) on the (optionally) smoothed series.
    - returns last index if it never crosses.
    """
    if not influence_curve:
        return -1

    x = list(influence_curve)

    # optional smoothing
    if smooth_ema_alpha and smooth_ema_alpha > 0:
        y = [x[0]]
        a = float(smooth_ema_alpha)
        for v in x[1:]:
            y.append(a * v + (1 - a) * y[-1])
        x = y

    ref = max(x)
    if ref <= 0:
        return len(x) - 1  # degenerate flat/zero

    start_idx = int(np.argmax(x)) if post_peak else 0
    thresh = threshold * ref

    consec = 0
    for i in range(start_idx, len(x)):
        if x[i] < thresh:
            consec += 1
            if consec >= require_k:
                return i
        else:
            consec = 0
    return len(x) - 1

