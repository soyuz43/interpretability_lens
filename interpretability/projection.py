# interpretability/projection.py

import numpy as np
from typing import List, Literal, Optional, Dict, Any
from sklearn.decomposition import PCA
import umap


def _pca_project(matrix: np.ndarray, n_components: int) -> np.ndarray:
    """
    PCA projection with safety for tiny N or D:
    - clamp n_components to feasible range
    - zero-pad columns if needed to return the requested shape
    """
    n_samples, n_features = matrix.shape
    feasible = max(1, min(n_components, n_samples, n_features))
    reducer = PCA(n_components=feasible)
    proj = reducer.fit_transform(matrix)  # shape: (N, feasible)
    if feasible < n_components:
        pad = n_components - feasible
        proj = np.pad(proj, ((0, 0), (0, pad)), mode="constant")
    return proj


def _safe_n_neighbors(requested: int, n_samples: int) -> int:
    """
    Compute a UMAP-safe neighbor count:
    - must be at least 2
    - must be at most n_samples - 1
    """
    if n_samples <= 1:
        return 1  # unused, we will not call UMAP with N <= 1
    upper = max(1, n_samples - 1)
    return max(2, min(requested, upper))


def project_hidden_states(
    hidden_states: List[np.ndarray],
    method: Literal["pca", "umap"] = "umap",
    n_components: int = 2,
    umap_args: Optional[Dict[str, Any]] = None,
) -> np.ndarray:
    """
    Reduce high-dimensional hidden states to low-dimensional coordinates.

    Parameters
    ----------
    hidden_states : List[np.ndarray]
        List of hidden state vectors per token (e.g., length N, each shape (H,)).
    method : {"pca","umap"}
        Dimensionality reduction method. Default "umap".
    n_components : int
        Number of output dimensions (default: 2).
    umap_args : dict, optional
        Extra args for UMAP. Defaults include:
          - n_neighbors: adaptively min(10, N-1) with floor=2
          - min_dist: 0.1
          - metric: "cosine"
          - random_state: 42 (deterministic; set to None to favor parallelism)

    Returns
    -------
    np.ndarray
        Array of shape (N, n_components) with reduced coordinates.
    """
    if not hidden_states:
        raise ValueError("Empty hidden state list provided.")

    matrix = np.stack(hidden_states)  # shape: (N, H)
    N, H = matrix.shape

    if method == "pca":
        return _pca_project(matrix, n_components=n_components)

    elif method == "umap":
        # PCA fallback for tiny sequences — avoids impossible neighbor constraints
        if N < 3:
            return _pca_project(matrix, n_components=n_components)

        # Defaults with determinism; allow user overrides
        default_args: Dict[str, Any] = {
            "n_neighbors": 10,
            "min_dist": 0.1,
            "metric": "cosine",
            "random_state": 42,  # deterministic by default; set None to enable parallelism
        }
        args = {**default_args, **(umap_args or {})}

        # Enforce a safe n_neighbors based on N, regardless of override
        args["n_neighbors"] = _safe_n_neighbors(int(args.get("n_neighbors", 10)), N)

        reducer = umap.UMAP(n_components=n_components, **args)
        projected = reducer.fit_transform(matrix)
        return projected

    else:
        raise ValueError(f"Unsupported projection method: {method}")
