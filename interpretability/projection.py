# interpretability/projection.py

import numpy as np
from typing import List, Literal
from sklearn.decomposition import PCA
import umap


def project_hidden_states(
    hidden_states: List[np.ndarray], 
    method: Literal["pca", "umap"] = "umap", 
    n_components: int = 2,
    umap_args: dict = None
) -> np.ndarray:
    """
    Reduces high-dimensional hidden states to low-dimensional coordinates for analysis.

    Parameters:
        hidden_states (List[np.ndarray]): List of hidden state vectors (e.g. 4096D each).
        method (str): Dimensionality reduction method: 'pca' or 'umap'.
        n_components (int): Number of output dimensions. Defaults to 2.
        umap_args (dict): Optional dictionary of arguments for UMAP.

    Returns:
        np.ndarray: Array of shape (N, n_components) with reduced coordinates.
    """
    if not hidden_states:
        raise ValueError("Empty hidden state list provided.")

    matrix = np.stack(hidden_states)

    if method == "pca":
        reducer = PCA(n_components=n_components)
        projected = reducer.fit_transform(matrix)
    elif method == "umap":
        args = umap_args or {"n_neighbors": 10, "min_dist": 0.1, "metric": "cosine"}
        reducer = umap.UMAP(n_components=n_components, **args)
        projected = reducer.fit_transform(matrix)
    else:
        raise ValueError(f"Unsupported projection method: {method}")

    return projected
