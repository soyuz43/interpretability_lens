# experiments/concept_decay.py
import os
import json
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np

# Hidden-state extraction (per token, per chosen layer)
from interpretability.trace_extractor import extract_hidden_state_sequence
# Projection & drift metrics (for visualization/arc length)
from interpretability.projection import project_hidden_states
from interpretability.divergence_tracker import compute_semantic_drift, measure_half_life
# Prompt manipulation
from interpretability.injector import inject_probe
# True metric in original space
from interpretability.utils import cosine_similarity


def log_results(
    results: dict,
    folder: str = "logs",
    fig: plt.Figure = None,
    batch_name: str = None,
) -> str:
    """
    Save results dictionary to a timestamped JSON file in the logs directory.
    Optionally save a matplotlib figure to a PNG file with the same timestamp.
    If batch_name is provided, saves within a subdirectory logs/batch_name/.
    """
    # Determine the final save directory
    final_folder = folder
    if batch_name:
        final_folder = os.path.join(folder, batch_name)

    os.makedirs(final_folder, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_filename = os.path.join(final_folder, f"concept_decay_{timestamp}.json")

    # Save JSON results (ensure serializable types)
    with open(json_filename, "w") as f:
        serializable_results = {
            "concept": results.get("concept"),
            "inject_position": results.get("inject_position"),
            "injected_prompt": results.get("injected_prompt"),
            "arc_length": float(results.get("arc_length", 0.0)),
            "half_life": int(results.get("half_life", -1)),
            # Projection-space drift curve (for visualization trajectory)
            "drift_curve": [float(val) for val in results.get("drift_curve", [])],
            # Original-space cosine distance curve (used for half-life + plotting)
            "cosine_from_start": [float(val) for val in results.get("cosine_from_start", [])],
        }
        json.dump(serializable_results, f, indent=2)

    # Save plot if provided
    if fig is not None:
        png_filename = os.path.join(final_folder, f"concept_decay_{timestamp}.png")
        fig.savefig(png_filename, dpi=300, bbox_inches="tight")
        print(f"[✓] Plot saved to: {png_filename}")

    return json_filename


def _ema(series, alpha: float):
    """Simple EMA smoothing; returns a new list."""
    if not series:
        return []
    y = [series[0]]
    a = float(alpha)
    for v in series[1:]:
        y.append(a * v + (1.0 - a) * y[-1])
    return y


def run_concept_decay_experiment(
    base_prompt: str,
    concept: str,
    inject_pos: str,
    model,
    tokenizer,
    visualize: bool = True,
    save: bool = True,
    layer_idx: int = -1,
    batch_name: str = None,
) -> dict:
    """
    Run a full concept decay experiment:
      1. Inject a concept into the prompt (or no-op if concept is None/empty).
      2. Extract hidden states during inference (chosen layer).
      3. Project hidden states to 2D for visualization; compute drift/arc length.
      4. Compute original-space cosine curve; measure half-life on this curve.
      5. Return metrics and optionally visualize + log output.

    Returns:
      results: dict – arc_length (projection), half_life (original-space cosine),
                       drift_curve (projection), cosine_from_start (original space),
                       and prompt metadata.
    """
    # Step 1: Construct probe-injected prompt (baseline-safe)
    full_prompt = inject_probe(base_prompt, concept, position=inject_pos)

    # Step 2: Run model and capture hidden state trajectory (per-token vectors)
    hidden_states = extract_hidden_state_sequence(
        full_prompt, model, tokenizer, layer_idx=layer_idx
    )  # List[np.ndarray], length = num_tokens

    # Guard: if too few tokens, produce empty curves gracefully
    num_tokens = len(hidden_states)
    # Original-space cosine distance from the first token's hidden state
    if num_tokens >= 2:
        start_vec = hidden_states[0]
        cosine_from_start = [1.0 - cosine_similarity(start_vec, v) for v in hidden_states[1:]]
    else:
        cosine_from_start = []

    # Step 3: Project to low-dimensional space for trajectory visualization
    # (deterministic UMAP via random_state for reproducibility)
    coords = project_hidden_states(
        hidden_states,
        method="umap",
        umap_args={"n_neighbors": 10, "min_dist": 0.1, "metric": "cosine", "random_state": 42},
    )

    # Step 4: Compute drift metrics in projection space, but measure half-life on original-space cosine
    drift_metrics = compute_semantic_drift(coords)
    # Robust half-life (defaults: threshold=0.5, EMA=0.2, k=2, post_peak=True)
    half_life = measure_half_life(cosine_from_start)

    # Prepare results (JSON-serializable)
    results = {
        "concept": concept,
        "inject_position": inject_pos,
        "injected_prompt": full_prompt,
        "arc_length": float(drift_metrics["arc_length"]),
        "half_life": int(half_life),
        "drift_curve": [float(val) for val in drift_metrics["distance_from_origin"]],
        "cosine_from_start": [float(val) for val in cosine_from_start],
    }

    # Step 5: Visualization (plot original-space cosine curve; projection only for arc length)
    fig = None
    if visualize or save:
        fig = plt.figure(figsize=(10, 5))

        # Plot the measurement curve we used for half-life
        plt.plot(cosine_from_start, label="Cosine distance from start (original space)")

        # Draw threshold consistent with measure_half_life defaults
        # (smooth with EMA, reference = post-peak max, threshold = 0.5)
        smooth_ema_alpha = 0.2
        threshold = 0.5
        post_peak = True

        x = list(cosine_from_start)
        if smooth_ema_alpha and smooth_ema_alpha > 0:
            x = _ema(x, smooth_ema_alpha)

        ref = max(x) if x else 0.0
        thresh_val = threshold * ref if x else 0.0
        if x:
            # Optional: could visualize the post-peak start index if desired
            # start_idx = int(np.argmax(x)) if post_peak else 0
            plt.axhline(
                y=thresh_val,
                color="r",
                linestyle="--",
                label="Half-life threshold (cosine)",
            )

        # Half-life index line (indexing is on the curve which starts at token 1)
        if half_life >= 0 and len(cosine_from_start) > 0:
            plt.axvline(x=half_life, color="g", linestyle=":", label="Half-life index")

        title_concept = concept if (concept is not None and str(concept).strip() != "") else "baseline"
        plt.title(f"Concept Decay: '{title_concept}' influence over tokens")
        plt.xlabel("Token Index (starting at 1)")
        plt.ylabel("Cosine distance from start (original space)")
        plt.legend()
        plt.tight_layout()

    # Step 6: Optional logging, visualization, and cleanup (save → show → close)
    if save:
        filename = log_results(results, fig=fig, batch_name=batch_name)
        print(f"[✓] Results logged to: {filename}")

    if visualize and fig is not None:
        plt.show()

    if fig is not None:
        plt.close(fig)
        fig = None

    return results
