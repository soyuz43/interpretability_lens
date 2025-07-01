# experiments/concept_decay.py

from interpretability.trace_extractor import extract_trace
from interpretability.projection import project_hidden_states
from interpretability.divergence_tracker import compute_semantic_drift, measure_half_life
from interpretability.injector import inject_probe
from interpretability import utils

import matplotlib.pyplot as plt
import numpy as np

def run_concept_decay_experiment(
    base_prompt: str,
    concept: str,
    inject_pos: str,
    model,
    tokenizer,
    visualize: bool = True
) -> dict:
    """
    Run a full concept decay experiment:
      1. Inject a concept into the prompt.
      2. Extract hidden states during inference.
      3. Project hidden states to 2D.
      4. Analyze semantic drift.
      5. Return metrics + optional visualization.
    """
    # Step 1: Construct probe-injected prompt
    full_prompt = inject_probe(base_prompt, concept, position=inject_pos)

    # Step 2: Run model and capture hidden state trajectory
    hidden_states = extract_trace(full_prompt, model, tokenizer)

    # Step 3: Project to low-dimensional space for analysis
    coords = project_hidden_states(hidden_states, method="umap")

    # Step 4: Measure semantic drift and half-life
    drift_metrics = compute_semantic_drift(coords)
    half_life = measure_half_life(drift_metrics["distance_from_origin"])

    results = {
        "arc_length": drift_metrics["arc_length"],
        "half_life": half_life,
        "concept": concept,
        "injected_prompt": full_prompt,
        "drift_curve": drift_metrics["distance_from_origin"]
    }

    # Step 5: Optional visualization
    if visualize:
        plt.figure(figsize=(10, 5))
        plt.plot(drift_metrics["distance_from_origin"], label="Distance from origin")
        plt.axhline(
            y=0.5 * drift_metrics["distance_from_origin"][0],
            color="r", linestyle="--", label="Half-life threshold"
        )
        plt.axvline(x=half_life, color="g", linestyle=":", label="Half-life index")
        plt.title(f"Concept Decay: '{concept}' Influence Over Tokens")
        plt.xlabel("Token Index")
        plt.ylabel("Cosine Distance from Start")
        plt.legend()
        plt.tight_layout()
        plt.show()

    return results
