# run_experiment.py

from interpretability.trace_extractor import extract_trace
from interpretability.projection import project_hidden_states
from interpretability.divergence_tracker import compute_semantic_drift, measure_half_life
from interpretability.injector import inject_probe
from interpretability import utils

import argparse
import matplotlib.pyplot as plt
import numpy as np

def main():
    parser = argparse.ArgumentParser(description="Run interpretability experiment.")
    parser.add_argument("--prompt", type=str, required=True, help="Base prompt to analyze")
    parser.add_argument("--concept", type=str, help="Concept to inject (optional)")
    parser.add_argument("--inject_pos", type=str, default="middle", help="Where to inject the concept")

    args = parser.parse_args()

    # Step 1: Prepare prompt (optional probe insertion)
    prompt = inject_probe(args.prompt, args.concept, args.inject_pos) if args.concept else args.prompt

    # Step 2: Extract hidden states
    hidden_states = extract_trace(prompt)

    # Step 3: Dimensionality reduction
    coords = project_hidden_states(hidden_states, method="umap")

    # Step 4: Drift metrics
    drift_metrics = compute_semantic_drift(coords)
    half_life = measure_half_life(drift_metrics["distance_from_origin"])

    # Step 5: Output results
    print("Arc length:", drift_metrics["arc_length"])
    print("Half-life index (in tokens):", half_life)

    # Step 6: Optional: Plot drift
    plt.plot(drift_metrics["distance_from_origin"], label="Distance from origin")
    plt.axhline(y=0.5 * drift_metrics["distance_from_origin"][0], color='r', linestyle='--', label="Half-life threshold")
    plt.axvline(x=half_life, color='g', linestyle=':', label="Half-life index")
    plt.legend()
    plt.title("Semantic Drift Over Time")
    plt.xlabel("Token Index")
    plt.ylabel("Cosine Distance")
    plt.show()


if __name__ == "__main__":
    main()
