# run_experiment.py

from interpretability.trace_extractor import extract_trace
from interpretability.projection import project_hidden_states
from interpretability.divergence_tracker import compute_semantic_drift, measure_half_life
from interpretability.injector import inject_probe
from interpretability import utils

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import argparse
import matplotlib.pyplot as plt
import numpy as np

def load_llama3_3b():
    model_id = "meta-llama/Llama-3.2-3B"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        output_hidden_states=True,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    model.eval()
    return model, tokenizer

def run_pipeline(prompt: str, model, tokenizer):
    # Step 1: Extract hidden states
    hidden_states = extract_trace(prompt, model, tokenizer)

    # Step 2: Reduce dimensionality
    coords = project_hidden_states(hidden_states, method="umap")

    # Step 3: Analyze drift
    drift_metrics = compute_semantic_drift(coords)
    half_life = measure_half_life(drift_metrics["distance_from_origin"])

    # Step 4: Print metrics
    print(f"\n--- Semantic Drift Analysis ---")
    print("Arc length:", round(drift_metrics["arc_length"], 4))
    print("Half-life index (in tokens):", half_life)

    # Step 5: Visualize drift
    plt.figure(figsize=(10, 5))
    plt.plot(drift_metrics["distance_from_origin"], label="Distance from origin")
    plt.axhline(y=0.5 * drift_metrics["distance_from_origin"][0], color='r', linestyle='--', label="Half-life threshold")
    plt.axvline(x=half_life, color='g', linestyle=':', label="Half-life index")
    plt.legend()
    plt.title("Semantic Drift Over Time")
    plt.xlabel("Token Index")
    plt.ylabel("Cosine Distance")
    plt.tight_layout()
    plt.show()

def main():
    parser = argparse.ArgumentParser(description="Run interpretability experiment.")
    parser.add_argument("--prompt", type=str, required=True, help="Base prompt to analyze")
    parser.add_argument("--concept", type=str, help="Concept to inject (optional)")
    parser.add_argument("--inject_pos", type=str, default="middle", help="Where to inject the concept")

    args = parser.parse_args()

    # Step 0: Load model
    print("Loading LLaMA 3B model...")
    model, tokenizer = load_llama3_3b()

    # Step 1: Prepare prompt
    full_prompt = inject_probe(args.prompt, args.concept, args.inject_pos) if args.concept else args.prompt

    # Step 2: Run interpretability pipeline
    run_pipeline(full_prompt, model, tokenizer)

if __name__ == "__main__":
    main()
