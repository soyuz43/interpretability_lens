# run_experiment.py

from interpretability.injector import inject_probe
from experiments.concept_decay import run_concept_decay_experiment

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import argparse

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

def main():
    parser = argparse.ArgumentParser(description="Run interpretability experiment.")
    parser.add_argument("--prompt", type=str, required=True, help="Base prompt to analyze")
    parser.add_argument("--concept", type=str, help="Concept to inject (optional)")
    parser.add_argument("--inject_pos", type=str, default="middle", help="Where to inject the concept")
    parser.add_argument("--no_visual", action="store_true", help="Disable plot visualization")

    args = parser.parse_args()

    # Step 0: Load model
    print("Loading LLaMA 3B model...")
    model, tokenizer = load_llama3_3b()

    # Step 1: Run experiment
    results = run_concept_decay_experiment(
        base_prompt=args.prompt,
        concept=args.concept or "",
        inject_pos=args.inject_pos,
        model=model,
        tokenizer=tokenizer,
        visualize=not args.no_visual
    )

    # Step 2: Print metrics
    print(f"\n--- Semantic Drift Results ---")
    print("Concept Injected:", results["concept"])
    print("Arc Length:", round(results["arc_length"], 4))
    print("Half-life (token index):", results["half_life"])
    print("Injected Prompt:\n", results["injected_prompt"])

if __name__ == "__main__":
    main()
