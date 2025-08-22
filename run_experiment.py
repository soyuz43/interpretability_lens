# run_experiment.py

from interpretability.injector import inject_probe
from experiments.concept_decay import run_concept_decay_experiment

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import argparse
import itertools

def load_llama3_3b():
    """Load the LLaMA 3.2 3B model and tokenizer."""
    model_id = "meta-llama/Llama-3.2-3B" # Ensure this is the correct ID
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    # Ensure pad_token_id is set for batch processing if needed later
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map="auto",
        # Add pad_token_id if it was missing
        pad_token_id=tokenizer.pad_token_id
    )
    model.eval()
    return model, tokenizer

def main():
    parser = argparse.ArgumentParser(
        description="Run concept decay experiment(s).",
        formatter_class=argparse.RawTextHelpFormatter # For better formatting of help text
    )
    parser.add_argument("--prompt", type=str, required=True, help="Base prompt for the model.")
    
    # Allow multiple concepts and injection positions using 'append' action
    parser.add_argument("--concept", type=str, action='append', 
                        help="Concept(s) to inject. Can be used multiple times.\n"
                             "Example: --concept 'happiness' --concept 'anger'\n"
                             "If omitted, runs once without injecting a concept.")
    
    parser.add_argument("--inject_pos", type=str, action='append', 
                        choices=["start", "middle", "end"], 
                        help="Position(s) to inject the concept. Can be used multiple times.\n"
                             "Choices: 'start', 'middle', 'end'. Default: 'middle'.\n"
                             "Example: --inject_pos 'start' --inject_pos 'end'")
    
    parser.add_argument("--layer", type=int, default=-1, 
                        help="Model layer to analyze (default: -1 for last layer).")
    
    parser.add_argument("--no_visual", action="store_true", 
                        help="Disable plot visualization for all experiments.")
    
    parser.add_argument("--no_save", action="store_true", 
                        help="Disable saving results (JSON/PNG) to logs for all experiments.")
    
    parser.add_argument("--batch_name", type=str, 
                        help="Optional name for the batch run. Creates a subdirectory in logs/\n"
                             "for all results from this execution.")

    args = parser.parse_args()

    # Handle defaults for lists
    concepts = args.concept if args.concept else [None] # If no concept provided, run once with None
    inject_positions = args.inject_pos if args.inject_pos else ["middle"] # Default to 'middle' if not specified

    # Load model once
    print("Loading LLaMA 3B model...")
    model, tokenizer = load_llama3_3b()

    # --- Batch Execution Logic ---
    # Generate the list of experiments to run based on provided arguments.
    # This handles various combinations of concepts and positions.
    
    experiments_to_run = []
    
    # Case 1: Explicit concepts provided via --concept
    if any(c is not None for c in concepts):
        # Run for every combination of provided concept and position
        experiments_to_run = list(itertools.product(concepts, inject_positions))
    else:
        # Case 2: No concept explicitly provided (--concept flag not used)
        # Run once for each specified injection position with no concept (None)
        experiments_to_run = list(itertools.product([None], inject_positions))

    # Ensure we have at least one experiment to run
    if not experiments_to_run:
        # This handles the edge case where --concept is used with an empty list somehow,
        # though 'append' action usually prevents that. Fallback to default.
        experiments_to_run = [(None, "middle")]

    total_experiments = len(experiments_to_run)
    print(f"\n--- Running {total_experiments} Experiment{'s' if total_experiments > 1 else ''} ---")
    
    # Run each experiment
    for i, (concept, inject_pos) in enumerate(experiments_to_run):
        print(f"\n--- Experiment {i+1}/{total_experiments} ---")
        print(f"Prompt: {args.prompt}")
        print(f"Concept: {concept}")
        print(f"Inject Position: {inject_pos}")
        print(f"Layer: {args.layer}")
        print("-" * 30)

        # Pass the batch_name to the experiment function for organized saving
        results = run_concept_decay_experiment(
            base_prompt=args.prompt,
            concept=concept,
            inject_pos=inject_pos,
            model=model,
            tokenizer=tokenizer,
            visualize=not args.no_visual,
            save=not args.no_save,
            layer_idx=args.layer,
            batch_name=args.batch_name # Pass the batch name for directory organization
        )

        # Optional: Print a brief summary per experiment if desired
        # print(f"Results: Arc Length={results['arc_length']:.4f}, Half-life={results['half_life']}")

    print(f"\n--- All {total_experiments} Experiment{'s' if total_experiments > 1 else ''} Completed ---")
    if not args.no_save and args.batch_name:
        print(f"Results saved to: logs/{args.batch_name}/")
    elif not args.no_save:
        print(f"Results saved to: logs/")

if __name__ == "__main__":
    main()