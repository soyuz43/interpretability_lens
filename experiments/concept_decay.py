# experiments/concept_decay.py
import os
import json
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
# Import the function to extract hidden states for a specific layer
from interpretability.trace_extractor import extract_hidden_state_sequence
# Import other necessary functions
from interpretability.projection import project_hidden_states
from interpretability.divergence_tracker import compute_semantic_drift, measure_half_life
from interpretability.injector import inject_probe

def log_results(results: dict, folder: str = "logs", fig: plt.Figure = None) -> str:
    """
    Save results dictionary to a timestamped JSON file in the logs directory.
    Optionally save a matplotlib figure to a PNG file with the same timestamp.
    """
    os.makedirs(folder, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_filename = os.path.join(folder, f"concept_decay_{timestamp}.json")

    # Save JSON results
    with open(json_filename, "w") as f:
        json.dump(results, f, indent=2)

    # Save plot if provided
    if fig is not None:
        png_filename = os.path.join(folder, f"concept_decay_{timestamp}.png")
        fig.savefig(png_filename, dpi=300, bbox_inches='tight') # Save with high DPI and tight layout
        print(f"[✓] Plot saved to: {png_filename}")

    return json_filename # Still return the JSON filename as before

def run_concept_decay_experiment(
    base_prompt: str,
    concept: str,
    inject_pos: str,
    model,
    tokenizer,
    visualize: bool = True,
    save: bool = True,
    layer_idx: int = -1 # Add layer_idx parameter
) -> dict:
    """
    Run a full concept decay experiment:
      1. Inject a concept into the prompt.
      2. Extract hidden states during inference.
      3. Project hidden states to 2D.
      4. Analyze semantic drift.
      5. Return metrics, optionally visualize and log output.
    Parameters:
    - base_prompt: str – initial neutral prompt
    - concept: str – concept word/phrase to inject
    - inject_pos: str – 'start', 'middle', or 'end'
    - model, tokenizer – HuggingFace-compatible model objects
    - visualize: bool – whether to show the drift curve
    - save: bool – whether to save the results JSON and plot to disk
    - layer_idx: int - Model layer to extract from (default: last layer)
    Returns:
    - results: dict – containing arc length, half-life, drift curve, and prompt metadata
    """
    # Step 1: Construct probe-injected prompt
    full_prompt = inject_probe(base_prompt, concept, position=inject_pos)
    
    # Step 2: Run model and capture hidden state trajectory
    # Use the specific layer extraction function and pass layer_idx
    hidden_states = extract_hidden_state_sequence(full_prompt, model, tokenizer, layer_idx=layer_idx)
    
    # Step 3: Project to low-dimensional space for analysis
    coords = project_hidden_states(hidden_states, method="umap")
    
    # Step 4: Measure semantic drift and half-life
    drift_metrics = compute_semantic_drift(coords)
    half_life = measure_half_life(drift_metrics["distance_from_origin"])
    
    # Ensure all values are JSON serializable
    results = {
        "concept": concept,
        "inject_position": inject_pos,
        "injected_prompt": full_prompt,
        "arc_length": float(drift_metrics["arc_length"]), # Convert to Python float
        "half_life": int(half_life),                     # Convert to Python int
        "drift_curve": [float(val) for val in drift_metrics["distance_from_origin"]], # Convert list of np.float32 to list of Python floats
    }

    fig = None # Initialize figure variable
    
    # Step 5: Visualization
    # Create the plot if we need to show it OR save it
    if visualize or save: 
        fig = plt.figure(figsize=(10, 5)) # Assign the figure object
        
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
        # Don't call plt.show() here yet

    # Step 6: Optional logging (including saving the plot)
    if save:
        # Pass the figure object to log_results
        filename = log_results(results, fig=fig) # fig will be saved inside log_results
        print(f"[✓] Results logged to: {filename}")
        # Close the figure after saving
        if fig is not None:
            plt.close(fig)
            fig = None
    elif fig is not None:
         # If not saving but we created a figure (only for visualization), close it
         plt.close(fig)
         fig = None

    # Show the plot only if visualize is True and we created one
    if visualize and fig is not None:
        plt.show()
    elif fig is not None:
        # Make sure to close the figure if it was created but not shown or saved
        plt.close(fig)

    return results