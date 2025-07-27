# Interpretability Lens

A toolkit for analyzing the internal dynamics of Large Language Models (LLMs), focusing on how specific concepts influence the model's hidden state trajectory during text processing.

## Overview

This project provides a pipeline to investigate "concept decay" within LLMs. By injecting a concept word or phrase into a base prompt and analyzing the sequence of hidden states produced by the model as it processes that prompt, we can quantify how the influence of the injected concept evolves and fades relative to the initial context.

Key metrics analyzed:
*   **Arc Length:** The total "distance" traveled by the hidden state trajectory in a reduced dimensional space. Measures overall change.
*   **Half-life:** The token position at which the model's hidden state distance from the initial state (defined by the first token) drops to 50% of its initial peak value. Measures persistence of influence.
*   **Drift Curve:** The sequence of distances from the initial state for each token position.

## Project Structure

*   `run_experiment.py`: Main script to run concept decay experiments.
*   `experiments/concept_decay.py`: Core logic for running the experiment, analysis, and logging.
*   `interpretability/`: Package containing utility modules:
    *   `injector.py`: Injects concepts into prompts.
    *   `trace_extractor.py`: Extracts hidden states from a model for a given prompt.
    *   `projection.py`: Reduces high-dimensional hidden states (e.g., using UMAP).
    *   `divergence_tracker.py`: Calculates semantic drift metrics (arc length, half-life, drift curve).
*   `requirements.txt`: Lists Python dependencies.
*   `logs/`: (Created during runtime) Stores experiment results (JSON) and visualizations (PNG).

## Installation

1.  **Clone the Repository:**
    ```bash
    git clone <repository_url>
    cd interpretability_lens
    ```
2.  **Create a Virtual Environment (Recommended):**
    ```bash
    python -m venv .venv
    source .venv/Scripts/activate # On Windows (Git Bash)
    # .venv\Scripts\activate.bat  # On Windows (Command Prompt)
    # source .venv/bin/activate   # On macOS/Linux
    ```
3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *Note:* If you encounter permission errors, try `pip install --user -r requirements.txt`.
4.  **Install Optional Dependency (for `device_map="auto"`):**
    ```bash
    pip install accelerate
    ```
    Add `accelerate` to `requirements.txt` if needed permanently.
5.  **(Windows) Hugging Face Cache Warning:** You might see a warning about symlinks not being supported for the Hugging Face cache. While functional, enabling Developer Mode can improve cache efficiency. See [Hugging Face Cache Limitations](https://huggingface.co/docs/huggingface_hub/how-to-cache#limitations).

## Usage

### Running a Single Experiment

Use `run_experiment.py` to run an experiment. You must provide a `--prompt`. Optionally inject a `--concept` at a specific `--inject_pos`.

```bash
python run_experiment.py --prompt "The weather today is nice and sunny." --concept "happiness" --inject_pos "middle"
```

**Arguments:**
*   `--prompt PROMPT`: (Required) The base prompt for the model.
*   `--concept CONCEPT`: (Optional) The concept word/phrase to inject.
*   `--inject_pos {start,middle,end}`: (Default: `middle`) Where to inject the concept.
*   `--layer LAYER`: (Default: `-1`) The model layer to analyze (e.g., `-1` for the last layer).
*   `--no_visual`: Disable the plot visualization window.
*   `--no_save`: Disable saving results to the `logs` directory.

### Running Batch Experiments

The script supports running multiple related experiments defined by lists of concepts and/or injection positions.

```bash
# Compare injection positions for a single concept
python run_experiment.py --prompt "The weather today is nice and sunny." --concept "happiness" --inject_pos start --inject_pos middle --inject_pos end --batch_name "happiness_position_comparison"

# Compare different concepts at the same position
python run_experiment.py --prompt "Write a short paragraph about everyday life." --concept "technology" --concept "nature" --concept "nostalgia" --inject_pos middle --batch_name "concept_type_comparison"

# Baseline drift (no concept injected) at different effective "positions"
python run_experiment.py --prompt "The cat sat on the mat." --inject_pos start --inject_pos middle --inject_pos end --batch_name "baseline_drift"
```

**Batch Arguments:**
*   `--concept CONCEPT`: Can be used multiple times to specify a list of concepts.
*   `--inject_pos {start,middle,end}`: Can be used multiple times to specify a list of injection positions.
*   `--batch_name BATCH_NAME`: (Optional) Creates a subdirectory in `logs/` to organize results from this batch run.

The script will run experiments for the Cartesian product of the provided concepts and positions. If `--concept` is omitted, it runs without injecting a concept.

### Model Access

The pipeline currently uses the **LLaMA 3.2 3B** model (`meta-llama/Llama-3.2-3B`).

1.  Ensure you have access to the model on the [Hugging Face Hub](https://huggingface.co/meta-llama/Llama-3.2-3B) and have accepted the license terms.
2.  Log in to your Hugging Face account using the CLI:
    ```bash
    huggingface-cli login
    ```

### Output

*   **Console:** Displays experiment parameters and results (`Arc Length`, `Half-life`, `Injected Prompt`).
*   **Plot Window:** Shows the `drift_curve` (Cosine Distance from the first token's hidden state) over token positions, including half-life markers. (Unless `--no_visual` is used).
*   **Logs:** (Unless `--no_save` is used)
    *   Results are saved in `logs/` (or `logs/<batch_name>/` if `--batch_name` is provided).
    *   Each experiment generates:
        *   A `concept_decay_YYYYMMDD_HHMMSS.json` file containing numerical results.
        *   A `concept_decay_YYYYMMDD_HHMMSS.png` file containing the drift plot.

## How It Works

1.  **Prompt Preparation:** A base prompt is taken, and if specified, a concept is injected at the desired position.
2.  **Model Inference:** The finalized prompt is passed to the LLM (e.g., LLaMA 3B). The model performs a forward pass, calculating the sequence of hidden states for each token in the *input prompt*.
3.  **Hidden State Extraction:** Hidden states from a specified layer are extracted for all tokens in the prompt.
4.  **Dimensionality Reduction:** The high-dimensional hidden states are projected into a 2D space (using UMAP by default) to facilitate analysis.
5.  **Drift Analysis:**
    *   The "distance from origin" is calculated as the Cosine Distance between the hidden state of each token and the hidden state of the *first* token in the prompt.
    *   The `arc_length` is computed as the total path length of the 2D trajectory.
    *   The `half_life` is determined as the token index where the "distance from origin" first drops to 50% of its *initial* value (distance at the second token).
6.  **Visualization & Logging:** The `drift_curve` is plotted, and results are saved.

*Note:* This pipeline analyzes the hidden states *as the model processes a fixed input prompt*. It does not involve the model generating new tokens beyond that prompt.

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

