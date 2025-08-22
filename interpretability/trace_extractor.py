# interpretability/trace_extractor.py

import json
from typing import List, Dict, Any

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM


def _ensure_hidden_states(outputs) -> List[torch.Tensor]:
    """
    Validate / extract hidden_states from a HF model output, raising a clear error if absent.
    """
    hs = getattr(outputs, "hidden_states", None)
    if hs is None:
        raise RuntimeError(
            "Model forward did not return hidden_states. "
            "Ensure you pass output_hidden_states=True on the forward call."
        )
    return hs


def extract_trace(prompt: str, model, tokenizer, top_k: int = 5) -> Dict[str, Any]:
    """
    Perform a full model pass and extract a trace:
      - token strings and IDs
      - per-token top-k predictions
      - entropy of the token distribution
      - full layerwise hidden states for each token

    Returns:
      dict with per-token diagnostic data.
    """
    device = next(model.parameters()).device
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        # Request hidden states explicitly on each forward (robust across models/configs)
        outputs = model(**inputs, output_hidden_states=True, return_dict=True)

    hidden_states = _ensure_hidden_states(outputs)  # (num_layers+1, batch, seq_len, hidden_size)
    logits = outputs.logits  # (batch, seq_len, vocab_size)
    input_ids = inputs["input_ids"][0]
    tokens = tokenizer.convert_ids_to_tokens(input_ids)

    trace: List[Dict[str, Any]] = []

    seq_len = input_ids.shape[0]
    for i in range(seq_len):
        token = tokens[i]
        token_id = input_ids[i].item()

        # Use float32 for numerical stability even if model/logits are fp16
        token_logits = logits[0, i].float()  # (vocab_size,)
        probs = F.softmax(token_logits, dim=-1)

        k = min(top_k, probs.shape[-1])
        topk = torch.topk(probs, k=k)
        topk_indices = topk.indices.tolist()
        topk_tokens = tokenizer.convert_ids_to_tokens(topk_indices)
        topk_probs = topk.values.tolist()

        # Entropy in nats
        entropy = float(-(probs * (probs + 1e-8).log()).sum().item())

        # Collect per-layer hidden state for this token (convert to Python lists)
        # hidden_states is a tuple/list of tensors with shape (batch, seq_len, hidden_dim)
        token_hidden_states = [layer[0, i].detach().cpu().numpy().tolist() for layer in hidden_states]

        trace.append(
            {
                "index": i,
                "token": token,
                "token_id": token_id,
                "top_k_tokens": topk_tokens,
                "top_k_probs": topk_probs,
                "entropy": entropy,
                "hidden_states": token_hidden_states,  # list of vectors per layer
            }
        )

    return {"prompt": prompt, "trace": trace}


def extract_hidden_state_sequence(prompt: str, model, tokenizer, layer_idx: int = -1) -> List[np.ndarray]:
    """
    Extract token-wise hidden states from a specific model layer.
    Returns:
      List[np.ndarray] of length `num_tokens`, each of shape (hidden_dim,)
    """
    device = next(model.parameters()).device
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        # Request hidden states explicitly on each forward (robust across models/configs)
        outputs = model(**inputs, output_hidden_states=True, return_dict=True)

    hidden_states = _ensure_hidden_states(outputs)  # List/Tuple[Tensor]: (layer, batch, seq_len, dim)
    # Support negative layer indices (HF returns num_layers+1 with embeddings at index 0)
    selected = hidden_states[layer_idx]  # shape: (batch, seq_len, dim)
    selected_layer = selected[0]  # (seq_len, dim)

    # Return as List[np.ndarray]
    return [vec.detach().cpu().numpy() for vec in selected_layer]


def save_trace_to_file(trace_data: Dict[str, Any], filepath: str) -> None:
    """
    Save detailed trace data (from extract_trace) to a JSON file.
    """
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(trace_data, f, indent=2)
