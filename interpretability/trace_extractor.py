# interpretability/trace_extractor.py

import torch
import json
import numpy as np
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

def extract_trace(prompt, model, tokenizer, top_k=5):
    """
    Perform a full model pass and extract a trace:
    - token strings and IDs
    - per-token top-k predictions
    - entropy of the token distribution
    - full layerwise hidden states for each token
    Returns a dict with per-token diagnostic data.
    """
    device = next(model.parameters()).device
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    hidden_states = outputs.hidden_states  # (num_layers+1, batch, seq_len, hidden_size)
    logits = outputs.logits  # (batch, seq_len, vocab_size)
    input_ids = inputs["input_ids"][0]
    tokens = tokenizer.convert_ids_to_tokens(input_ids)

    trace = []

    for i, token_id in enumerate(input_ids):
        token = tokens[i]
        token_logits = logits[0, i]
        probs = F.softmax(token_logits, dim=-1)

        topk = torch.topk(probs, k=top_k)
        topk_indices = topk.indices.tolist()
        topk_tokens = tokenizer.convert_ids_to_tokens(topk_indices)
        topk_probs = topk.values.tolist()

        entropy = -torch.sum(probs * torch.log(probs + 1e-8)).item()

        token_hidden_states = [layer[0, i].cpu().numpy().tolist() for layer in hidden_states]

        trace.append({
            "index": i,
            "token": token,
            "token_id": token_id.item(),
            "top_k_tokens": topk_tokens,
            "top_k_probs": topk_probs,
            "entropy": entropy,
            "hidden_states": token_hidden_states  # list of vectors per layer
        })

    return {
        "prompt": prompt,
        "trace": trace
    }

def extract_hidden_state_sequence(prompt, model, tokenizer, layer_idx: int = -1):
    """
    Extract token-wise hidden states from a specific model layer.
    Returns a List[np.array] of shape [num_tokens x hidden_dim]
    """
    device = next(model.parameters()).device
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    hidden_states = outputs.hidden_states  # List[Tensor]: (layer, batch, seq_len, dim)
    selected_layer = hidden_states[layer_idx][0]  # shape: (seq_len, dim)

    return [vec.cpu().numpy() for vec in selected_layer]

def save_trace_to_file(trace_data, filepath):
    """
    Save detailed trace data (from extract_trace) to a JSON file.
    """
    with open(filepath, "w") as f:
        json.dump(trace_data, f, indent=2)
