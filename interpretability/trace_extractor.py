# interpretability/trace_extractor.py

import torch
import json
import numpy as np
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM


def load_model_and_tokenizer(model_name="meta-llama/Llama-2-7b-hf"):
    """
    Load a transformer model and tokenizer with hidden state outputs enabled.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        output_hidden_states=True,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    return model, tokenizer


def extract_trace(prompt, model, tokenizer, top_k=5):
    """
    Perform a full model pass and extract a trace:
    - token strings and IDs
    - per-token top-k predictions
    - entropy of the token distribution
    - full layerwise hidden states for each token
    """
    device = next(model.parameters()).device
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    hidden_states = outputs.hidden_states  # (num_layers + 1, batch, seq_len, hidden_size)
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

        entropy = -torch.sum(probs * probs.log()).item()
        token_hidden_states = [layer[0, i].cpu().numpy().tolist() for layer in hidden_states]

        trace.append({
            "index": i,
            "token": token,
            "token_id": token_id.item(),
            "top_k_tokens": topk_tokens,
            "top_k_probs": topk_probs,
            "entropy": entropy,
            "hidden_states": token_hidden_states
        })

    return {
        "prompt": prompt,
        "trace": trace
    }


def save_trace_to_file(trace_data, filepath):
    """
    Save trace data as a JSON file.
    """
    with open(filepath, "w") as f:
        json.dump(trace_data, f, indent=2)
