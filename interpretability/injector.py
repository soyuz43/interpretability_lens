# interpretability/injector.py

from typing import List

def inject_probe(base_prompt: str, concept: str, position: str = "middle") -> str:
    """
    Inserts a concept into the base_prompt at the desired position.
    
    Parameters:
        base_prompt (str): The original prompt or sentence.
        concept (str): The concept word or phrase to inject.
        position (str): Where to inject it — 'start', 'middle', or 'end'.

    Returns:
        str: The modified prompt with the concept inserted.
    """
    if concept is None or str(concept).strip() == "":
        return base_prompt
    tokens = base_prompt.strip().split()

    if position == "start":
        modified = f"{concept} " + base_prompt
    elif position == "middle":
        midpoint = len(tokens) // 2
        modified = " ".join(tokens[:midpoint] + [concept] + tokens[midpoint:])
    elif position == "end":
        modified = base_prompt + f" {concept}"
    else:
        raise ValueError("Invalid position. Choose from 'start', 'middle', or 'end'.")

    return modified


def batch_generate_prompts(base_prompts: List[str], concept: str, positions: List[str]) -> List[str]:
    """
    Automates generation of multiple prompt variants using the same concept.

    Parameters:
        base_prompts (List[str]): List of neutral or scaffold prompts.
        concept (str): The concept to inject.
        positions (List[str]): List of positions to test: ['start', 'middle', 'end']

    Returns:
        List[str]: All generated prompt variants.
    """
    generated = []
    for prompt in base_prompts:
        for pos in positions:
            variant = inject_probe(prompt, concept, pos)
            generated.append(variant)
    return generated
