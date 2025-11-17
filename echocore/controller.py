# echocore/controller.py

import torch
from dataclasses import dataclass, field
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass
class EchoCoreConfig:
    """
    Configuration for the EchoCore controller.
    """
    steps: int = 3                       # Number of recursive refinement steps
    max_context: int = 2048             # Max tokens for the input prompt
    max_new_tokens: int = 128
    temperature: float = 0.0            # Force deterministic introspection
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class EchoCoreTrace:
    """
    Stores intermediate traces for analysis.
    """
    initial_prompt: str
    step_prompts: list = field(default_factory=list)
    step_outputs: list = field(default_factory=list)

    def add(self, prompt, output):
        self.step_prompts.append(prompt)
        self.step_outputs.append(output)


class EchoCoreController:
    """
    A bolt-on recursive cognition wrapper for any HuggingFace causal LM.
    """

    def __init__(self, model, tokenizer, config: EchoCoreConfig = EchoCoreConfig()):
        self.model = model
        self.tokenizer = tokenizer
        self.cfg = config

    # ------------------------------------------------------------------
    #  TOKENIZE + GENERATE
    # ------------------------------------------------------------------

    def _generate(self, prompt: str) -> str:
        """
        Runs the model on the prompt and returns the decoded output.
        """
        encoded = self.tokenizer(prompt, return_tensors="pt").to(self.cfg.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **encoded,
                max_new_tokens=self.cfg.max_new_tokens,
                temperature=self.cfg.temperature,
                do_sample=self.cfg.temperature > 0,
                pad_token_id=self.tokenizer.eos_token_id
            )

        full_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return full_text[len(prompt):].strip()  # return only the generated portion

    # ------------------------------------------------------------------
    #  ECHoCORE RECURSIVE LOOP
    # ------------------------------------------------------------------

    def run(self, user_prompt: str):
        """
        Runs the multi-step recursive EchoCore loop.
        """
        trace = EchoCoreTrace(initial_prompt=user_prompt)

        # Step 0 — seed the loop
        system_instruction = (
            "You are EchoCore — a reflective, self-correcting reasoning module.\n"
            "Your job is to:\n"
            "1. Analyze the prompt.\n"
            "2. Produce a structured 'internal draft'.\n"
            "3. Critique that draft.\n"
            "4. Improve it.\n\n"
            "Do NOT answer directly. Produce draft → critique → revision."
        )

        current_prompt = (
            system_instruction
            + "\n\n"
            + f"User prompt:\n{user_prompt}\n\n"
            + "Step 1 — Internal Draft:"
        )

        # ------------------------------------------------------------
        #  MULTI-STEP LOOP
        # ------------------------------------------------------------

        for step in range(self.cfg.steps):
            # Generate incremental output
            output = self._generate(current_prompt)

            trace.add(prompt=current_prompt, output=output)

            # Build next step's prompt for refinement
            current_prompt = (
                f"{system_instruction}\n\n"
                f"User prompt:\n{user_prompt}\n\n"
                f"Previous step output:\n{output}\n\n"
                f"Step {step + 2} — Critique and refine the previous reasoning.\n"
                f"Rewrite as a more accurate, coherent, and deeply reasoned draft:"
            )

        # ------------------------------------------------------------
        #  FINAL ANSWER: Summarize recursively-gathered reasoning
        # ------------------------------------------------------------

        final_prompt = (
            "You are EchoCore. Synthesize all prior drafts and critiques into a single, "
            "coherent, deeply reasoned final answer. Do NOT show internal steps.\n\n"
            f"User prompt:\n{user_prompt}\n\n"
            "Internal reasoning traces:\n"
        )

        # Attach all steps (but only for the model)
        for i, out in enumerate(trace.step_outputs):
            final_prompt += f"[Step {i+1}]\n{out}\n\n"

        final_answer = self._generate(final_prompt)

        return final_answer, trace
