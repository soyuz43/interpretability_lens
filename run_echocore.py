# run_echocore.py

import argparse
import torch

from transformers import AutoModelForCausalLM, AutoTokenizer

from echocore.controller import EchoCoreController, EchoCoreConfig


def load_llama3_3b():
    """
    Same loader pattern as run_experiment.py, but shared here for EchoCore.
    If you want to avoid duplication, you can move this into a shared module.
    """
    model_id = "meta-llama/Llama-3.2-3B"

    print(f"[✓] Loading tokenizer: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    print(f"[✓] Loading model: {model_id}")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map="auto",
        pad_token_id=tokenizer.pad_token_id,
    )
    model.eval()
    return model, tokenizer


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run EchoCore recursive reasoning on LLaMA 3.2 3B.",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    parser.add_argument(
        "--prompt",
        type=str,
        required=True,
        help="User prompt/question to feed into EchoCore."
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=3,
        help="Number of EchoCore refinement steps (default: 3)."
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=128,
        help="Max new tokens per EchoCore generation call (default: 128)."
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature for EchoCore (default: 0.0 = greedy)."
    )
    parser.add_argument(
        "--show_trace",
        action="store_true",
        help="Print internal EchoCore reasoning steps."
    )

    return parser.parse_args()


def main():
    args = parse_args()

    print("[+] Loading LLaMA 3.2 3B for EchoCore...")
    model, tokenizer = load_llama3_3b()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = EchoCoreConfig(
        steps=args.steps,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        device=device,
    )

    controller = EchoCoreController(model, tokenizer, cfg)

    print("\n=== EchoCore: Running Recursive Reasoning ===")
    print(f"Prompt: {args.prompt}\n")

    final_answer, trace = controller.run(args.prompt)

    print("=== FINAL ANSWER ===")
    print(final_answer.strip())

    if args.show_trace:
        print("\n=== INTERNAL ECHOCORE TRACE ===")
        for i, (p, o) in enumerate(zip(trace.step_prompts, trace.step_outputs), start=1):
            print(f"\n--- Step {i} Prompt ---")
            print(p)
            print(f"\n--- Step {i} Output ---")
            print(o)

    print("\n[✓] EchoCore run complete.")


if __name__ == "__main__":
    main()
