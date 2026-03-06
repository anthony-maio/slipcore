"""
Test a finetuned Slipstream model against v3 Force+Object outputs.

Usage:
    python scripts/test_slipstream.py
    python scripts/test_slipstream.py --model ./output/slipstream-merged
    python scripts/test_slipstream.py --interactive
"""

import argparse


def load_model(model_path: str):
    """Load the finetuned model."""
    from unsloth import FastLanguageModel

    print(f"Loading model from: {model_path}")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_path,
        max_seq_length=2048,
        dtype=None,
        load_in_4bit=True,
        trust_remote_code=True,
    )

    FastLanguageModel.for_inference(model)
    return model, tokenizer


def generate_slip(model, tokenizer, instruction: str):
    """Generate a SLIP message for the given instruction."""
    from unsloth.chat_templates import get_chat_template

    tokenizer = get_chat_template(tokenizer, chat_template="chatml")

    messages = [
        {
            "role": "system",
            "content": """You are an AI agent that communicates using Slipstream v3.

Wire format: SLIP v3 <src> <dst> <Force> <Object> [payload...]

Forces: Observe, Inform, Ask, Request, Propose, Commit, Eval, Meta, Accept, Reject, Error, Fallback

Always respond with a valid SLIP v3 wire message.""",
        },
        {"role": "user", "content": instruction},
    ]

    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to(model.device)

    outputs = model.generate(
        input_ids=inputs,
        max_new_tokens=128,
        temperature=0.7,
        top_p=0.9,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
    )

    response = tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True)
    return response.strip()


def run_tests(model, tokenizer):
    """Run standard test cases."""
    test_cases = [
        ("Tell bob to review my authentication code", "Request Review"),
        ("Let the manager know implementation is complete", "Inform Complete"),
        ("Ask if deployment permission is granted", "Ask Permission"),
        ("Suggest a rollback plan", "Propose Rollback"),
        ("Acknowledge receipt of the request", "Meta Ack"),
        ("Report a timeout in the worker", "Error Timeout"),
    ]

    print("\n" + "=" * 70)
    print("SLIPSTREAM MODEL TESTS (v3)")
    print("=" * 70)

    passed = 0
    failed = 0

    for instruction, expected_pair in test_cases:
        print(f"\n{'─' * 70}")
        print(f"INPUT: {instruction}")
        print(f"EXPECTED CONTAINS: {expected_pair}")

        response = generate_slip(model, tokenizer, instruction)
        print(f"OUTPUT:\n{response}")

        if "SLIP v3" in response and expected_pair in response:
            print("✓ PASS")
            passed += 1
        else:
            print(f"✗ FAIL (expected Force/Object pair: {expected_pair})")
            failed += 1

    print(f"\n{'=' * 70}")
    print(f"RESULTS: {passed}/{passed + failed} passed ({100 * passed / (passed + failed):.1f}%)")
    print("=" * 70)

    return passed, failed


def interactive_mode(model, tokenizer):
    """Interactive REPL for testing."""
    print("\n" + "=" * 70)
    print("INTERACTIVE MODE")
    print("Type instructions to generate SLIP messages. Type 'quit' to exit.")
    print("=" * 70)

    while True:
        try:
            instruction = input("\n> ").strip()
            if instruction.lower() in ("quit", "exit", "q"):
                break
            if not instruction:
                continue

            response = generate_slip(model, tokenizer, instruction)
            print(f"\n{response}")

        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {e}")

    print("\nGoodbye!")


def main():
    parser = argparse.ArgumentParser(description="Test the finetuned Slipstream model")
    parser.add_argument("--model", type=str, default="./output/slipstream-merged", help="Path to model")
    parser.add_argument("--interactive", "-i", action="store_true", help="Run in interactive mode")

    args = parser.parse_args()

    model, tokenizer = load_model(args.model)

    if args.interactive:
        interactive_mode(model, tokenizer)
    else:
        run_tests(model, tokenizer)


if __name__ == "__main__":
    main()
