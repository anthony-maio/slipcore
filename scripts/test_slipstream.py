"""
Test the finetuned Slipstream model.

Usage:
    python scripts/test_slipstream.py
    python scripts/test_slipstream.py --model ./output/slipstream-merged
    python scripts/test_slipstream.py --interactive
"""

import argparse
from pathlib import Path


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

    # Enable inference mode
    FastLanguageModel.for_inference(model)

    return model, tokenizer


def generate_slip(model, tokenizer, instruction: str, src: str = "agent", dst: str = "other"):
    """Generate a SLIP message for the given instruction."""
    from unsloth.chat_templates import get_chat_template

    tokenizer = get_chat_template(tokenizer, chat_template="chatml")

    messages = [
        {"role": "system", "content": """You are an AI agent that communicates using the Slipstream protocol (SLIP).

Slipstream uses semantic quantization - instead of verbose natural language, you transmit compact messages that reference a shared concept codebook (UCR).

Wire format: SLIP v1 <src> <dst> <anchor> [payload...]

Example:
- To request a code review: SLIP v1 alice bob RequestReview
- To report completion: SLIP v1 worker manager InformComplete task42
- To propose a plan: SLIP v1 planner team ProposePlan auth_refactor

Always respond with SLIP messages when coordinating with other agents."""},
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
        max_new_tokens=256,
        temperature=0.7,
        top_p=0.9,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
    )

    response = tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True)
    return response


def run_tests(model, tokenizer):
    """Run standard test cases."""
    test_cases = [
        # Request tests
        ("Tell bob to review my authentication code", "RequestReview"),
        ("Ask the team to implement the caching feature", "RequestTask"),
        ("Get help from alice with the database issue", "RequestHelp"),
        ("Request a deployment plan from devops", "RequestPlan"),

        # Inform tests
        ("Let the manager know the API refactor is done", "InformComplete"),
        ("Update the team on deployment progress", "InformProgress"),
        ("Report that we're blocked on credentials", "InformBlocked"),

        # Eval tests
        ("Approve bob's pull request", "EvalApprove"),
        ("Reject the proposed architecture change", "EvalReject"),
        ("Tell alice her code needs more error handling", "EvalNeedsWork"),

        # Propose tests
        ("Suggest using Redis for caching to the team", "ProposeAlternative"),
        ("Propose a migration plan to the architect", "ProposePlan"),

        # Meta tests
        ("Acknowledge receipt of the task from manager", "MetaAck"),
        ("Accept alice's request to help", "Accept"),
        ("Decline the meeting invitation from bob", "Reject"),

        # Fallback test (should recognize as unquantizable)
        ("Tell devops to check the kubernetes pod memory limits for the redis-cluster-0 instance", "Fallback"),
    ]

    print("\n" + "=" * 70)
    print("SLIPSTREAM MODEL TESTS")
    print("=" * 70)

    passed = 0
    failed = 0

    for instruction, expected_anchor in test_cases:
        print(f"\n{'─' * 70}")
        print(f"INPUT: {instruction}")
        print(f"EXPECTED: {expected_anchor}")

        response = generate_slip(model, tokenizer, instruction)
        print(f"OUTPUT:\n{response}")

        # Check if expected anchor is in response
        if expected_anchor in response:
            print(f"✓ PASS")
            passed += 1
        else:
            print(f"✗ FAIL (expected {expected_anchor})")
            failed += 1

    print(f"\n{'=' * 70}")
    print(f"RESULTS: {passed}/{passed + failed} passed ({100*passed/(passed+failed):.1f}%)")
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
    parser.add_argument("--model", type=str, default="./output/slipstream-merged",
                        help="Path to the finetuned model")
    parser.add_argument("--interactive", "-i", action="store_true",
                        help="Run in interactive mode")
    parser.add_argument("--lora", type=str, default=None,
                        help="Path to LoRA adapter (loads on top of base model)")

    args = parser.parse_args()

    # Load model
    if args.lora:
        # Load base + LoRA
        from peft import PeftModel
        model, tokenizer = load_model("zai-org/GLM-Z1-9B-0414")
        model = PeftModel.from_pretrained(model, args.lora)
    else:
        model, tokenizer = load_model(args.model)

    if args.interactive:
        interactive_mode(model, tokenizer)
    else:
        run_tests(model, tokenizer)


if __name__ == "__main__":
    main()
