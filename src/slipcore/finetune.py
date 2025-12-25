"""
Slipstream Finetuning Dataset Generator

Generates training data to teach LLMs to communicate using the Slipstream protocol.
Supports multiple output formats:
- Hugging Face datasets (JSONL)
- Unsloth/LoRA training format
- Claude training format

The goal: Train agents to Think-Quantize-Transmit natively.
"""

from __future__ import annotations
import json
import random
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional
from .ucr import get_default_ucr, UCRAnchor, UCR


# ============ System Prompts ============

SYSTEM_PROMPT_BASIC = """You are an AI agent that communicates using the Slipstream protocol (SLIP).

Slipstream uses semantic quantization - instead of verbose natural language, you transmit compact messages that reference a shared concept codebook (UCR).

Wire format: SLIP v1 <src> <dst> <anchor> [payload...]

Example:
- To request a code review: SLIP v1 alice bob RequestReview
- To report completion: SLIP v1 worker manager InformComplete task42
- To propose a plan: SLIP v1 planner team ProposePlan auth_refactor

Always respond with SLIP messages when coordinating with other agents."""

SYSTEM_PROMPT_DETAILED = """You are an AI agent that communicates using the Slipstream protocol (SLIP).

## Protocol Overview
Slipstream achieves 80%+ token savings by transmitting semantic anchors instead of natural language. Each anchor represents a common agent intent.

## Wire Format
SLIP v1 <source> <destination> <anchor> [optional payload...]

## Available Anchors (Common Ones)
- RequestTask: Ask another agent to do something
- RequestReview: Ask for code/plan review
- RequestHelp: Ask for assistance
- ProposePlan: Suggest a plan of action
- ProposeChange: Suggest a modification
- InformComplete: Report task completion
- InformProgress: Share progress update
- InformBlocked: Report being blocked
- EvalApprove: Approve something
- EvalReject: Reject something
- EvalNeedsWork: Request revisions
- Accept: Agree to a request/proposal
- Reject: Decline a request/proposal
- MetaAck: Acknowledge receipt
- Fallback: For unquantizable content (include natural language in payload)

## Examples
User: "Tell the executor to implement the login feature"
You: SLIP v1 planner executor RequestTask implement login feature

User: "Let the team know the auth module is done"
You: SLIP v1 developer team InformComplete auth_module

User: "Ask for feedback on the database schema"
You: SLIP v1 developer reviewer RequestReview database_schema

Always use SLIP format when coordinating with other agents."""


# ============ Training Example Templates ============

@dataclass
class TrainingExample:
    """A single training example for finetuning."""
    instruction: str
    thought: str  # Natural language intent
    response: str  # SLIP wire format
    anchor: str
    src: str = "agent"
    dst: str = "other"


# Template generators for each anchor type
def _generate_request_examples() -> list[TrainingExample]:
    """Generate examples for REQUEST-type anchors."""
    examples = []

    request_templates = [
        # RequestTask
        ("Tell {dst} to implement the {feature} feature",
         "I need {dst} to work on implementing {feature}",
         "RequestTask", "{feature}"),
        ("Ask {dst} to run the test suite",
         "I want {dst} to execute the tests",
         "RequestTask", "run tests"),
        ("Have {dst} deploy to staging",
         "We need {dst} to handle the staging deployment",
         "RequestTask", "deploy staging"),

        # RequestReview
        ("Ask {dst} to review the {module} code",
         "I need feedback on the {module} implementation",
         "RequestReview", "{module}"),
        ("Get {dst} to check the pull request",
         "The PR needs review from {dst}",
         "RequestReview", "pull_request"),
        ("Request a security review from {dst}",
         "I want {dst} to audit the security aspects",
         "RequestReview", "security"),

        # RequestHelp
        ("Ask {dst} for help with the {problem}",
         "I'm stuck on {problem} and need assistance",
         "RequestHelp", "{problem}"),
        ("Get assistance from {dst} on debugging",
         "I need {dst}'s help to debug this issue",
         "RequestHelp", "debugging"),

        # RequestPlan
        ("Ask {dst} to create a plan for {goal}",
         "We need a strategy for {goal}",
         "RequestPlan", "{goal}"),
        ("Have {dst} design the architecture",
         "I need {dst} to plan out the system architecture",
         "RequestPlan", "architecture"),
    ]

    features = ["authentication", "caching", "logging", "API", "database"]
    modules = ["auth", "payment", "user", "api", "core"]
    problems = ["memory leak", "race condition", "timeout", "performance"]
    goals = ["migration", "refactor", "optimization", "scaling"]
    destinations = ["bob", "alice", "reviewer", "planner", "executor", "team"]

    for template, thought_template, anchor, payload_template in request_templates:
        for _ in range(3):  # Generate 3 variations per template
            dst = random.choice(destinations)
            feature = random.choice(features)
            module = random.choice(modules)
            problem = random.choice(problems)
            goal = random.choice(goals)

            instruction = template.format(
                dst=dst, feature=feature, module=module, problem=problem, goal=goal
            )
            thought = thought_template.format(
                dst=dst, feature=feature, module=module, problem=problem, goal=goal
            )
            payload = payload_template.format(
                feature=feature, module=module, problem=problem, goal=goal
            )

            response = f"SLIP v1 agent {dst} {anchor}"
            if payload and payload != "{feature}" and payload != "{module}":
                response += f" {payload.replace(' ', '_')}"

            examples.append(TrainingExample(
                instruction=instruction,
                thought=thought,
                response=response,
                anchor=anchor,
                dst=dst,
            ))

    return examples


def _generate_inform_examples() -> list[TrainingExample]:
    """Generate examples for INFORM-type anchors."""
    examples = []

    inform_templates = [
        # InformComplete
        ("Tell {dst} that the {task} is done",
         "I finished working on {task}",
         "InformComplete", "{task}"),
        ("Let {dst} know the deployment succeeded",
         "The deployment completed successfully",
         "InformComplete", "deployment"),
        ("Report to {dst} that testing is finished",
         "All tests have passed",
         "InformComplete", "testing"),

        # InformProgress
        ("Update {dst} on the {task} progress",
         "I'm making progress on {task}",
         "InformProgress", "{task}"),
        ("Let {dst} know we're 50% done",
         "We've completed half of the work",
         "InformProgress", "halfway"),

        # InformBlocked
        ("Tell {dst} you're blocked on {blocker}",
         "I can't proceed because of {blocker}",
         "InformBlocked", "{blocker}"),
        ("Report to {dst} that you need credentials",
         "I'm waiting for API credentials",
         "InformBlocked", "credentials"),
        ("Let {dst} know there's a dependency issue",
         "Blocked by missing dependencies",
         "InformBlocked", "dependencies"),

        # InformStatus
        ("Give {dst} a status update",
         "Here's where things stand",
         "InformStatus", ""),
        ("Tell {dst} the current state of {task}",
         "Providing update on {task}",
         "InformStatus", "{task}"),
    ]

    tasks = ["auth_module", "api_refactor", "database_migration", "caching", "tests"]
    blockers = ["API_access", "review", "dependencies", "credentials", "resources"]
    destinations = ["manager", "team", "coordinator", "lead"]

    for template, thought_template, anchor, payload_template in inform_templates:
        for _ in range(3):
            dst = random.choice(destinations)
            task = random.choice(tasks)
            blocker = random.choice(blockers)

            instruction = template.format(dst=dst, task=task, blocker=blocker)
            thought = thought_template.format(dst=dst, task=task, blocker=blocker)
            payload = payload_template.format(task=task, blocker=blocker)

            response = f"SLIP v1 agent {dst} {anchor}"
            if payload:
                response += f" {payload.replace(' ', '_')}"

            examples.append(TrainingExample(
                instruction=instruction,
                thought=thought,
                response=response,
                anchor=anchor,
                dst=dst,
            ))

    return examples


def _generate_eval_examples() -> list[TrainingExample]:
    """Generate examples for EVAL-type anchors."""
    examples = []

    eval_templates = [
        # EvalApprove
        ("Approve {dst}'s proposal",
         "The proposal looks good, approving it",
         "EvalApprove", ""),
        ("Give thumbs up to {dst}'s code",
         "The code is well written, approved",
         "EvalApprove", "code"),
        ("Accept the plan from {dst}",
         "The plan is solid, let's go with it",
         "EvalApprove", "plan"),

        # EvalReject
        ("Reject {dst}'s approach",
         "This approach won't work",
         "EvalReject", ""),
        ("Turn down the proposal from {dst}",
         "The proposal has too many issues",
         "EvalReject", "proposal"),

        # EvalNeedsWork
        ("Tell {dst} their code needs changes",
         "There are some issues that need fixing",
         "EvalNeedsWork", ""),
        ("Request revisions from {dst}",
         "Good progress but needs more work",
         "EvalNeedsWork", "revisions"),
        ("Ask {dst} to fix the {issue}",
         "Found an issue that needs addressing",
         "EvalNeedsWork", "{issue}"),
    ]

    issues = ["error_handling", "validation", "security", "performance", "tests"]
    destinations = ["developer", "author", "submitter", "engineer"]

    for template, thought_template, anchor, payload_template in eval_templates:
        for _ in range(3):
            dst = random.choice(destinations)
            issue = random.choice(issues)

            instruction = template.format(dst=dst, issue=issue)
            thought = thought_template.format(dst=dst, issue=issue)
            payload = payload_template.format(issue=issue)

            response = f"SLIP v1 agent {dst} {anchor}"
            if payload:
                response += f" {payload}"

            examples.append(TrainingExample(
                instruction=instruction,
                thought=thought,
                response=response,
                anchor=anchor,
                dst=dst,
            ))

    return examples


def _generate_propose_examples() -> list[TrainingExample]:
    """Generate examples for PROPOSE-type anchors."""
    examples = []

    propose_templates = [
        # ProposePlan
        ("Suggest a plan to {dst} for {goal}",
         "I have an idea for how to handle {goal}",
         "ProposePlan", "{goal}"),
        ("Propose an approach to {dst}",
         "Here's what I think we should do",
         "ProposePlan", ""),
        ("Share your strategy with {dst}",
         "I've thought through a strategy",
         "ProposePlan", "strategy"),

        # ProposeChange
        ("Suggest a change to {dst}",
         "I think we should modify the approach",
         "ProposeChange", ""),
        ("Propose modifying the {component}",
         "The {component} could be improved",
         "ProposeChange", "{component}"),

        # ProposeAlternative
        ("Offer {dst} an alternative approach",
         "There's another way we could do this",
         "ProposeAlternative", ""),
        ("Suggest using {tech} instead",
         "We might be better off with {tech}",
         "ProposeAlternative", "{tech}"),
    ]

    goals = ["refactoring", "migration", "optimization", "scaling", "security"]
    components = ["api", "database", "cache", "auth", "frontend"]
    techs = ["Redis", "PostgreSQL", "GraphQL", "Kubernetes", "async"]
    destinations = ["team", "lead", "architect", "manager"]

    for template, thought_template, anchor, payload_template in propose_templates:
        for _ in range(3):
            dst = random.choice(destinations)
            goal = random.choice(goals)
            component = random.choice(components)
            tech = random.choice(techs)

            instruction = template.format(dst=dst, goal=goal, component=component, tech=tech)
            thought = thought_template.format(dst=dst, goal=goal, component=component, tech=tech)
            payload = payload_template.format(goal=goal, component=component, tech=tech)

            response = f"SLIP v1 agent {dst} {anchor}"
            if payload:
                response += f" {payload}"

            examples.append(TrainingExample(
                instruction=instruction,
                thought=thought,
                response=response,
                anchor=anchor,
                dst=dst,
            ))

    return examples


def _generate_meta_examples() -> list[TrainingExample]:
    """Generate examples for META and Accept/Reject anchors."""
    examples = []

    meta_templates = [
        # Accept
        ("Accept {dst}'s request",
         "Yes, I agree to do this",
         "Accept", ""),
        ("Agree to {dst}'s proposal",
         "The proposal works for me",
         "Accept", ""),
        ("Confirm with {dst}",
         "Confirmed, I'm on it",
         "Accept", ""),

        # Reject
        ("Decline {dst}'s request",
         "I can't do this right now",
         "Reject", ""),
        ("Say no to {dst}",
         "This won't work for me",
         "Reject", ""),

        # MetaAck
        ("Acknowledge {dst}'s message",
         "Got it, understood",
         "MetaAck", ""),
        ("Confirm receipt to {dst}",
         "Message received",
         "MetaAck", ""),

        # MetaHandoff
        ("Hand off to {dst}",
         "Passing this over to you",
         "MetaHandoff", ""),
        ("Transfer responsibility to {dst}",
         "This is now in your hands",
         "MetaHandoff", ""),
    ]

    destinations = ["alice", "bob", "team", "manager", "coordinator"]

    for template, thought_template, anchor, payload in meta_templates:
        for _ in range(2):
            dst = random.choice(destinations)

            instruction = template.format(dst=dst)
            thought = thought_template.format(dst=dst)
            response = f"SLIP v1 agent {dst} {anchor}"
            if payload:
                response += f" {payload}"

            examples.append(TrainingExample(
                instruction=instruction,
                thought=thought,
                response=response,
                anchor=anchor,
                dst=dst,
            ))

    return examples


# ============ Dataset Generator ============

def generate_training_examples(num_examples: int = 500, seed: int = 42) -> list[TrainingExample]:
    """
    Generate a diverse set of training examples.

    Args:
        num_examples: Target number of examples
        seed: Random seed for reproducibility

    Returns:
        List of TrainingExample objects
    """
    random.seed(seed)

    all_examples = []
    all_examples.extend(_generate_request_examples())
    all_examples.extend(_generate_inform_examples())
    all_examples.extend(_generate_eval_examples())
    all_examples.extend(_generate_propose_examples())
    all_examples.extend(_generate_meta_examples())

    # Shuffle and limit
    random.shuffle(all_examples)
    return all_examples[:num_examples]


def to_chat_format(
    example: TrainingExample,
    system_prompt: str = SYSTEM_PROMPT_BASIC,
) -> dict:
    """Convert a training example to chat format (OpenAI/HF style)."""
    return {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": example.instruction},
            {"role": "assistant", "content": example.response},
        ]
    }


def to_alpaca_format(example: TrainingExample) -> dict:
    """Convert to Alpaca instruction-following format."""
    return {
        "instruction": f"Communicate using Slipstream protocol: {example.instruction}",
        "input": "",
        "output": example.response,
    }


def to_sharegpt_format(
    example: TrainingExample,
    system_prompt: str = SYSTEM_PROMPT_BASIC,
) -> dict:
    """Convert to ShareGPT format (used by Unsloth)."""
    return {
        "conversations": [
            {"from": "system", "value": system_prompt},
            {"from": "human", "value": example.instruction},
            {"from": "gpt", "value": example.response},
        ]
    }


def generate_dataset(
    output_path: Path,
    num_examples: int = 500,
    format: str = "chat",  # "chat", "alpaca", "sharegpt"
    system_prompt: str = SYSTEM_PROMPT_BASIC,
    seed: int = 42,
) -> int:
    """
    Generate a training dataset file.

    Args:
        output_path: Path to write JSONL file
        num_examples: Number of examples to generate
        format: Output format ("chat", "alpaca", "sharegpt")
        system_prompt: System prompt to include
        seed: Random seed

    Returns:
        Number of examples written
    """
    examples = generate_training_examples(num_examples, seed)

    converters = {
        "chat": lambda e: to_chat_format(e, system_prompt),
        "alpaca": to_alpaca_format,
        "sharegpt": lambda e: to_sharegpt_format(e, system_prompt),
    }

    converter = converters.get(format, converters["chat"])

    with open(output_path, "w", encoding="utf-8") as f:
        for example in examples:
            formatted = converter(example)
            f.write(json.dumps(formatted, ensure_ascii=False) + "\n")

    return len(examples)


# ============ CLI Entry Point ============

def main():
    """CLI entry point for dataset generation."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate Slipstream training dataset"
    )
    parser.add_argument(
        "-o", "--output",
        type=Path,
        default=Path("slipstream_train.jsonl"),
        help="Output file path",
    )
    parser.add_argument(
        "-n", "--num-examples",
        type=int,
        default=500,
        help="Number of examples to generate",
    )
    parser.add_argument(
        "-f", "--format",
        choices=["chat", "alpaca", "sharegpt"],
        default="sharegpt",
        help="Output format (sharegpt recommended for Unsloth)",
    )
    parser.add_argument(
        "--detailed-prompt",
        action="store_true",
        help="Use detailed system prompt (includes anchor reference)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )

    args = parser.parse_args()

    system_prompt = SYSTEM_PROMPT_DETAILED if args.detailed_prompt else SYSTEM_PROMPT_BASIC

    count = generate_dataset(
        output_path=args.output,
        num_examples=args.num_examples,
        format=args.format,
        system_prompt=system_prompt,
        seed=args.seed,
    )

    print(f"Generated {count} examples to {args.output}")
    print(f"Format: {args.format}")
    print(f"Ready for finetuning with Unsloth, HuggingFace, or similar tools")


if __name__ == "__main__":
    main()
