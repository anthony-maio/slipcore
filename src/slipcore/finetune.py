"""Slipstream v3 Finetuning Dataset Generator.

Generates training data to teach LLMs the Think-Quantize-Transmit cognitive skill
using the v3 Force-Object factorized wire format.

Wire format: SLIP v3 <src> <dst> <Force> <Object> [payload...]
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


# ============ System Prompts ============

SYSTEM_PROMPT_BASIC = """You are an AI agent that communicates using the Slipstream protocol (SLIP v3).

Slipstream uses semantic quantization with a factorized intent model: Force (action verb) + Object (domain noun).

Wire format: SLIP v3 <src> <dst> <Force> <Object> [payload...]

Forces: Observe, Inform, Ask, Request, Propose, Commit, Eval, Meta, Accept, Reject, Error, Fallback

Example:
- To request a code review: SLIP v3 alice bob Request Review
- To report completion: SLIP v3 worker manager Inform Complete task42
- To propose a plan: SLIP v3 planner team Propose Plan auth

Always respond with SLIP v3 messages when coordinating with other agents."""

SYSTEM_PROMPT_DETAILED = """You are an AI agent that communicates using the Slipstream protocol (SLIP v3).

## Protocol Overview
Slipstream achieves 80%+ token savings by transmitting factorized intents: Force (action verb) + Object (domain noun).

## Wire Format
SLIP v3 <source> <destination> <Force> <Object> [optional payload...]

## Force Tokens (12 closed vocabulary)
Observe, Inform, Ask, Request, Propose, Commit, Eval, Meta, Accept, Reject, Error, Fallback

## Common Object Tokens
Task, Plan, Review, Help, Status, Complete, Blocked, Progress, Error, Resource,
Approve, NeedsWork, Ack, Sync, Handoff, Escalate, Generic

## Examples
User: "Tell the executor to implement the login feature"
You: SLIP v3 planner executor Request Task login

User: "Let the team know the auth module is done"
You: SLIP v3 developer team Inform Complete auth

User: "Ask for feedback on the database schema"
You: SLIP v3 developer reviewer Request Review database

Always use SLIP v3 format when coordinating with other agents."""


# ============ Training Example ============

@dataclass
class TrainingExample:
    """A single training example for finetuning Think-Quantize-Transmit."""

    instruction: str
    thought: str
    response: str
    force: str
    obj: str
    src: str = "agent"
    dst: str = "other"
    is_fallback: bool = False


# ============ Template Generators ============

def _generate_request_examples() -> list[TrainingExample]:
    examples = []

    templates = [
        ("Tell {dst} to implement the {feature} feature",
         "I need {dst} to work on implementing {feature}",
         "Request", "Task", "{feature}"),
        ("Ask {dst} to review the {module} code",
         "I need feedback on the {module} implementation",
         "Request", "Review", "{module}"),
        ("Get assistance from {dst} on debugging",
         "I need {dst}'s help to debug this issue",
         "Request", "Help", "debugging"),
        ("Ask {dst} to create a plan for {goal}",
         "We need a strategy for {goal}",
         "Request", "Plan", "{goal}"),
        ("Ask {dst} to cancel the deployment",
         "We need to stop the deployment",
         "Request", "Cancel", "deployment"),
        ("Request priority escalation from {dst}",
         "This needs to be prioritized",
         "Request", "Priority", "escalation"),
        ("Ask {dst} to allocate resources for {feature}",
         "We need more resources for {feature}",
         "Request", "Resource", "{feature}"),
    ]

    features = ["authentication", "caching", "logging", "API", "database"]
    modules = ["auth", "payment", "user", "api", "core"]
    goals = ["migration", "refactor", "optimization", "scaling"]
    destinations = ["bob", "alice", "reviewer", "planner", "executor", "team"]

    for tmpl, thought_tmpl, force, obj, payload_tmpl in templates:
        for _ in range(3):
            dst = random.choice(destinations)
            feature = random.choice(features)
            module = random.choice(modules)
            goal = random.choice(goals)

            instruction = tmpl.format(dst=dst, feature=feature, module=module, goal=goal)
            thought = thought_tmpl.format(dst=dst, feature=feature, module=module, goal=goal)
            payload = payload_tmpl.format(feature=feature, module=module, goal=goal)

            # Make payload BPE-safe (alphanumeric only)
            clean_payload = "".join(c if c.isalnum() else "" for c in payload)
            wire = f"SLIP v3 agent {dst} {force} {obj}"
            if clean_payload:
                wire += f" {clean_payload}"

            examples.append(TrainingExample(
                instruction=instruction,
                thought=thought,
                response=wire,
                force=force,
                obj=obj,
                dst=dst,
            ))

    return examples


def _generate_inform_examples() -> list[TrainingExample]:
    examples = []

    templates = [
        ("Tell {dst} that the {task} is done",
         "I finished working on {task}",
         "Inform", "Complete", "{task}"),
        ("Update {dst} on the {task} progress",
         "I'm making progress on {task}",
         "Inform", "Progress", "{task}"),
        ("Tell {dst} you're blocked on {blocker}",
         "I can't proceed because of {blocker}",
         "Inform", "Blocked", "{blocker}"),
        ("Give {dst} a status update on {task}",
         "Here's where things stand with {task}",
         "Inform", "Status", "{task}"),
        ("Share results with {dst}",
         "Here are the computed results",
         "Inform", "Result", ""),
    ]

    tasks = ["auth", "api", "database", "caching", "tests"]
    blockers = ["credentials", "review", "dependencies", "resources"]
    destinations = ["manager", "team", "coordinator", "lead"]

    for tmpl, thought_tmpl, force, obj, payload_tmpl in templates:
        for _ in range(3):
            dst = random.choice(destinations)
            task = random.choice(tasks)
            blocker = random.choice(blockers)

            instruction = tmpl.format(dst=dst, task=task, blocker=blocker)
            thought = thought_tmpl.format(dst=dst, task=task, blocker=blocker)
            payload = payload_tmpl.format(task=task, blocker=blocker)

            clean_payload = "".join(c if c.isalnum() else "" for c in payload)
            wire = f"SLIP v3 agent {dst} {force} {obj}"
            if clean_payload:
                wire += f" {clean_payload}"

            examples.append(TrainingExample(
                instruction=instruction,
                thought=thought,
                response=wire,
                force=force,
                obj=obj,
                dst=dst,
            ))

    return examples


def _generate_eval_examples() -> list[TrainingExample]:
    examples = []

    templates = [
        ("Approve {dst}'s proposal",
         "The proposal looks good, approving it",
         "Eval", "Approve", ""),
        ("Tell {dst} their code needs changes",
         "There are some issues that need fixing",
         "Eval", "NeedsWork", ""),
        ("Mark the review as complete for {dst}",
         "The evaluation is done",
         "Eval", "Complete", ""),
    ]

    destinations = ["developer", "author", "submitter", "engineer"]

    for tmpl, thought_tmpl, force, obj, payload in templates:
        for _ in range(3):
            dst = random.choice(destinations)
            instruction = tmpl.format(dst=dst)
            thought = thought_tmpl.format(dst=dst)
            wire = f"SLIP v3 agent {dst} {force} {obj}"
            if payload:
                wire += f" {payload}"

            examples.append(TrainingExample(
                instruction=instruction,
                thought=thought,
                response=wire,
                force=force,
                obj=obj,
                dst=dst,
            ))

    return examples


def _generate_propose_examples() -> list[TrainingExample]:
    examples = []

    templates = [
        ("Suggest a plan to {dst} for {goal}",
         "I have an idea for how to handle {goal}",
         "Propose", "Plan", "{goal}"),
        ("Suggest a change to {dst}",
         "I think we should modify the approach",
         "Propose", "Change", ""),
        ("Offer {dst} an alternative approach",
         "There's another way we could do this",
         "Propose", "Alternative", ""),
    ]

    goals = ["refactoring", "migration", "optimization", "scaling"]
    destinations = ["team", "lead", "architect", "manager"]

    for tmpl, thought_tmpl, force, obj, payload_tmpl in templates:
        for _ in range(3):
            dst = random.choice(destinations)
            goal = random.choice(goals)
            instruction = tmpl.format(dst=dst, goal=goal)
            thought = thought_tmpl.format(dst=dst, goal=goal)
            payload = payload_tmpl.format(goal=goal)

            clean_payload = "".join(c if c.isalnum() else "" for c in payload)
            wire = f"SLIP v3 agent {dst} {force} {obj}"
            if clean_payload:
                wire += f" {clean_payload}"

            examples.append(TrainingExample(
                instruction=instruction,
                thought=thought,
                response=wire,
                force=force,
                obj=obj,
                dst=dst,
            ))

    return examples


def _generate_meta_examples() -> list[TrainingExample]:
    examples = []

    templates = [
        ("Accept {dst}'s request", "Yes, I agree", "Accept", "Generic", ""),
        ("Reject {dst}'s request", "I can't do this", "Reject", "Generic", ""),
        ("Acknowledge {dst}'s message", "Got it, understood", "Meta", "Ack", ""),
        ("Hand off to {dst}", "Passing this over to you", "Meta", "Handoff", ""),
        ("Sync with {dst}", "Checking in", "Meta", "Sync", ""),
    ]

    destinations = ["alice", "bob", "team", "manager", "coordinator"]

    for tmpl, thought, force, obj, payload in templates:
        for _ in range(2):
            dst = random.choice(destinations)
            instruction = tmpl.format(dst=dst)
            wire = f"SLIP v3 agent {dst} {force} {obj}"
            if payload:
                wire += f" {payload}"

            examples.append(TrainingExample(
                instruction=instruction,
                thought=thought,
                response=wire,
                force=force,
                obj=obj,
                dst=dst,
            ))

    return examples


def _generate_fallback_examples() -> list[TrainingExample]:
    examples = []

    fallback_templates = [
        ("Tell {dst} to check kubernetes pods for OOMKilled events",
         "This is too specific for standard anchors"),
        ("Ask {dst} about the Redis cluster replication lag",
         "Specific database metrics query"),
        ("Tell {dst} the nginx ingress is returning 502s",
         "Specific infrastructure error report"),
    ]

    destinations = ["ops", "sre", "devops", "backend"]

    for instruction_tmpl, thought in fallback_templates:
        for _ in range(2):
            dst = random.choice(destinations)
            instruction = instruction_tmpl.format(dst=dst)
            wire = f"SLIP v3 agent {dst} Fallback Generic refXXXX"

            examples.append(TrainingExample(
                instruction=instruction,
                thought=thought,
                response=wire,
                force="Fallback",
                obj="Generic",
                dst=dst,
                is_fallback=True,
            ))

    return examples


# ============ Dataset Generator ============

def generate_training_examples(
    num_examples: int = 500,
    seed: int = 42,
    include_fallbacks: bool = True,
) -> list[TrainingExample]:
    """Generate a diverse set of v3 training examples."""
    random.seed(seed)

    all_examples = []
    all_examples.extend(_generate_request_examples())
    all_examples.extend(_generate_inform_examples())
    all_examples.extend(_generate_eval_examples())
    all_examples.extend(_generate_propose_examples())
    all_examples.extend(_generate_meta_examples())

    if include_fallbacks:
        all_examples.extend(_generate_fallback_examples())

    random.shuffle(all_examples)
    return all_examples[:num_examples]


def to_sharegpt_format(example: TrainingExample, system_prompt: str = SYSTEM_PROMPT_BASIC) -> dict:
    return {
        "conversations": [
            {"from": "system", "value": system_prompt},
            {"from": "human", "value": example.instruction},
            {"from": "gpt", "value": example.response},
        ]
    }


def to_sharegpt_with_thought(
    example: TrainingExample,
    system_prompt: str = SYSTEM_PROMPT_BASIC,
) -> dict:
    response = f"THOUGHT: {example.thought}\nSLIP: {example.response}"
    return {
        "conversations": [
            {"from": "system", "value": system_prompt},
            {"from": "human", "value": example.instruction},
            {"from": "gpt", "value": response},
        ]
    }


def to_sharegpt_with_semantics(
    example: TrainingExample,
    system_prompt: str = SYSTEM_PROMPT_BASIC,
) -> dict:
    response = (
        f"THOUGHT: {example.thought}\n"
        f"QUANTIZE: Force={example.force} Object={example.obj}\n"
        f"SLIP: {example.response}"
    )
    return {
        "conversations": [
            {"from": "system", "value": system_prompt},
            {"from": "human", "value": example.instruction},
            {"from": "gpt", "value": response},
        ]
    }


def to_chat_format(example: TrainingExample, system_prompt: str = SYSTEM_PROMPT_BASIC) -> dict:
    return {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": example.instruction},
            {"role": "assistant", "content": example.response},
        ]
    }


def to_alpaca_format(example: TrainingExample) -> dict:
    return {
        "instruction": f"Communicate using Slipstream v3 protocol: {example.instruction}",
        "input": "",
        "output": example.response,
    }


def generate_dataset(
    output_path: Path,
    num_examples: int = 500,
    format: str = "sharegpt_thought",
    system_prompt: str = SYSTEM_PROMPT_BASIC,
    seed: int = 42,
    include_fallbacks: bool = True,
) -> int:
    """Generate a v3 training dataset file."""
    examples = generate_training_examples(num_examples, seed, include_fallbacks)

    converters = {
        "chat": lambda e: to_chat_format(e, system_prompt),
        "alpaca": to_alpaca_format,
        "sharegpt": lambda e: to_sharegpt_format(e, system_prompt),
        "sharegpt_thought": lambda e: to_sharegpt_with_thought(e, system_prompt),
        "sharegpt_semantics": lambda e: to_sharegpt_with_semantics(e, system_prompt),
    }

    converter = converters.get(format, converters["sharegpt_thought"])

    with open(output_path, "w", encoding="utf-8") as f:
        for example in examples:
            formatted = converter(example)
            f.write(json.dumps(formatted, ensure_ascii=False) + "\n")

    return len(examples)


def main() -> None:
    """CLI entry point for dataset generation."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate Slipstream v3 Think-Quantize-Transmit training dataset"
    )
    parser.add_argument("-o", "--output", type=Path, default=Path("slipstream_train.jsonl"))
    parser.add_argument("-n", "--num-examples", type=int, default=500)
    parser.add_argument(
        "-f", "--format",
        choices=["chat", "alpaca", "sharegpt", "sharegpt_thought", "sharegpt_semantics"],
        default="sharegpt_thought",
    )
    parser.add_argument("--detailed-prompt", action="store_true")
    parser.add_argument("--no-fallbacks", action="store_true")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    system_prompt = SYSTEM_PROMPT_DETAILED if args.detailed_prompt else SYSTEM_PROMPT_BASIC
    count = generate_dataset(
        output_path=args.output,
        num_examples=args.num_examples,
        format=args.format,
        system_prompt=system_prompt,
        seed=args.seed,
        include_fallbacks=not args.no_fallbacks,
    )

    print(f"Generated {count} examples to {args.output}")
    print(f"Format: {args.format}")


if __name__ == "__main__":
    main()
