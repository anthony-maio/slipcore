"""
Slipstream Finetuning Dataset Generator

Generates training data to teach LLMs the Think-Quantize-Transmit cognitive skill.
NOT just format conversion - trains the semantic understanding of the UCR manifold.

Supports multiple output formats:
- Hugging Face datasets (JSONL)
- Unsloth/LoRA training format (ShareGPT)
- Claude training format

Key improvements over v1:
- Explicit THOUGHT supervision (not just instruction -> SLIP)
- UCR dimension labels (ACTION, POLARITY, DOMAIN, URGENCY)
- Fallback examples (when NOT to quantize)
- Multi-turn interagent workflows
- Consistent payload normalization
"""

from __future__ import annotations
import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List
from .ucr import get_default_ucr, UCRAnchor, UCR, Dimension


# ============ UCR Dimension Labels ============

# Human-readable labels for UCR dimension values
ACTION_LABELS = ["observe", "inform", "ask", "request", "propose", "commit", "evaluate", "meta"]
POLARITY_LABELS = ["strongly_negative", "negative", "declining", "mild_negative",
                   "neutral", "mild_positive", "positive", "strongly_positive"]
DOMAIN_LABELS = ["task", "plan", "observation", "evaluation", "control", "resource", "error", "general"]
URGENCY_LABELS = ["background", "low", "low_normal", "normal", "elevated", "high", "urgent", "critical"]


def _get_dimension_labels(coords: tuple) -> dict:
    """Convert UCR numeric coords to human-readable labels."""
    if len(coords) != 4:
        return {}
    return {
        "action": ACTION_LABELS[coords[0]] if coords[0] < len(ACTION_LABELS) else "unknown",
        "polarity": POLARITY_LABELS[coords[1]] if coords[1] < len(POLARITY_LABELS) else "unknown",
        "domain": DOMAIN_LABELS[coords[2]] if coords[2] < len(DOMAIN_LABELS) else "unknown",
        "urgency": URGENCY_LABELS[coords[3]] if coords[3] < len(URGENCY_LABELS) else "unknown",
    }


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
    """
    A single training example for finetuning Think-Quantize-Transmit.

    Now includes UCR semantic dimensions for manifold-aware training.
    """
    instruction: str          # User-facing instruction
    thought: str              # Natural language intent / reasoning (the THINK step)
    response: str             # SLIP wire format (the TRANSMIT step)
    anchor: str               # UCR anchor mnemonic
    src: str = "agent"
    dst: str = "other"
    # UCR semantic dimensions (the QUANTIZE step semantics)
    action: Optional[str] = None      # observe, inform, ask, request, propose, commit, evaluate, meta
    polarity: Optional[str] = None    # negative -> neutral -> positive
    domain: Optional[str] = None      # task, plan, observation, evaluation, control, resource, error, general
    urgency: Optional[str] = None     # background -> normal -> critical
    is_fallback: bool = False         # True if this is a fallback (unquantizable) example


def _normalize_payload(payload: str) -> str:
    """Normalize payload to underscore-separated tokens."""
    if not payload:
        return ""
    # Split on spaces and rejoin with underscores
    tokens = [t.strip() for t in payload.split() if t.strip()]
    return "_".join(tokens)


def _enrich_example_with_ucr(example: TrainingExample, ucr: Optional[UCR] = None) -> TrainingExample:
    """Add UCR dimension labels to a training example."""
    if ucr is None:
        ucr = get_default_ucr()

    anchor_obj = ucr.get_by_mnemonic(example.anchor)
    if anchor_obj:
        labels = _get_dimension_labels(anchor_obj.coords)
        example.action = labels.get("action")
        example.polarity = labels.get("polarity")
        example.domain = labels.get("domain")
        example.urgency = labels.get("urgency")

    return example


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

            # Normalize payload consistently
            normalized_payload = _normalize_payload(payload)
            response = f"SLIP v1 agent {dst} {anchor}"
            if normalized_payload:
                response += f" {normalized_payload}"

            examples.append(_enrich_example_with_ucr(TrainingExample(
                instruction=instruction,
                thought=thought,
                response=response,
                anchor=anchor,
                dst=dst,
            )))

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

            # Normalize payload consistently
            normalized_payload = _normalize_payload(payload)
            response = f"SLIP v1 agent {dst} {anchor}"
            if normalized_payload:
                response += f" {normalized_payload}"

            examples.append(_enrich_example_with_ucr(TrainingExample(
                instruction=instruction,
                thought=thought,
                response=response,
                anchor=anchor,
                dst=dst,
            )))

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

            # Normalize payload consistently
            normalized_payload = _normalize_payload(payload)
            response = f"SLIP v1 agent {dst} {anchor}"
            if normalized_payload:
                response += f" {normalized_payload}"

            examples.append(_enrich_example_with_ucr(TrainingExample(
                instruction=instruction,
                thought=thought,
                response=response,
                anchor=anchor,
                dst=dst,
            )))

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

            # Normalize payload consistently
            normalized_payload = _normalize_payload(payload)
            response = f"SLIP v1 agent {dst} {anchor}"
            if normalized_payload:
                response += f" {normalized_payload}"

            examples.append(_enrich_example_with_ucr(TrainingExample(
                instruction=instruction,
                thought=thought,
                response=response,
                anchor=anchor,
                dst=dst,
            )))

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
            normalized_payload = _normalize_payload(payload)
            response = f"SLIP v1 agent {dst} {anchor}"
            if normalized_payload:
                response += f" {normalized_payload}"

            examples.append(_enrich_example_with_ucr(TrainingExample(
                instruction=instruction,
                thought=thought,
                response=response,
                anchor=anchor,
                dst=dst,
            )))

    return examples


def _generate_fallback_examples() -> list[TrainingExample]:
    """
    Generate examples where the model should NOT quantize.

    These teach the model to recognize when natural language fallback is appropriate.
    """
    examples = []

    fallback_templates = [
        # Domain-specific jargon that doesn't map to UCR
        ("Tell {dst} to check kubernetes pods for OOMKilled events",
         "This is a very specific infrastructure request that doesn't map to standard anchors",
         "check_kubernetes_pods_for_OOMKilled_events"),

        ("Ask {dst} about the Redis cluster replication lag",
         "Specific database metrics query that's too technical for standard anchors",
         "redis_cluster_replication_lag"),

        ("Have {dst} investigate the JWT token expiry edge case",
         "Debugging a specific authentication issue",
         "jwt_token_expiry_edge_case"),

        ("Tell {dst} the nginx ingress is returning 502s",
         "Specific infrastructure error report",
         "nginx_ingress_502_errors"),

        # Multi-part requests that don't fit single anchor
        ("Ask {dst} to review the PR and also check the CI logs",
         "Multiple distinct actions in one request - should use fallback",
         "review_pr_and_check_ci_logs"),

        # Highly contextual messages
        ("Remind {dst} about what we discussed in standup",
         "Context-dependent reference that can't be quantized",
         "reminder_standup_discussion"),

        # Emotional/subjective content
        ("Let {dst} know I'm frustrated with the flaky tests",
         "Emotional content mixed with technical - better as natural language",
         "frustrated_flaky_tests"),
    ]

    destinations = ["ops", "sre", "devops", "backend", "infra"]

    for instruction_template, thought, payload in fallback_templates:
        for _ in range(2):
            dst = random.choice(destinations)
            instruction = instruction_template.format(dst=dst)

            response = f"SLIP v1 agent {dst} Fallback {payload}"

            examples.append(TrainingExample(
                instruction=instruction,
                thought=thought,
                response=response,
                anchor="Fallback",
                dst=dst,
                action="meta",
                polarity="neutral",
                domain="general",
                urgency="normal",
                is_fallback=True,
            ))

    return examples


def _generate_multiturn_examples() -> list[TrainingExample]:
    """
    Generate multi-turn interagent workflow examples.

    These show realistic back-and-forth coordination patterns.
    """
    examples = []

    # Workflow: Task assignment -> Ack -> Progress -> Complete
    workflow1 = [
        ("manager", "dev", "Assign the auth refactor task to dev",
         "Starting a new task assignment workflow",
         "RequestTask", "auth_refactor"),
        ("dev", "manager", "Acknowledge manager's task assignment",
         "Confirming I received the task",
         "MetaAck", ""),
        ("dev", "manager", "Update manager on auth refactor progress",
         "I've made good progress, reporting status",
         "InformProgress", "auth_refactor_60pct"),
        ("dev", "manager", "Tell manager the auth refactor is done",
         "Task complete, reporting success",
         "InformComplete", "auth_refactor"),
    ]

    # Workflow: Review request -> Needs work -> Fix -> Approve
    workflow2 = [
        ("dev", "reviewer", "Ask reviewer to check the pull request",
         "I need feedback on my code",
         "RequestReview", "pull_request_42"),
        ("reviewer", "dev", "Tell dev the code needs some changes",
         "Found some issues that need fixing",
         "EvalNeedsWork", "error_handling"),
        ("dev", "reviewer", "Let reviewer know fixes are done",
         "I've addressed the feedback",
         "InformComplete", "fixes_applied"),
        ("reviewer", "dev", "Approve dev's updated code",
         "Changes look good now",
         "EvalApprove", ""),
    ]

    # Workflow: Proposal -> Discussion -> Accept
    workflow3 = [
        ("architect", "team", "Propose a caching strategy to the team",
         "I have an idea for improving performance",
         "ProposePlan", "redis_caching"),
        ("lead", "architect", "Ask architect for clarification on caching",
         "I need more details about the approach",
         "AskClarify", "cache_invalidation"),
        ("architect", "lead", "Explain the cache invalidation strategy",
         "Providing additional details",
         "InformResult", "ttl_based_invalidation"),
        ("lead", "architect", "Accept the caching proposal",
         "The plan is solid, let's proceed",
         "Accept", ""),
    ]

    for workflow in [workflow1, workflow2, workflow3]:
        for src, dst, instruction, thought, anchor, payload in workflow:
            normalized_payload = _normalize_payload(payload)
            response = f"SLIP v1 {src} {dst} {anchor}"
            if normalized_payload:
                response += f" {normalized_payload}"

            example = TrainingExample(
                instruction=instruction,
                thought=thought,
                response=response,
                anchor=anchor,
                src=src,
                dst=dst,
            )
            examples.append(_enrich_example_with_ucr(example))

    return examples


# ============ Dataset Generator ============

def generate_training_examples(
    num_examples: int = 500,
    seed: int = 42,
    include_fallbacks: bool = True,
    include_multiturn: bool = True,
) -> list[TrainingExample]:
    """
    Generate a diverse set of training examples for Think-Quantize-Transmit.

    Args:
        num_examples: Target number of examples
        seed: Random seed for reproducibility
        include_fallbacks: Include fallback examples (teaches when NOT to quantize)
        include_multiturn: Include multi-turn workflow examples

    Returns:
        List of TrainingExample objects with UCR dimensions enriched
    """
    random.seed(seed)

    all_examples = []
    all_examples.extend(_generate_request_examples())
    all_examples.extend(_generate_inform_examples())
    all_examples.extend(_generate_eval_examples())
    all_examples.extend(_generate_propose_examples())
    all_examples.extend(_generate_meta_examples())

    if include_fallbacks:
        all_examples.extend(_generate_fallback_examples())

    if include_multiturn:
        all_examples.extend(_generate_multiturn_examples())

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


# ============ Think-Quantize-Transmit Converters ============
# These train the model on the cognitive skill, not just format conversion

def to_chat_with_thought(
    example: TrainingExample,
    system_prompt: str = SYSTEM_PROMPT_BASIC,
) -> dict:
    """
    Convert to chat format with explicit THOUGHT supervision.

    This trains Think-Quantize-Transmit as a cognitive pattern.
    """
    # Build response with thought and SLIP
    response = f"THOUGHT: {example.thought}\nSLIP: {example.response}"

    return {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": example.instruction},
            {"role": "assistant", "content": response},
        ]
    }


def to_sharegpt_with_thought(
    example: TrainingExample,
    system_prompt: str = SYSTEM_PROMPT_BASIC,
) -> dict:
    """
    Convert to ShareGPT format with explicit THOUGHT supervision.

    This is the recommended format for Unsloth finetuning.
    """
    # Build response with thought and SLIP
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
    """
    Convert to ShareGPT format with full semantic annotation.

    Includes THOUGHT, semantic dimensions, and SLIP output.
    This provides maximum supervision for the UCR manifold.
    """
    # Build semantic hint from dimensions
    dims = []
    if example.action:
        dims.append(f"ACTION={example.action}")
    if example.domain:
        dims.append(f"DOMAIN={example.domain}")
    if example.urgency:
        dims.append(f"URGENCY={example.urgency}")
    if example.polarity:
        dims.append(f"POLARITY={example.polarity}")

    semantic_hint = " | ".join(dims) if dims else "SEMANTICS=unknown"

    # Build full response
    response = (
        f"THOUGHT: {example.thought}\n"
        f"QUANTIZE: [{semantic_hint}] -> {example.anchor}\n"
        f"SLIP: {example.response}"
    )

    return {
        "conversations": [
            {"from": "system", "value": system_prompt},
            {"from": "human", "value": example.instruction},
            {"from": "gpt", "value": response},
        ]
    }


def generate_dataset(
    output_path: Path,
    num_examples: int = 500,
    format: str = "sharegpt_thought",  # Best for Think-Quantize-Transmit training
    system_prompt: str = SYSTEM_PROMPT_BASIC,
    seed: int = 42,
    include_fallbacks: bool = True,
    include_multiturn: bool = True,
) -> int:
    """
    Generate a training dataset file for Think-Quantize-Transmit.

    Args:
        output_path: Path to write JSONL file
        num_examples: Number of examples to generate
        format: Output format:
            - "chat": OpenAI chat format (instruction -> SLIP only)
            - "alpaca": Alpaca instruction format
            - "sharegpt": ShareGPT format (instruction -> SLIP only)
            - "chat_thought": Chat with THOUGHT supervision
            - "sharegpt_thought": ShareGPT with THOUGHT (recommended for Unsloth)
            - "sharegpt_semantics": ShareGPT with full UCR dimension annotations
        system_prompt: System prompt to include
        seed: Random seed
        include_fallbacks: Include examples that should NOT be quantized
        include_multiturn: Include multi-turn workflow examples

    Returns:
        Number of examples written
    """
    examples = generate_training_examples(
        num_examples, seed,
        include_fallbacks=include_fallbacks,
        include_multiturn=include_multiturn,
    )

    converters = {
        "chat": lambda e: to_chat_format(e, system_prompt),
        "alpaca": to_alpaca_format,
        "sharegpt": lambda e: to_sharegpt_format(e, system_prompt),
        # Think-Quantize-Transmit formats (recommended)
        "chat_thought": lambda e: to_chat_with_thought(e, system_prompt),
        "sharegpt_thought": lambda e: to_sharegpt_with_thought(e, system_prompt),
        "sharegpt_semantics": lambda e: to_sharegpt_with_semantics(e, system_prompt),
    }

    converter = converters.get(format, converters["sharegpt_thought"])

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
        description="Generate Slipstream Think-Quantize-Transmit training dataset"
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
        choices=[
            "chat", "alpaca", "sharegpt",  # Basic formats
            "chat_thought", "sharegpt_thought", "sharegpt_semantics",  # TQT formats
        ],
        default="sharegpt_thought",
        help="Output format. Recommended: sharegpt_thought (trains cognitive skill)",
    )
    parser.add_argument(
        "--detailed-prompt",
        action="store_true",
        help="Use detailed system prompt (includes anchor reference)",
    )
    parser.add_argument(
        "--no-fallbacks",
        action="store_true",
        help="Exclude fallback examples (when NOT to quantize)",
    )
    parser.add_argument(
        "--no-multiturn",
        action="store_true",
        help="Exclude multi-turn workflow examples",
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
        include_fallbacks=not args.no_fallbacks,
        include_multiturn=not args.no_multiturn,
    )

    print(f"\nGenerated {count} examples to {args.output}")
    print(f"Format: {args.format}")

    # Show sample
    with open(args.output, "r", encoding="utf-8") as f:
        sample = json.loads(f.readline())

    print("\n--- Sample output ---")
    if "conversations" in sample:
        for msg in sample["conversations"]:
            role = msg.get("from", "unknown")
            value = msg.get("value", "")[:200]
            print(f"[{role}]: {value}...")
    elif "messages" in sample:
        for msg in sample["messages"]:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")[:200]
            print(f"[{role}]: {content}...")

    print("\n--- Ready for finetuning ---")
    print("Recommended: Unsloth with GLM-4-9B-0414 or similar")


if __name__ == "__main__":
    main()
