#!/usr/bin/env python
"""
Enhanced dataset generator using Ollama for diverse, realistic examples.

This generator uses an LLM to create:
1. Diverse, realistic goal descriptions (not just templates)
2. Optionally: more natural agent responses

Usage:
    python -m slipcore.generate_dataset_llm --num-conversations 100 --model qwen2.5:14b

    # Use remote Ollama (e.g., lab machine with dual 3090s)
    python -m slipcore.generate_dataset_llm --ollama-url http://192.168.1.100:11434 --model qwen3:30b
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import urllib.request
import urllib.error

from slipcore.protocol import (
    SlipMessage,
    Act,
    FrameType,
    Slot,
    encode_message,
)


@dataclass
class OllamaClient:
    """Simple Ollama API client."""
    base_url: str = "http://localhost:11434"
    model: str = "qwen2.5:14b"
    timeout: int = 300  # 5 min timeout for larger models

    def generate(self, prompt: str, temperature: float = 0.8, max_tokens: int = 256) -> str:
        """Generate text from Ollama."""
        url = f"{self.base_url}/api/generate"
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            }
        }

        data = json.dumps(payload).encode('utf-8')
        req = urllib.request.Request(url, data=data, headers={'Content-Type': 'application/json'})

        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as response:
                result = json.loads(response.read().decode('utf-8'))
                text = result.get('response', '').strip()
                # Handle qwen3's thinking mode - strip <think>...</think> blocks
                if '<think>' in text:
                    import re
                    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()
                return text
        except urllib.error.URLError as e:
            raise ConnectionError(f"Failed to connect to Ollama at {self.base_url}: {e}")

    def list_models(self) -> List[str]:
        """List available models."""
        url = f"{self.base_url}/api/tags"
        req = urllib.request.Request(url)
        try:
            with urllib.request.urlopen(req, timeout=10) as response:
                result = json.loads(response.read().decode('utf-8'))
                return [m['name'] for m in result.get('models', [])]
        except urllib.error.URLError:
            return []


# Domain categories for diverse goal generation
DOMAINS = [
    "authentication and authorization",
    "payment processing",
    "search and discovery",
    "inventory management",
    "order fulfillment",
    "notification systems",
    "analytics and reporting",
    "user profile management",
    "content management",
    "API gateway",
    "caching layer",
    "message queue",
    "file storage",
    "logging and monitoring",
    "rate limiting",
    "session management",
    "webhook handling",
    "data migration",
    "backup and recovery",
    "CI/CD pipeline",
]

TASK_TYPES = [
    "implement a new feature",
    "fix a bug",
    "optimize performance",
    "add error handling",
    "write tests",
    "refactor code",
    "add logging",
    "implement validation",
    "add caching",
    "improve security",
    "add documentation",
    "create API endpoint",
    "integrate external service",
    "implement retry logic",
    "add feature flags",
]


def generate_goals_with_llm(
    client: OllamaClient,
    num_goals: int,
    batch_size: int = 10,
    verbose: bool = True,
) -> List[str]:
    """Generate diverse, realistic goals using an LLM."""
    goals = []

    prompt_template = """Generate {batch_size} realistic software engineering task descriptions for a multi-agent coding system.

Requirements:
- Each task should be specific and actionable
- Focus on backend/infrastructure work
- Vary the complexity (some simple, some complex)
- Include different domains: {domains}
- Include different task types: {task_types}

Format: Return ONLY a JSON array of strings, no explanation.
Example: ["Add rate limiting to the auth API endpoint", "Implement retry logic for payment webhook failures"]

Generate {batch_size} tasks:"""

    batches_needed = (num_goals + batch_size - 1) // batch_size

    for i in range(batches_needed):
        if verbose:
            print(f"  Generating batch {i+1}/{batches_needed}...", end=" ", flush=True)

        # Randomize domains and task types for variety
        selected_domains = random.sample(DOMAINS, min(5, len(DOMAINS)))
        selected_tasks = random.sample(TASK_TYPES, min(5, len(TASK_TYPES)))

        prompt = prompt_template.format(
            batch_size=batch_size,
            domains=", ".join(selected_domains),
            task_types=", ".join(selected_tasks),
        )

        try:
            response = client.generate(prompt, temperature=0.9, max_tokens=1024)

            # Parse JSON array from response
            # Handle cases where LLM adds extra text
            start = response.find('[')
            end = response.rfind(']') + 1
            if start >= 0 and end > start:
                json_str = response[start:end]
                batch_goals = json.loads(json_str)
                goals.extend(batch_goals)
                if verbose:
                    print(f"got {len(batch_goals)} goals")
            else:
                if verbose:
                    print("failed to parse, using fallback")
                # Fallback: generate simple goals
                goals.extend([
                    f"Implement {random.choice(TASK_TYPES)} for {random.choice(DOMAINS)}"
                    for _ in range(batch_size)
                ])
        except Exception as e:
            if verbose:
                print(f"error: {e}, using fallback")
            goals.extend([
                f"Implement {random.choice(TASK_TYPES)} for {random.choice(DOMAINS)}"
                for _ in range(batch_size)
            ])

        # Small delay to avoid hammering the API
        if i < batches_needed - 1:
            time.sleep(0.5)

    return goals[:num_goals]


def make_conversation_from_goal(
    conv_id: int,
    goal_text: str,
    rng: random.Random,
) -> Dict[str, object]:
    """Build a synthetic conversation from a goal."""
    goal_id = conv_id
    task_id = conv_id
    result_id = conv_id
    priority = rng.randint(1, 3)

    # 1) Coordinator REQUEST to planner
    req_msg = SlipMessage(
        conv_id=conv_id,
        turn=1,
        src=0,
        dst=1,
        act=Act.REQUEST,
        frame=FrameType.TASK,
        slots={
            Slot.GOAL_ID: goal_id,
            Slot.TASK_ID: task_id,
            Slot.PRIORITY: priority,
            Slot.TAG: f"goal_{goal_id}",
        },
    )
    req_wire = encode_message(req_msg)

    # 2) Planner PROPOSE/PLAN
    plan_tag = rng.choice(["plan_v1", "plan_safe", "plan_fast", "plan_minimal", "plan_robust"])
    prop_msg = SlipMessage(
        conv_id=conv_id,
        turn=2,
        src=1,
        dst=0,
        act=Act.PROPOSE,
        frame=FrameType.PLAN,
        slots={
            Slot.GOAL_ID: goal_id,
            Slot.TASK_ID: task_id,
            Slot.PRIORITY: priority,
            Slot.TAG: plan_tag,
        },
    )
    prop_wire = encode_message(prop_msg)

    # 3) Executor INFORM/OBSERVATION
    status = rng.choice(["in_progress", "done", "blocked", "waiting", "partial"])
    exec_msg = SlipMessage(
        conv_id=conv_id,
        turn=3,
        src=2,
        dst=0,
        act=Act.INFORM,
        frame=FrameType.OBSERVATION,
        slots={
            Slot.GOAL_ID: goal_id,
            Slot.TASK_ID: task_id,
            Slot.RESULT_ID: result_id,
            Slot.STATUS: status,
            Slot.TAG: "exec_status",
        },
    )
    exec_wire = encode_message(exec_msg)

    # 4) Critic EVAL/EVALUATION
    if status == "done":
        score = rng.randint(8, 10)
        eval_tag = "ok"
    elif status in ("in_progress", "partial"):
        score = rng.randint(5, 7)
        eval_tag = "partial"
    else:
        score = rng.randint(0, 4)
        eval_tag = "blocked"

    critic_msg = SlipMessage(
        conv_id=conv_id,
        turn=4,
        src=3,
        dst=0,
        act=Act.EVAL,
        frame=FrameType.EVALUATION,
        slots={
            Slot.GOAL_ID: goal_id,
            Slot.TASK_ID: task_id,
            Slot.SCORE: score,
            Slot.TAG: eval_tag,
        },
    )
    critic_wire = encode_message(critic_msg)

    return {
        "goal_id": goal_id,
        "task_id": task_id,
        "result_id": result_id,
        "goal_text": goal_text,
        "request_wire": req_wire,
        "proposal_wire": prop_wire,
        "exec_wire": exec_wire,
        "critic_wire": critic_wire,
    }


def build_planner_example(conv: Dict[str, object]) -> Tuple[str, str]:
    goal_id = conv["goal_id"]
    goal_text = conv["goal_text"]
    req_wire = conv["request_wire"]
    target = conv["proposal_wire"]

    prompt = (
        "You are a planner agent in a multi-agent system.\n"
        "Agents communicate using the SLIPCore protocol in a compact wire format called nSLIP.\n"
        "You receive a REQUEST/TASK message and a goal description.\n"
        "Your job is to reply with a single PROPOSE/PLAN nSLIP message.\n\n"
        f"Current message:\n{req_wire}\n\n"
        f"Goal[{goal_id}]:\n{goal_text}\n\n"
        "Respond with only one nSLIP line, no explanations."
    )
    return prompt, target


def build_executor_example(conv: Dict[str, object]) -> Tuple[str, str]:
    goal_id = conv["goal_id"]
    goal_text = conv["goal_text"]
    prop_wire = conv["proposal_wire"]
    target = conv["exec_wire"]

    prompt = (
        "You are an executor agent in a multi-agent system.\n"
        "Agents communicate using the SLIPCore protocol in a compact wire format called nSLIP.\n"
        "You receive a PROPOSE/PLAN message and a goal description.\n"
        "Your job is to reply with a single INFORM/OBSERVATION nSLIP message\n"
        "reporting your execution status.\n\n"
        f"Current message:\n{prop_wire}\n\n"
        f"Goal[{goal_id}]:\n{goal_text}\n\n"
        "Respond with only one nSLIP line, no explanations."
    )
    return prompt, target


def build_critic_example(conv: Dict[str, object]) -> Tuple[str, str]:
    goal_id = conv["goal_id"]
    goal_text = conv["goal_text"]
    exec_wire = conv["exec_wire"]
    target = conv["critic_wire"]

    prompt = (
        "You are a critic agent in a multi-agent system.\n"
        "Agents communicate using the SLIPCore protocol in a compact wire format called nSLIP.\n"
        "You receive an INFORM/OBSERVATION message about a goal.\n"
        "Your job is to reply with a single EVAL/EVALUATION nSLIP message\n"
        "scoring the outcome and tagging its quality.\n\n"
        f"Current message:\n{exec_wire}\n\n"
        f"Goal[{goal_id}]:\n{goal_text}\n\n"
        "Respond with only one nSLIP line, no explanations."
    )
    return prompt, target


def build_coordinator_example(conv: Dict[str, object]) -> Tuple[str, str]:
    goal_id = conv["goal_id"]
    goal_text = conv["goal_text"]
    target = conv["request_wire"]

    prompt = (
        "You are a coordinator agent in a multi-agent system.\n"
        "Your job is to convert human goals into SLIPCore REQUEST/TASK messages\n"
        "in the compact nSLIP wire format.\n\n"
        f"Goal[{goal_id}]:\n{goal_text}\n\n"
        "Respond with only one nSLIP line, no explanations."
    )
    return prompt, target


def generate_dataset(
    goals: List[str],
    out_path: Path,
    seed: int = 0,
    include_coordinator: bool = True,
    verbose: bool = True,
) -> None:
    """Generate dataset from list of goals."""
    rng = random.Random(seed)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8") as f:
        for conv_id, goal_text in enumerate(goals, start=1):
            conv = make_conversation_from_goal(conv_id, goal_text, rng)

            if include_coordinator:
                inp, tgt = build_coordinator_example(conv)
                row = {"role": "coordinator", "input": inp, "target": tgt}
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

            inp, tgt = build_planner_example(conv)
            row = {"role": "planner", "input": inp, "target": tgt}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

            inp, tgt = build_executor_example(conv)
            row = {"role": "executor", "input": inp, "target": tgt}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

            inp, tgt = build_critic_example(conv)
            row = {"role": "critic", "input": inp, "target": tgt}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

            if verbose and conv_id % 100 == 0:
                print(f"  Wrote {conv_id} conversations...")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate nSLIP finetune dataset using Ollama for diverse goals."
    )
    p.add_argument(
        "--num-conversations", "-n",
        type=int,
        default=100,
        help="Number of conversations to generate (default: 100).",
    )
    p.add_argument(
        "--output", "-o",
        type=str,
        default="data/finetune/nslip_pairs.llm.jsonl",
        help="Output JSONL path.",
    )
    p.add_argument(
        "--ollama-url",
        type=str,
        default="http://localhost:11434",
        help="Ollama API URL (default: http://localhost:11434).",
    )
    p.add_argument(
        "--model", "-m",
        type=str,
        default="qwen2.5:14b",
        help="Ollama model to use (default: qwen2.5:14b).",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42).",
    )
    p.add_argument(
        "--no-coordinator",
        action="store_true",
        help="Exclude coordinator examples.",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=10,
        help="Goals to generate per LLM call (default: 10).",
    )
    p.add_argument(
        "--list-models",
        action="store_true",
        help="List available Ollama models and exit.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    client = OllamaClient(
        base_url=args.ollama_url,
        model=args.model,
    )

    # List models mode
    if args.list_models:
        print(f"Models available at {args.ollama_url}:")
        models = client.list_models()
        if models:
            for m in models:
                print(f"  - {m}")
        else:
            print("  (none found or connection failed)")
        return

    # Check connection
    print(f"Connecting to Ollama at {args.ollama_url}...")
    models = client.list_models()
    if not models:
        print(f"Error: Cannot connect to Ollama at {args.ollama_url}")
        sys.exit(1)

    if args.model not in models:
        print(f"Warning: Model '{args.model}' not found. Available: {', '.join(models)}")
        print("Continuing anyway (Ollama may pull it)...")

    print(f"Using model: {args.model}")
    print(f"Generating {args.num_conversations} diverse goals...")

    random.seed(args.seed)
    goals = generate_goals_with_llm(
        client,
        args.num_conversations,
        batch_size=args.batch_size,
        verbose=True,
    )

    print(f"\nGenerating dataset...")
    out_path = Path(args.output)
    generate_dataset(
        goals,
        out_path,
        seed=args.seed,
        include_coordinator=not args.no_coordinator,
        verbose=True,
    )

    num_examples = args.num_conversations * (4 if not args.no_coordinator else 3)
    print(f"\nDone! Wrote {num_examples} examples to {out_path}")

    # Show a few sample goals
    print("\nSample goals generated:")
    for g in goals[:5]:
        print(f"  - {g}")


if __name__ == "__main__":
    main()
