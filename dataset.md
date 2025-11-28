#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple

from slipcore.protocol import (
    SlipMessage,
    Act,
    FrameType,
    Slot,
    encode_message,
)


# ---------- Simple goal templates ----------

GOAL_TEMPLATES = [
    "Implement a health-check endpoint for the {service} service.",
    "Refactor the {service} module to remove legacy token logic.",
    "Add structured logging to the {service} component.",
    "Optimize the database queries in the {service} path.",
    "Add basic input validation to the {service} API.",
    "Write unit tests for the {service} service.",
    "Introduce feature flag support for the {service} rollout.",
    "Implement retry logic for transient failures in {service}.",
]

SERVICES = [
    "auth",
    "billing",
    "search",
    "inventory",
    "orders",
    "notifications",
]


def sample_goal_text(rng: random.Random, idx: int) -> str:
    tmpl = rng.choice(GOAL_TEMPLATES)
    service = rng.choice(SERVICES)
    return tmpl.format(service=service)


# ---------- Conversation synthesis ----------

def make_conversation(
    conv_id: int,
    rng: random.Random,
) -> Dict[str, object]:
    """
    Build a synthetic 4-message conversation:

    - coordinator -> planner : REQUEST/TASK
    - planner     -> coord   : PROPOSE/PLAN
    - executor    -> coord   : INFORM/OBSERVATION
    - critic      -> coord   : EVAL/EVALUATION
    """
    goal_id = conv_id  # simple 1-1 mapping
    task_id = conv_id
    result_id = conv_id

    goal_text = sample_goal_text(rng, conv_id)
    priority = rng.randint(1, 3)

    # 1) Coordinator REQUEST to planner
    req_msg = SlipMessage(
        conv_id=conv_id,
        turn=1,
        src=0,  # coordinator
        dst=1,  # planner
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

    # 2) Planner PROPOSE/PLAN to coordinator
    plan_tag = rng.choice(["plan_v1", "plan_safe", "plan_fast"])
    prop_msg = SlipMessage(
        conv_id=conv_id,
        turn=2,
        src=1,  # planner
        dst=0,  # coordinator
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

    # 3) Executor INFORM/OBSERVATION to coordinator
    status = rng.choice(["in_progress", "done", "blocked"])
    exec_msg = SlipMessage(
        conv_id=conv_id,
        turn=3,
        src=2,  # executor
        dst=0,  # coordinator
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

    # 4) Critic EVAL/EVALUATION to coordinator
    # simple score based on status
    if status == "done":
        score = rng.randint(8, 10)
        eval_tag = "ok"
    elif status == "in_progress":
        score = rng.randint(5, 7)
        eval_tag = "partial"
    else:  # blocked
        score = rng.randint(0, 4)
        eval_tag = "blocked"

    critic_msg = SlipMessage(
        conv_id=conv_id,
        turn=4,
        src=3,  # critic
        dst=0,  # coordinator
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


# ---------- Prompt builders ----------

def build_planner_example(conv: Dict[str, object]) -> Tuple[str, str]:
    """
    Train planner: REQUEST/TASK -> PROPOSE/PLAN
    """
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
    """
    Train executor: PROPOSE/PLAN -> INFORM/OBSERVATION
    """
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
    """
    Train critic: INFORM/OBSERVATION -> EVAL/EVALUATION
    """
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
    """
    Optional: train coordinator to turn human goal into REQUEST/TASK.
    """
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


# ---------- Dataset generation ----------

def generate_dataset(
    num_conversations: int,
    out_path: Path,
    seed: int = 0,
    include_coordinator: bool = True,
) -> None:
    rng = random.Random(seed)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    roles = ["planner", "executor", "critic"]
    if include_coordinator:
        roles.insert(0, "coordinator")

    with out_path.open("w", encoding="utf-8") as f:
        for conv_id in range(1, num_conversations + 1):
            conv = make_conversation(conv_id, rng)

            # Coordinator example
            if include_coordinator:
                inp, tgt = build_coordinator_example(conv)
                row = {"role": "coordinator", "input": inp, "target": tgt}
                f.write(json.dumps(row, ensure_ascii=False))
                f.write("\n")

            # Planner
            inp, tgt = build_planner_example(conv)
            row = {"role": "planner", "input": inp, "target": tgt}
            f.write(json.dumps(row, ensure_ascii=False))
            f.write("\n")

            # Executor
            inp, tgt = build_executor_example(conv)
            row = {"role": "executor", "input": inp, "target": tgt}
            f.write(json.dumps(row, ensure_ascii=False))
            f.write("\n")

            # Critic
            inp, tgt = build_critic_example(conv)
            row = {"role": "critic", "input": inp, "target": tgt}
            f.write(json.dumps(row, ensure_ascii=False))
            f.write("\n")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate synthetic nSLIP finetune dataset (JSONL)."
    )
    p.add_argument(
        "--num-conversations",
        type=int,
        default=1000,
        help="Number of synthetic conversations to generate (default: 1000).",
    )
    p.add_argument(
        "--output",
        type=str,
        default="data/finetune/nslip_pairs.generated.jsonl",
        help="Output JSONL path (default: data/finetune/nslip_pairs.generated.jsonl).",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed (default: 0).",
    )
    p.add_argument(
        "--no-coordinator",
        action="store_true",
        help="If set, do not include coordinator examples.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_path = Path(args.output)
    generate_dataset(
        num_conversations=args.num_conversations,
        out_path=out_path,
        seed=args.seed,
        include_coordinator=not args.no_coordinator,
    )
    print(f"Wrote dataset to {out_path}")


if __name__ == "__main__":
    main()
