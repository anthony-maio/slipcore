"""
LLM-Enhanced Slipstream v3 Dataset Generator.

Generates high-quality Think-Quantize-Transmit training data using LLM APIs.
More diverse, realistic examples than template-based generation.

Wire format: SLIP v3 <src> <dst> <Force> <Object> [payload...]

Supports:
- Anthropic Claude API (claude-sonnet-4, claude-haiku)
- Google Gemini API (gemini-2.5-flash, gemini-2.0-flash)
- OpenAI API (gpt-4o-mini, gpt-4o)
- OpenRouter API (qwen/qwen-2.5-72b-instruct, and hundreds more)
- Together.ai API
- Fireworks.ai API
- DeepSeek API
- Any OpenAI-compatible endpoint

Usage:
    # With Claude (1000 examples)
    python -m slipcore.finetune_llm -n 1000 --provider anthropic --model claude-sonnet-4-20250514

    # With Gemini 2.5 Flash (1000 examples)
    python -m slipcore.finetune_llm -n 1000 --provider gemini --model gemini-2.5-flash

    # With OpenRouter (cheapest bulk option)
    python -m slipcore.finetune_llm -n 50000 --provider openrouter --model qwen/qwen-2.5-72b-instruct

    # Combined dataset: 1000 Claude + 1000 Gemini
    python -m slipcore.finetune_llm -n 1000 --provider anthropic -o train_claude.jsonl
    python -m slipcore.finetune_llm -n 1000 --provider gemini -o train_gemini.jsonl
    cat train_claude.jsonl train_gemini.jsonl > train_combined.jsonl

    # With TQT (Think-Quantize-Transmit) format
    python -m slipcore.finetune_llm -n 1000 -f sharegpt_thought --provider gemini

    # Resume after crash (picks up where it left off)
    python -m slipcore.finetune_llm -n 50000 --provider openrouter --resume
"""

from __future__ import annotations

import json
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from .finetune import SYSTEM_PROMPT_BASIC, SYSTEM_PROMPT_DETAILED

# ============ LLM Provider Configs ============

PROVIDERS = {
    "anthropic": {
        "base_url": "https://api.anthropic.com",
        "env_key": "ANTHROPIC_API_KEY",
        "default_model": "claude-haiku-4-5",
    },
    "gemini": {
        "base_url": "https://generativelanguage.googleapis.com/v1beta",
        "env_key": "GEMINI_API_KEY",
        "default_model": "gemini-2.5-flash",
    },
    "openai": {
        "base_url": "https://api.openai.com/v1",
        "env_key": "OPENAI_API_KEY",
        "default_model": "gpt-4o-mini",
    },
    "together": {
        "base_url": "https://api.together.xyz/v1",
        "env_key": "TOGETHER_API_KEY",
        "default_model": "meta-llama/Llama-3.3-70B-Instruct-Turbo",
    },
    "fireworks": {
        "base_url": "https://api.fireworks.ai/inference/v1",
        "env_key": "FIREWORKS_API_KEY",
        "default_model": "accounts/fireworks/models/llama-v3p1-70b-instruct",
    },
    "deepseek": {
        "base_url": "https://api.deepseek.com/v1",
        "env_key": "DEEPSEEK_API_KEY",
        "default_model": "deepseek-chat",
    },
    "openrouter": {
        "base_url": "https://openrouter.ai/api/v1",
        "env_key": "OPENROUTER_API_KEY",
        "default_model": "qwen/qwen-2.5-72b-instruct",
    },
}


# ============ Generation Prompts ============

SCENARIO_GENERATION_PROMPT = """You are generating training data for the Slipstream v3 protocol - a semantic quantization system for multi-agent AI coordination.

Slipstream v3 uses a factorized intent model: Force (action verb) + Object (domain noun).

Wire format: SLIP v3 <src> <dst> <Force> <Object> [payload...]

Generate {batch_size} diverse, realistic scenarios where AI agents need to communicate. For each scenario, provide:
1. A natural language instruction (what a user might say)
2. The agent's internal thought/reasoning about this communication
3. The source agent name (alphanumeric only, 1-20 chars)
4. The destination agent name (alphanumeric only, 1-20 chars)
5. The appropriate Force token (action verb)
6. The appropriate Object token (domain noun)
7. Optional payload tokens (alphanumeric only)

Force tokens (12 closed vocabulary - use EXACTLY one):
- Observe: Passively notice state/change/error
- Inform: Report information (status, completion, blockage, progress)
- Ask: Request information (clarification, status, permission)
- Request: Ask for action (task, review, help, plan)
- Propose: Suggest something (plan, change, alternative)
- Commit: Commit to something (task, deadline, resource)
- Eval: Evaluate work (approve, needs work)
- Meta: Protocol-level (acknowledge, sync, handoff, escalate)
- Accept: Accept a proposal/request
- Reject: Decline a proposal/request
- Error: Report system error (timeout, resource, permission, validation)
- Fallback: Content too specific for standard tokens

Object tokens (use one of these):
- State, Change, Error, Result, Status, Complete, Blocked, Progress
- Clarify, Permission, Resource, Task, Plan, Review, Help
- Cancel, Priority, Alternative, Rollback, Deadline
- Approve, NeedsWork, Ack, Sync, Handoff, Escalate, Abort
- Condition, Defer, Timeout, Validation, Generic

IMPORTANT: Generate diverse scenarios across different domains:
- Software development (code reviews, deployments, debugging)
- Data science (model training, data pipelines, experiments)
- DevOps (infrastructure, monitoring, incidents)
- Product management (features, priorities, stakeholders)
- Research (papers, experiments, findings)
- Creative work (designs, content, feedback)

Use realistic agent names like: alice, bob, coordinator, planner, executor, reviewer, team, manager, devops, mlEngineer, frontend, backend, qa, architect, etc.

Output as JSON array:
[
  {{
    "instruction": "Tell the ML team to retrain the model with the new dataset",
    "thought": "The new dataset is ready and I need the ML team to update the model",
    "src": "dataEngineer",
    "dst": "mlTeam",
    "force": "Request",
    "obj": "Task",
    "payload": ["retrain", "newDataset"]
  }},
  {{
    "instruction": "Check if the kubernetes pods are healthy",
    "thought": "This is a specific infrastructure check that doesn't map to standard tokens",
    "src": "devops",
    "dst": "infra",
    "force": "Fallback",
    "obj": "Generic",
    "payload": ["refK8sHealth"]
  }},
  ...
]

Generate exactly {batch_size} examples. Be creative and diverse!
Include some Fallback examples for complex/domain-specific requests that don't fit standard tokens.
All payload tokens must be alphanumeric only (no underscores, hyphens, or special characters)."""


VALIDATION_PROMPT = """Validate this Slipstream v3 training example. Check:
1. The Force+Object correctly match the intent
2. The wire format is correct: SLIP v3 <src> <dst> <Force> <Object> [payload...]
3. Force is one of: Observe, Inform, Ask, Request, Propose, Commit, Eval, Meta, Accept, Reject, Error, Fallback
4. The instruction is natural and clear

Example:
Instruction: "{instruction}"
Expected output: "{wire}"

Is this correct? Reply with just "VALID" or "INVALID: <reason>" """


# ============ API Clients ============

def call_anthropic(messages: list[dict], model: str, api_key: str) -> str:
    """Call Anthropic Claude API."""
    import httpx

    # Extract system message if present
    system = None
    chat_messages = []
    for msg in messages:
        if msg["role"] == "system":
            system = msg["content"]
        else:
            chat_messages.append(msg)

    payload = {
        "model": model,
        "max_tokens": 4096,
        "messages": chat_messages,
    }
    if system:
        payload["system"] = system

    response = httpx.post(
        "https://api.anthropic.com/v1/messages",
        headers={
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        },
        json=payload,
        timeout=120,
    )
    response.raise_for_status()
    return response.json()["content"][0]["text"]


def call_openai_compatible(
    messages: list[dict],
    model: str,
    api_key: str,
    base_url: str,
) -> str:
    """Call OpenAI-compatible API (OpenAI, Together, Fireworks, etc.)."""
    import httpx

    response = httpx.post(
        f"{base_url}/chat/completions",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": model,
            "messages": messages,
            "max_tokens": 4096,
            "temperature": 0.8,
        },
        timeout=120,
    )
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]


def call_gemini(messages: list[dict], model: str, api_key: str) -> str:
    """Call Google Gemini API."""
    import httpx

    system_instruction = None
    contents = []

    for msg in messages:
        role = msg["role"]
        content = msg["content"]

        if role == "system":
            system_instruction = content
        elif role == "user":
            contents.append({
                "role": "user",
                "parts": [{"text": content}]
            })
        elif role == "assistant":
            contents.append({
                "role": "model",
                "parts": [{"text": content}]
            })

    payload = {
        "contents": contents,
        "generationConfig": {
            "temperature": 0.8,
            "maxOutputTokens": 4096,
        }
    }

    if system_instruction:
        payload["systemInstruction"] = {
            "parts": [{"text": system_instruction}]
        }

    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"

    response = httpx.post(
        url,
        headers={"Content-Type": "application/json"},
        json=payload,
        timeout=120,
    )
    response.raise_for_status()

    result = response.json()

    try:
        return result["candidates"][0]["content"]["parts"][0]["text"]
    except (KeyError, IndexError) as e:
        raise ValueError(f"Unexpected Gemini response format: {result}") from e


def call_llm(
    messages: list[dict],
    provider: str,
    model: str,
    api_key: str,
) -> str:
    """Unified LLM caller."""
    if provider == "anthropic":
        return call_anthropic(messages, model, api_key)
    elif provider == "gemini":
        return call_gemini(messages, model, api_key)
    else:
        base_url = PROVIDERS[provider]["base_url"]
        return call_openai_compatible(messages, model, api_key, base_url)


# ============ Dataset Generation ============

VALID_FORCES = frozenset({
    "Observe", "Inform", "Ask", "Request", "Propose", "Commit",
    "Eval", "Meta", "Accept", "Reject", "Error", "Fallback",
})

@dataclass
class LLMExample:
    """A training example generated by LLM."""
    instruction: str
    src: str
    dst: str
    force: str
    obj: str
    payload: list[str] = field(default_factory=list)
    wire: str = ""
    thought: str = ""


def _clean_token(s: str) -> str:
    """Remove non-alphanumeric chars from a token."""
    return "".join(c for c in s if c.isalnum())


def generate_batch(
    batch_size: int,
    provider: str,
    model: str,
    api_key: str,
) -> list[LLMExample]:
    """Generate a batch of examples using LLM."""
    prompt = SCENARIO_GENERATION_PROMPT.format(batch_size=batch_size)

    messages = [
        {"role": "user", "content": prompt}
    ]

    response = call_llm(messages, provider, model, api_key)

    # Parse JSON from response
    text = response.strip()

    # Remove markdown code blocks
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:])
        if text.rstrip().endswith("```"):
            text = text.rstrip()[:-3]

    import re
    match = re.search(r'\[[\s\S]*\]', text)
    if match:
        text = match.group()

    # Fix common JSON issues from LLMs
    text = re.sub(r',\s*([}\]])', r'\1', text)
    text = re.sub(r'[\x00-\x1f]', ' ', text)

    try:
        scenarios = json.loads(text)
    except json.JSONDecodeError as e:
        scenarios = []
        try:
            obj_matches = re.findall(r'\{[^{}]*\}', text)
            for obj_str in obj_matches:
                try:
                    obj = json.loads(obj_str)
                    if "instruction" in obj and "force" in obj:
                        scenarios.append(obj)
                except json.JSONDecodeError:
                    continue
            if scenarios:
                print(f"Partial recovery: {len(scenarios)} examples from malformed JSON")
        except Exception:
            pass

        if not scenarios:
            print(f"JSON parse error: {e}")
            return []

    examples = []
    for scenario in scenarios:
        try:
            force = scenario["force"]
            obj = scenario.get("obj", "Generic")

            # Validate force token
            if force not in VALID_FORCES:
                # Try to fix common casing issues
                for vf in VALID_FORCES:
                    if vf.lower() == force.lower():
                        force = vf
                        break
                else:
                    print(f"Skipping unknown force: {force}")
                    continue

            # Clean agent IDs
            src = _clean_token(scenario.get("src", "agent"))[:20] or "agent"
            dst = _clean_token(scenario.get("dst", "other"))[:20] or "other"

            # Clean payload
            raw_payload = scenario.get("payload", [])
            if isinstance(raw_payload, str):
                raw_payload = [raw_payload] if raw_payload else []
            payload = [_clean_token(p) for p in raw_payload if _clean_token(p)]

            # Build v3 wire format
            wire_parts = ["SLIP", "v3", src, dst, force, obj]
            wire_parts.extend(payload)
            wire = " ".join(wire_parts)

            thought = scenario.get("thought", "")
            if not thought:
                thought = f"I need to {force.lower()} {obj.lower()} to {dst}"

            examples.append(LLMExample(
                instruction=scenario["instruction"],
                src=src,
                dst=dst,
                force=force,
                obj=obj,
                payload=payload,
                wire=wire,
                thought=thought,
            ))
        except (KeyError, TypeError) as e:
            print(f"Skipping malformed scenario: {e}")
            continue

    return examples


def to_sharegpt(example: LLMExample, system_prompt: str) -> dict:
    """Convert to ShareGPT format."""
    return {
        "conversations": [
            {"from": "system", "value": system_prompt},
            {"from": "human", "value": example.instruction},
            {"from": "gpt", "value": example.wire},
        ]
    }


def to_chat(example: LLMExample, system_prompt: str) -> dict:
    """Convert to chat format."""
    return {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": example.instruction},
            {"role": "assistant", "content": example.wire},
        ]
    }


def to_alpaca(example: LLMExample) -> dict:
    """Convert to Alpaca format."""
    return {
        "instruction": f"Communicate using Slipstream v3 protocol: {example.instruction}",
        "input": "",
        "output": example.wire,
    }


# ============ Think-Quantize-Transmit Converters ============

def to_sharegpt_thought(example: LLMExample, system_prompt: str) -> dict:
    """Convert to ShareGPT format with THOUGHT supervision (recommended for TQT training)."""
    response = f"THOUGHT: {example.thought}\nSLIP: {example.wire}"
    return {
        "conversations": [
            {"from": "system", "value": system_prompt},
            {"from": "human", "value": example.instruction},
            {"from": "gpt", "value": response},
        ]
    }


def to_chat_thought(example: LLMExample, system_prompt: str) -> dict:
    """Convert to chat format with THOUGHT supervision."""
    response = f"THOUGHT: {example.thought}\nSLIP: {example.wire}"
    return {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": example.instruction},
            {"role": "assistant", "content": response},
        ]
    }


def to_sharegpt_semantics(example: LLMExample, system_prompt: str) -> dict:
    """Convert to ShareGPT with full semantic annotation (maximum supervision)."""
    response = (
        f"THOUGHT: {example.thought}\n"
        f"QUANTIZE: Force={example.force} Object={example.obj}\n"
        f"SLIP: {example.wire}"
    )
    return {
        "conversations": [
            {"from": "system", "value": system_prompt},
            {"from": "human", "value": example.instruction},
            {"from": "gpt", "value": response},
        ]
    }


def _count_existing_lines(path: Path) -> int:
    """Count the number of lines in an existing file.

    Args:
        path: Path to the file.

    Returns:
        Number of lines, or 0 if the file does not exist.
    """
    if not path.exists():
        return 0
    count = 0
    with open(path, "r", encoding="utf-8") as f:
        for _ in f:
            count += 1
    return count


def _format_duration(seconds: float) -> str:
    """Format seconds into a human-readable duration string.

    Args:
        seconds: Duration in seconds.

    Returns:
        Formatted string like '1h 23m 45s' or '5m 12s'.
    """
    if seconds < 0:
        return "0s"
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    if h > 0:
        return f"{h}h {m:02d}m {s:02d}s"
    elif m > 0:
        return f"{m}m {s:02d}s"
    else:
        return f"{s}s"


def generate_dataset_llm(
    output_path: Path,
    num_examples: int = 1000,
    provider: str = "anthropic",
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    format: str = "sharegpt_thought",
    system_prompt: str = SYSTEM_PROMPT_BASIC,
    batch_size: int = 25,
    max_workers: int = 4,
    delay_between_batches: float = 0.5,
    resume: bool = False,
) -> int:
    """Generate a high-quality v3 Think-Quantize-Transmit dataset using LLM APIs.

    Args:
        output_path: Where to save the JSONL file.
        num_examples: Target number of examples.
        provider: API provider (anthropic, gemini, openai, together, fireworks,
            deepseek, openrouter).
        model: Model name (uses provider default if not specified).
        api_key: API key (reads from env if not specified).
        format: Output format:
            - Basic: sharegpt, chat, alpaca (instruction -> SLIP v3)
            - TQT: sharegpt_thought, chat_thought, sharegpt_semantics (includes THOUGHT)
        system_prompt: System prompt for training examples.
        batch_size: Examples per API call.
        max_workers: Parallel API calls.
        delay_between_batches: Seconds between batches (rate limiting).
        resume: If True and the output file already exists, count existing lines
            and continue generating from where it left off. The file is opened
            in append mode so no existing data is lost.

    Returns:
        Total number of examples in the output file (existing + newly generated).
    """
    if provider not in PROVIDERS:
        raise ValueError(f"Unknown provider: {provider}. Choose from: {list(PROVIDERS.keys())}")

    if api_key is None:
        env_key = PROVIDERS[provider]["env_key"]
        api_key = os.environ.get(env_key)
        if not api_key:
            raise ValueError(f"No API key provided. Set {env_key} environment variable or pass --api-key")

    if model is None:
        model = PROVIDERS[provider]["default_model"]

    # Handle resume: count existing lines and adjust target
    existing_count = 0
    file_mode = "w"
    if resume and output_path.exists():
        existing_count = _count_existing_lines(output_path)
        if existing_count >= num_examples:
            print(f"Resume: {output_path} already has {existing_count} examples "
                  f"(target: {num_examples}). Nothing to do.")
            return existing_count
        file_mode = "a"
        print(f"Resume: found {existing_count} existing examples in {output_path}")

    remaining = num_examples - existing_count

    print(f"Generating {remaining} v3 examples using {provider}/{model}"
          + (f" (resuming from {existing_count})" if existing_count > 0 else ""))
    print(f"Batch size: {batch_size}, Workers: {max_workers}")

    num_batches = (remaining + batch_size - 1) // batch_size

    all_examples: list[LLMExample] = []

    converters = {
        "sharegpt": lambda e: to_sharegpt(e, system_prompt),
        "chat": lambda e: to_chat(e, system_prompt),
        "alpaca": to_alpaca,
        "sharegpt_thought": lambda e: to_sharegpt_thought(e, system_prompt),
        "chat_thought": lambda e: to_chat_thought(e, system_prompt),
        "sharegpt_semantics": lambda e: to_sharegpt_semantics(e, system_prompt),
    }
    converter = converters.get(format, converters["sharegpt_thought"])

    start_time = time.monotonic()
    completed_batches = 0

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for i in range(num_batches):
            if len(all_examples) >= remaining:
                break

            future = executor.submit(
                generate_batch,
                batch_size=batch_size,
                provider=provider,
                model=model,
                api_key=api_key,
            )
            futures.append(future)

            time.sleep(delay_between_batches)

        for i, future in enumerate(as_completed(futures)):
            try:
                batch = future.result()
                all_examples.extend(batch)
                completed_batches += 1
                total_so_far = existing_count + len(all_examples)
                print(f"Batch {completed_batches}/{num_batches}: "
                      f"+{len(batch)} examples (total: {total_so_far})")

                # Progress report every 10 batches
                if completed_batches % 10 == 0:
                    elapsed = time.monotonic() - start_time
                    examples_generated = len(all_examples)
                    if examples_generated > 0:
                        rate = examples_generated / elapsed
                        examples_left = remaining - examples_generated
                        eta = examples_left / rate if rate > 0 else 0
                        print(f"  -- Progress: {total_so_far}/{num_examples} examples | "
                              f"Elapsed: {_format_duration(elapsed)} | "
                              f"Rate: {rate:.1f} ex/s | "
                              f"ETA: {_format_duration(eta)}")

            except Exception as e:
                completed_batches += 1
                print(f"Batch {completed_batches}/{num_batches} failed: {e}")

    random.shuffle(all_examples)
    all_examples = all_examples[:remaining]

    with open(output_path, file_mode, encoding="utf-8") as f:
        for example in all_examples:
            formatted = converter(example)
            f.write(json.dumps(formatted, ensure_ascii=False) + "\n")

    total_count = existing_count + len(all_examples)
    elapsed_total = time.monotonic() - start_time
    print(f"\nGenerated {len(all_examples)} new v3 examples to {output_path} "
          f"in {_format_duration(elapsed_total)}")
    if existing_count > 0:
        print(f"Total examples in file: {total_count}")
    return total_count


# ============ CLI ============

def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate high-quality Slipstream v3 dataset using LLM APIs"
    )
    parser.add_argument(
        "-o", "--output",
        type=Path,
        default=Path("slipstream_train_llm.jsonl"),
    )
    parser.add_argument(
        "-n", "--num-examples",
        type=int,
        default=1000,
    )
    parser.add_argument(
        "--provider",
        choices=list(PROVIDERS.keys()),
        default="anthropic",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
    )
    parser.add_argument(
        "-f", "--format",
        choices=[
            "chat", "alpaca", "sharegpt",
            "chat_thought", "sharegpt_thought", "sharegpt_semantics",
        ],
        default="sharegpt_thought",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=25,
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--detailed-prompt",
        action="store_true",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume generation if output file already exists. "
             "Counts existing lines and continues from where it left off.",
    )

    args = parser.parse_args()

    system_prompt = SYSTEM_PROMPT_DETAILED if args.detailed_prompt else SYSTEM_PROMPT_BASIC

    try:
        count = generate_dataset_llm(
            output_path=args.output,
            num_examples=args.num_examples,
            provider=args.provider,
            model=args.model,
            api_key=args.api_key,
            format=args.format,
            system_prompt=system_prompt,
            batch_size=args.batch_size,
            max_workers=args.workers,
            resume=args.resume,
        )
        print(f"\nSuccess! Generated {count} v3 examples")
        print("Ready for finetuning with Unsloth or similar tools")
    except Exception as e:
        print(f"Error: {e}")
        raise


if __name__ == "__main__":
    main()
