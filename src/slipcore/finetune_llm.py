"""
LLM-Enhanced Slipstream Dataset Generator

Generates high-quality Think-Quantize-Transmit training data using LLM APIs.
More diverse, realistic examples than template-based generation.

Supports:
- Anthropic Claude API (claude-sonnet-4, claude-haiku)
- Google Gemini API (gemini-2.5-flash, gemini-2.0-flash)
- OpenAI API (gpt-4o-mini, gpt-4o)
- Together.ai API
- Any OpenAI-compatible endpoint

Usage:
    # With Claude (1000 examples)
    python -m slipcore.finetune_llm -n 1000 --provider anthropic --model claude-sonnet-4-20250514

    # With Gemini 2.5 Flash (1000 examples)
    python -m slipcore.finetune_llm -n 1000 --provider gemini --model gemini-2.5-flash

    # Combined dataset: 1000 Claude + 1000 Gemini
    python -m slipcore.finetune_llm -n 1000 --provider anthropic -o train_claude.jsonl
    python -m slipcore.finetune_llm -n 1000 --provider gemini -o train_gemini.jsonl
    cat train_claude.jsonl train_gemini.jsonl > train_combined.jsonl

    # With TQT (Think-Quantize-Transmit) format
    python -m slipcore.finetune_llm -n 1000 -f sharegpt_thought --provider gemini
"""

from __future__ import annotations
import json
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Literal
from concurrent.futures import ThreadPoolExecutor, as_completed

from .ucr import get_default_ucr, UCRAnchor
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
}


# ============ Generation Prompts ============

SCENARIO_GENERATION_PROMPT = """You are generating training data for the Slipstream protocol - a semantic quantization system for multi-agent AI coordination.

Generate {batch_size} diverse, realistic scenarios where AI agents need to communicate. For each scenario, provide:
1. A natural language instruction (what a user might say)
2. The agent's internal thought/reasoning about this communication
3. The source agent name
4. The destination agent name
5. The appropriate Slipstream anchor (intent)
6. Optional payload (additional context)

Available anchors and their meanings:
- RequestTask: Ask another agent to perform a task
- RequestReview: Request code/document review
- RequestHelp: Ask for assistance
- RequestPlan: Request a plan be created
- InformComplete: Report task completion
- InformProgress: Share progress update
- InformBlocked: Report being blocked
- InformStatus: General status update
- ProposePlan: Suggest a plan
- ProposeChange: Suggest a modification
- ProposeAlternative: Offer an alternative approach
- EvalApprove: Approve something
- EvalReject: Reject something
- EvalNeedsWork: Request revisions
- Accept: Agree to request/proposal
- Reject: Decline request/proposal
- MetaAck: Acknowledge receipt
- MetaHandoff: Transfer responsibility
- Fallback: For complex unquantizable content

IMPORTANT: Generate diverse scenarios across different domains:
- Software development (code reviews, deployments, debugging)
- Data science (model training, data pipelines, experiments)
- DevOps (infrastructure, monitoring, incidents)
- Product management (features, priorities, stakeholders)
- Research (papers, experiments, findings)
- Creative work (designs, content, feedback)

Use realistic agent names like: alice, bob, coordinator, planner, executor, reviewer, team, manager, devops, ml_engineer, frontend, backend, qa, architect, etc.

Output as JSON array:
[
  {{
    "instruction": "Tell the ML team to retrain the model with the new dataset",
    "thought": "The new dataset is ready and I need the ML team to update the model",
    "src": "data_engineer",
    "dst": "ml_team",
    "anchor": "RequestTask",
    "payload": ["retrain", "new_dataset"]
  }},
  {{
    "instruction": "Check if the kubernetes pods are healthy",
    "thought": "This is a specific infrastructure check that doesn't map to standard anchors",
    "src": "devops",
    "dst": "infra",
    "anchor": "Fallback",
    "payload": ["check_kubernetes_pod_health"]
  }},
  ...
]

Generate exactly {batch_size} examples. Be creative and diverse!
Include some Fallback examples for complex/domain-specific requests that don't fit standard anchors."""


VALIDATION_PROMPT = """Validate this Slipstream training example. Check:
1. The anchor correctly matches the intent
2. The wire format is correct: SLIP v1 <src> <dst> <anchor> [payload...]
3. The instruction is natural and clear

Example:
Instruction: "{instruction}"
Expected output: "{wire}"

Is this correct? Reply with just "VALID" or "INVALID: <reason>"""


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
    """
    Call Google Gemini API.

    Supports gemini-2.5-flash, gemini-2.0-flash, gemini-1.5-pro, etc.
    """
    import httpx

    # Convert messages to Gemini format
    # Gemini uses "contents" with "parts" structure
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

    # Build request payload
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

    # Gemini API endpoint
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"

    response = httpx.post(
        url,
        headers={"Content-Type": "application/json"},
        json=payload,
        timeout=120,
    )
    response.raise_for_status()

    result = response.json()

    # Extract text from response
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

@dataclass
class LLMExample:
    """A training example generated by LLM."""
    instruction: str
    src: str
    dst: str
    anchor: str
    payload: list[str]
    wire: str
    thought: str = ""  # Natural language reasoning for TQT training


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

    # Parse JSON from response - handle various LLM output quirks
    text = response.strip()

    # Remove markdown code blocks
    if text.startswith("```"):
        lines = text.split("\n")
        # Remove first line (```json) and last line (```)
        text = "\n".join(lines[1:])
        if text.rstrip().endswith("```"):
            text = text.rstrip()[:-3]

    # Try to extract JSON array
    import re
    match = re.search(r'\[[\s\S]*\]', text)
    if match:
        text = match.group()

    # Fix common JSON issues from LLMs
    # 1. Remove trailing commas before ] or }
    text = re.sub(r',\s*([}\]])', r'\1', text)
    # 2. Fix unescaped newlines in strings (replace with \n)
    # 3. Remove control characters
    text = re.sub(r'[\x00-\x1f]', ' ', text)

    try:
        scenarios = json.loads(text)
    except json.JSONDecodeError as e:
        # Try line by line parsing for partial recovery
        scenarios = []
        try:
            # Attempt to parse each object individually
            obj_matches = re.findall(r'\{[^{}]*\}', text)
            for obj_str in obj_matches:
                try:
                    obj = json.loads(obj_str)
                    if "instruction" in obj and "anchor" in obj:
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
            payload = scenario.get("payload", [])
            if isinstance(payload, str):
                payload = [payload] if payload else []

            # Build wire format
            wire_parts = ["SLIP", "v1", scenario["src"], scenario["dst"], scenario["anchor"]]
            wire_parts.extend(payload)
            wire = " ".join(wire_parts)

            # Extract thought (generate one if not provided)
            thought = scenario.get("thought", "")
            if not thought:
                thought = f"I need to {scenario['anchor'].lower()} to {scenario['dst']}"

            examples.append(LLMExample(
                instruction=scenario["instruction"],
                src=scenario["src"],
                dst=scenario["dst"],
                anchor=scenario["anchor"],
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
        "instruction": f"Communicate using Slipstream protocol: {example.instruction}",
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
    # Infer UCR dimensions from anchor
    from .ucr import get_default_ucr
    from .finetune import _get_dimension_labels

    ucr = get_default_ucr()
    anchor_obj = ucr.get_by_mnemonic(example.anchor)

    dims = []
    if anchor_obj:
        labels = _get_dimension_labels(anchor_obj.coords)
        if labels.get("action"):
            dims.append(f"ACTION={labels['action']}")
        if labels.get("domain"):
            dims.append(f"DOMAIN={labels['domain']}")
        if labels.get("urgency"):
            dims.append(f"URGENCY={labels['urgency']}")
        if labels.get("polarity"):
            dims.append(f"POLARITY={labels['polarity']}")

    semantic_hint = " | ".join(dims) if dims else f"ANCHOR={example.anchor}"

    response = (
        f"THOUGHT: {example.thought}\n"
        f"QUANTIZE: [{semantic_hint}] -> {example.anchor}\n"
        f"SLIP: {example.wire}"
    )

    return {
        "conversations": [
            {"from": "system", "value": system_prompt},
            {"from": "human", "value": example.instruction},
            {"from": "gpt", "value": response},
        ]
    }


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
) -> int:
    """
    Generate a high-quality Think-Quantize-Transmit dataset using LLM APIs.

    Args:
        output_path: Where to save the JSONL file
        num_examples: Target number of examples
        provider: API provider (anthropic, gemini, openai, together, fireworks, deepseek)
        model: Model name (uses provider default if not specified)
        api_key: API key (reads from env if not specified)
        format: Output format:
            - Basic: sharegpt, chat, alpaca (instruction -> SLIP)
            - TQT: sharegpt_thought, chat_thought, sharegpt_semantics (includes THOUGHT)
        system_prompt: System prompt for training examples
        batch_size: Examples per API call
        max_workers: Parallel API calls
        delay_between_batches: Seconds between batches (rate limiting)

    Returns:
        Number of examples generated
    """
    if provider not in PROVIDERS:
        raise ValueError(f"Unknown provider: {provider}. Choose from: {list(PROVIDERS.keys())}")

    # Get API key
    if api_key is None:
        env_key = PROVIDERS[provider]["env_key"]
        api_key = os.environ.get(env_key)
        if not api_key:
            raise ValueError(f"No API key provided. Set {env_key} environment variable or pass --api-key")

    # Get model
    if model is None:
        model = PROVIDERS[provider]["default_model"]

    print(f"Generating {num_examples} examples using {provider}/{model}")
    print(f"Batch size: {batch_size}, Workers: {max_workers}")

    # Calculate batches needed
    num_batches = (num_examples + batch_size - 1) // batch_size

    all_examples: list[LLMExample] = []

    converters = {
        # Basic formats (instruction -> SLIP only)
        "sharegpt": lambda e: to_sharegpt(e, system_prompt),
        "chat": lambda e: to_chat(e, system_prompt),
        "alpaca": to_alpaca,
        # TQT formats (Think-Quantize-Transmit supervision)
        "sharegpt_thought": lambda e: to_sharegpt_thought(e, system_prompt),
        "chat_thought": lambda e: to_chat_thought(e, system_prompt),
        "sharegpt_semantics": lambda e: to_sharegpt_semantics(e, system_prompt),
    }
    converter = converters.get(format, converters["sharegpt_thought"])

    # Generate in batches
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for i in range(num_batches):
            if len(all_examples) >= num_examples:
                break

            future = executor.submit(
                generate_batch,
                batch_size=batch_size,
                provider=provider,
                model=model,
                api_key=api_key,
            )
            futures.append(future)

            # Rate limiting
            time.sleep(delay_between_batches)

        for i, future in enumerate(as_completed(futures)):
            try:
                batch = future.result()
                all_examples.extend(batch)
                print(f"Batch {i+1}/{num_batches}: +{len(batch)} examples (total: {len(all_examples)})")
            except Exception as e:
                print(f"Batch {i+1} failed: {e}")

    # Trim to exact count and shuffle
    random.shuffle(all_examples)
    all_examples = all_examples[:num_examples]

    # Write output
    with open(output_path, "w", encoding="utf-8") as f:
        for example in all_examples:
            formatted = converter(example)
            f.write(json.dumps(formatted, ensure_ascii=False) + "\n")

    print(f"\nGenerated {len(all_examples)} examples to {output_path}")
    return len(all_examples)


# ============ CLI ============

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate high-quality Slipstream dataset using LLM APIs"
    )
    parser.add_argument(
        "-o", "--output",
        type=Path,
        default=Path("slipstream_train_llm.jsonl"),
        help="Output file path",
    )
    parser.add_argument(
        "-n", "--num-examples",
        type=int,
        default=1000,
        help="Number of examples to generate",
    )
    parser.add_argument(
        "--provider",
        choices=list(PROVIDERS.keys()),
        default="anthropic",
        help="LLM provider",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model name (uses provider default if not specified)",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="API key (reads from environment if not specified)",
    )
    parser.add_argument(
        "-f", "--format",
        choices=[
            "chat", "alpaca", "sharegpt",  # Basic
            "chat_thought", "sharegpt_thought", "sharegpt_semantics",  # TQT
        ],
        default="sharegpt_thought",
        help="Output format. TQT formats (sharegpt_thought, sharegpt_semantics) recommended",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=25,
        help="Examples per API call",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Parallel API calls",
    )
    parser.add_argument(
        "--detailed-prompt",
        action="store_true",
        help="Use detailed system prompt",
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
        )
        print(f"\nSuccess! Generated {count} examples")
        print(f"Ready for finetuning with Unsloth or similar tools")
    except Exception as e:
        print(f"Error: {e}")
        raise


if __name__ == "__main__":
    main()
