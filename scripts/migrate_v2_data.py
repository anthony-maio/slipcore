"""Migrate v2 training data to v3 Force-Object format.

Converts SLIP v1 flat-anchor wire messages to SLIP v3 factorized wire messages.

Usage:
    python scripts/migrate_v2_data.py data/slipstream-tqt.jsonl data/slipstream-tqt-v3.jsonl
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

# Add src to path so we can import slipcore
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from slipcore.intent import V2_TO_V3
from slipcore.finetune import SYSTEM_PROMPT_BASIC


def migrate_wire(wire: str) -> str | None:
    """Convert a v1 wire message to v3 format.

    Returns None if the wire can't be migrated.
    """
    parts = wire.strip().split()
    if len(parts) < 4:
        return None

    prefix, version = parts[0], parts[1]
    if prefix != "SLIP" or version != "v1":
        return None

    src, dst = parts[2], parts[3]
    anchor = parts[4] if len(parts) > 4 else None
    payload = parts[5:] if len(parts) > 5 else []

    if not anchor:
        return None

    pair = V2_TO_V3.get(anchor)
    if pair is None:
        # Try common variations
        for v2_mnemonic, v3_pair in V2_TO_V3.items():
            if v2_mnemonic.lower() == anchor.lower():
                pair = v3_pair
                break
        if pair is None:
            return None

    force, obj = pair

    # Clean payload tokens (v2 allowed underscores, colons; v3 is alphanumeric only)
    clean_payload = []
    for token in payload:
        clean = "".join(c for c in token if c.isalnum())
        if clean:
            clean_payload.append(clean[:30])  # MAX_PAYLOAD_TOKEN_LEN

    v3_parts = ["SLIP", "v3", src, dst, force, obj]
    v3_parts.extend(clean_payload[:20])  # MAX_PAYLOAD_TOKENS
    return " ".join(v3_parts)


def migrate_quantize_line(quantize_line: str, force: str, obj: str) -> str:
    """Convert a v2 QUANTIZE line to v3 format."""
    # v2: QUANTIZE: [ACTION=request | DOMAIN=plan | URGENCY=elevated | POLARITY=neutral] -> RequestPlan
    # v3: QUANTIZE: Force=Request Object=Plan
    return f"QUANTIZE: Force={force} Object={obj}"


def migrate_system_prompt(prompt: str) -> str:
    """Replace v1/v2 system prompts with v3."""
    if "SLIP v1" in prompt or "SLIP" in prompt and "v1" in prompt:
        return SYSTEM_PROMPT_BASIC
    return prompt


def migrate_gpt_response(response: str) -> str | None:
    """Migrate an assistant/gpt response from v1 to v3 format."""
    lines = response.strip().split("\n")
    new_lines = []
    force = None
    obj = None

    for line in lines:
        stripped = line.strip()

        # Handle SLIP wire line
        if stripped.startswith("SLIP:") or stripped.startswith("SLIP "):
            # Extract the wire part
            if stripped.startswith("SLIP:"):
                wire_part = stripped[len("SLIP:"):].strip()
            else:
                wire_part = stripped

            migrated = migrate_wire(wire_part)
            if migrated is None:
                return None

            # Extract force/obj from migrated wire for QUANTIZE fixup
            mparts = migrated.split()
            if len(mparts) >= 6:
                force = mparts[4]
                obj = mparts[5]

            if stripped.startswith("SLIP:"):
                new_lines.append(f"SLIP: {migrated}")
            else:
                new_lines.append(migrated)

        # Handle QUANTIZE line
        elif stripped.startswith("QUANTIZE:"):
            # We'll fix this after we know force/obj
            new_lines.append("__QUANTIZE_PLACEHOLDER__")

        # Handle THOUGHT line (passthrough)
        elif stripped.startswith("THOUGHT:"):
            new_lines.append(stripped)

        else:
            new_lines.append(stripped)

    # Fix up QUANTIZE placeholder
    if force and obj:
        new_lines = [
            migrate_quantize_line(l, force, obj) if l == "__QUANTIZE_PLACEHOLDER__" else l
            for l in new_lines
        ]
    else:
        new_lines = [l for l in new_lines if l != "__QUANTIZE_PLACEHOLDER__"]

    return "\n".join(new_lines)


def migrate_example(example: dict) -> dict | None:
    """Migrate a single training example from v2 to v3."""
    if "conversations" in example:
        new_convos = []
        for turn in example["conversations"]:
            role = turn.get("from", turn.get("role", ""))
            value = turn.get("value", turn.get("content", ""))

            if role == "system":
                new_convos.append({"from": "system", "value": migrate_system_prompt(value)})
            elif role in ("human", "user"):
                new_convos.append({"from": "human", "value": value})
            elif role in ("gpt", "assistant"):
                migrated = migrate_gpt_response(value)
                if migrated is None:
                    return None
                new_convos.append({"from": "gpt", "value": migrated})
            else:
                new_convos.append(turn)

        return {"conversations": new_convos}

    elif "messages" in example:
        new_msgs = []
        for msg in example["messages"]:
            role = msg["role"]
            content = msg["content"]

            if role == "system":
                new_msgs.append({"role": "system", "content": migrate_system_prompt(content)})
            elif role == "user":
                new_msgs.append({"role": "user", "content": content})
            elif role == "assistant":
                migrated = migrate_gpt_response(content)
                if migrated is None:
                    return None
                new_msgs.append({"role": "assistant", "content": migrated})
            else:
                new_msgs.append(msg)

        return {"messages": new_msgs}

    elif "instruction" in example and "output" in example:
        migrated = migrate_gpt_response(example["output"])
        if migrated is None:
            return None
        return {
            "instruction": example["instruction"],
            "input": example.get("input", ""),
            "output": migrated,
        }

    return None


def main() -> None:
    if len(sys.argv) < 3:
        print("Usage: python scripts/migrate_v2_data.py <input.jsonl> <output.jsonl>")
        sys.exit(1)

    input_path = Path(sys.argv[1])
    output_path = Path(sys.argv[2])

    if not input_path.exists():
        print(f"Input file not found: {input_path}")
        sys.exit(1)

    migrated = 0
    skipped = 0
    total = 0

    with open(input_path, encoding="utf-8") as fin, \
         open(output_path, "w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            total += 1

            try:
                example = json.loads(line)
            except json.JSONDecodeError:
                skipped += 1
                continue

            result = migrate_example(example)
            if result is None:
                skipped += 1
                continue

            fout.write(json.dumps(result, ensure_ascii=False) + "\n")
            migrated += 1

    print(f"Migration complete:")
    print(f"  Total:    {total}")
    print(f"  Migrated: {migrated}")
    print(f"  Skipped:  {skipped}")
    print(f"  Output:   {output_path}")


if __name__ == "__main__":
    main()
