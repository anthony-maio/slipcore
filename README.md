# SLIPCore – Streamlined Intragent Protocol

SLIPCore provides a compact, schema-driven protocol for **agent-to-agent messages** inside an agentic system.

It is designed for:
- LLM-based agents (planner, executor, critic, memory, etc.)
- A coordinating agent or orchestrator
- Frameworks like LangGraph, LangChain, custom graphs

Natural language lives at the *edges* (human ↔ system). Inside, agents speak a small, regular, ID-based protocol that's easy to parse and cheap in tokens.

## Key ideas

- **Acts**: A small closed set of speech acts (REQUEST, PROPOSE, EVAL, etc.).
- **Frames**: Task/plan/observation/evaluation frames with typed slots.
- **IDs not blobs**: Goals, tasks, artifacts referenced by integer IDs.
- **nSLIP wire form**: A compact string like `@a3|f0|cB|s0|d1|t2|gK#` for LLM prompts/logs.
- **Python-first**: Internally, you manipulate `SlipMessage` dataclasses, not strings.

## Quickstart

```bash
pip install -e .
```

```python
from slipcore.protocol import SlipMessage, Act, FrameType, Slot, encode_message, decode_message

msg = SlipMessage(
    conv_id=3,
    turn=1,
    src=0,  # coordinator
    dst=1,  # planner
    act=Act.REQUEST,
    frame=FrameType.TASK,
    slots={
        Slot.GOAL_ID: 17,
        Slot.TASK_ID: 42,
        Slot.PRIORITY: 2,
        Slot.TAG: "refactor_auth",
    },
)

wire = encode_message(msg)
print("Wire:", wire)

parsed = decode_message(wire)
print("Parsed:", parsed)
```

## Integration pattern (LangGraph / other)

* Add a `wire.slip: str` field in your graph state.
* Nodes decode `wire.slip` → `SlipMessage`, inspect/update shared stores (goals, tasks).
* Agents (LLM-backed) see prompts built from `SlipMessage` + shared state and produce new `SlipMessage`s.
* Encode with `encode_message` and write back to `wire.slip`.

See `examples/` for minimal demos.

## Finetuning / PEFT

`data/finetune/nslip_pairs.sample.jsonl` shows how to construct training pairs:

* `input`: model sees prompt + previous context.
* `target`: expected nSLIP message string.

You can generate many synthetic task/plan/eval cycles to fine-tune specific agents (planner, executor, critic) using LoRA / QLoRA / other PEFT methods.

## Generate Training Data

```bash
python -m slipcore.generate_dataset --num-conversations 1000 --output data/finetune/nslip_pairs.generated.jsonl
```
