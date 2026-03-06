# LangGraph Integration Guide

This is the fastest way to adopt Slipstream in a LangGraph platform without retraining models.

## 1. Install

```bash
pip install slipcore langgraph
```

## 2. Use the Adapter in Graph Nodes

Slipstream ships a dependency-free LangGraph adapter:

- `LangGraphSlipstreamAdapter`
- `make_encode_node(...)`
- `make_decode_node(...)`
- `make_force_router(...)`
- `make_force_object_router(...)`

```python
from typing import TypedDict
from langgraph.graph import START, END, StateGraph

from slipcore import (
    LangGraphSlipstreamAdapter,
    make_encode_node,
    make_decode_node,
    make_force_object_router,
)


class AgentState(TypedDict, total=False):
    thought: str
    src: str
    dst: str
    slip_wire: str
    slip_force: str
    slip_obj: str
    slip_fallback_text: str


adapter = LangGraphSlipstreamAdapter()
encode_node = make_encode_node(adapter)
decode_node = make_decode_node(adapter)
route = make_force_object_router(adapter)
```

## 3. Recommended Graph Topology

Use this structure:

1. Agent node produces natural-language intent (`thought`, `src`, `dst`).
2. `slip_encode` node converts state to `slip_wire`.
3. Transport/message bus sends `slip_wire`.
4. `slip_decode` node parses wire and resolves fallback refs.
5. Conditional edges route on `Force:Object`.

```python
builder = StateGraph(AgentState)
builder.add_node("slip_encode", encode_node)
builder.add_node("slip_decode", decode_node)
builder.add_conditional_edges(
    "slip_decode",
    route,
    {
        "Request:Review": "review_agent",
        "Inform:Status": "status_agent",
        "Fallback:Generic": "fallback_agent",
    },
)
```

## 4. Fallback Behavior (v3.1 strict)

If quantization confidence is low, the adapter emits:

```text
SLIP v3 <src> <dst> Fallback Generic <ref>
```

Raw text is stored out-of-band in the adapter's `FallbackStore`. On decode, `slip_fallback_text` is populated when the ref exists in store.

For distributed deployments, replace in-memory fallback storage with shared storage (Redis/DB/KV).

## 5. Rollout Plan

1. Shadow mode: emit Slipstream + current JSON for comparison.
2. Measure fallback rate and routing correctness.
3. Cut over internal agent-to-agent transport to Slipstream.
4. Keep legacy compatibility path for one release window.

## 6. No-Training Adoption Path

You can adopt immediately with the built-in keyword quantizer.
Train or finetune models later only if fallback rate/accuracy is not acceptable for your workload.

## Related Files

- `examples/langgraph_slipstream.py`
- `examples/langgraph_platform_skeleton/`
- `src/slipcore/langgraph.py`
- `docs/start-here.md`
- `spec/spec-00-invariants.md`
