# LangGraph Platform Skeleton

This folder is a drop-in starting point for integrating Slipstream transport into a LangGraph-based agent platform.

It includes:

- `state.py`: shared graph state schema.
- `routing.py`: explicit `Force:Object` route map.
- `fallback_backend.py`: pluggable fallback backend adapter.
- `graph.py`: graph builder using Slipstream encode/decode nodes.

## Why this pattern

Use Slipstream at the orchestrator layer first:

1. agent produces natural-language `thought`
2. Slipstream encode node emits `SLIP v3 ...` wire
3. transport carries compact wire
4. Slipstream decode node restores structured intent
5. conditional edges route by `Force:Object`

This works without model retraining.

## Run

```bash
pip install langgraph slipcore
python examples/langgraph_platform_skeleton/graph.py
```

## Production notes

- Replace `InMemoryFallbackBackend` with Redis/Postgres/KV backend.
- Keep fallback ref storage shared across workers.
- Start in shadow mode (JSON + SLIP) before full cutover.
