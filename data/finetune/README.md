# Finetune data for SLIP-aware agents

The idea: train or PEFT-finetune LLM agents (planner, executor, critic) to read/write nSLIP messages reliably.

## Format

We use JSONL with fields:

- `role`: which agent type this example is for (`planner`, `executor`, `critic`, `coordinator`).
- `input`: the full prompt the model sees (including instructions + context).
- `target`: the nSLIP string the model should output.

This matches common chat/seq2seq finetune setups (LoRA, QLoRA, etc.).

Example:

```json
{
  "role": "planner",
  "input": "You are a planner agent. You receive nSLIP messages and a goal registry.\nCurrent message: @a3|f0|c3|s0|d1|t1|g1|k1|q1|t\"demo_task\"#\nGoal[1]: Refactor the auth module to remove legacy token path.\nRespond with a PROPOSE/PLAN nSLIP message.",
  "target": "@a4|f1|c3|s1|d0|t2|g1|k1|q1|t\"plan_v1\"#"
}
```

For executor:

```json
{
  "role": "executor",
  "input": "You are an executor agent. You receive PLAN messages and produce INFORM/EVAL messages.\nPlan message: @a4|f1|c3|s1|d2|t3|g1|k1|t\"plan_v1\"#\nGoal[1]: Refactor auth module.",
  "target": "@a1|f2|c3|s2|d0|t4|g1|k1|r2|u\"in_progress\"#"
}
```

You can generate thousands of synthetic examples by:

1. Sampling random goals (small natural language descriptions).
2. Generating canonical nSLIP REQUEST/TASK messages.
3. Using template-based or LLM-based generators to create PROPOSE, INFORM, EVAL responses.
4. Writing `input`/`target` pairs into JSONL.

## Generate Dataset

```bash
python -m slipcore.generate_dataset --num-conversations 1000 --output data/finetune/nslip_pairs.generated.jsonl
```

## Usage

* For LoRA/QLoRA: treat `input` as source text, `target` as label.
* For multi-role models: either
  * train one model per `role`, or
  * include `role` at the top of `input` and train a single multi-role model.
