# Datasheet: Slipstream Think-Quantize-Transmit Dataset

## Motivation

The dataset trains models to convert natural-language coordination intent into Slipstream v3 wire messages:

```text
SLIP v3 <src> <dst> <Force> <Object> [payload...]
```

## Composition

- Primary file: `slipstream-tqt-v3.jsonl`
- Legacy compatibility file: `slipstream-tqt.jsonl` (v1 style)
- Records are ShareGPT-style conversations with system/human/gpt turns.
- `gpt` responses include `THOUGHT`, `QUANTIZE`, and `SLIP` lines.

## Collection Process

Data was generated with template-based and LLM-assisted generation scripts in `src/slipcore/` and reviewed for schema correctness.

## Intended Use

- Finetuning models for Slipstream v3 communication.
- Studying token-efficient multi-agent coordination behavior.

Not intended for general QA/chat tasks or safety-critical decision systems.

## Distribution

- Hugging Face: `anthonym21/slipstream-tqt`
- Repository: `github.com/anthony-maio/slipcore`
- License: Apache-2.0

## Maintenance

Maintainer: Anthony Maio (`anthony@making-minds.ai`)

Protocol or schema changes must keep paper/spec/code/tests synchronized in the same release cycle.
