---
language:
- en
license: apache-2.0
task_categories:
- text-generation
- text2text-generation
tags:
- multi-agent
- agent-communication
- semantic-quantization
- slipstream
- protocol
- llm-finetuning
- think-quantize-transmit
pretty_name: Slipstream Think-Quantize-Transmit Dataset
size_categories:
- 1K<n<10K
dataset_info:
  features:
  - name: conversations
    sequence:
    - name: from
      dtype: string
    - name: value
      dtype: string
  splits:
  - name: train
    num_examples: 2283
  download_size: 4200000
  dataset_size: 4200000
configs:
- config_name: default
  data_files:
  - split: train
    path: slipstream_train_combined.jsonl
---

# Slipstream Think-Quantize-Transmit Dataset

Training data for teaching LLMs the **Slipstream protocol** - a semantic quantization system that achieves **82% token reduction** in multi-agent AI communication.

## Dataset Description

This dataset trains models to perform **Think-Quantize-Transmit (TQT)** - a cognitive pattern where:

1. **THINK**: Reason about the communication intent
2. **QUANTIZE**: Map intent to a semantic anchor in the UCR manifold
3. **TRANSMIT**: Output a compact SLIP wire format message

### Example

**Input:**
```
Tell bob to review my authentication code
```

**Output:**
```
THOUGHT: I need bob to do a code review on the auth module
QUANTIZE: [ACTION=request | DOMAIN=task | URGENCY=normal | POLARITY=neutral] -> RequestReview
SLIP: SLIP v1 alice bob RequestReview auth_module
```

## Dataset Statistics

| Metric | Value |
|--------|-------|
| Total examples | 2,283 |
| With THOUGHT | 100% |
| With QUANTIZE | 78.1% |
| Fallback examples | 8% (182) |
| Unique anchors | 21 |
| Avg sequence length | ~150 tokens |

### Anchor Distribution

| Anchor | Count | % |
|--------|-------|---|
| RequestTask | 237 | 10.4% |
| Fallback | 182 | 8.0% |
| InformComplete | 173 | 7.6% |
| RequestHelp | 165 | 7.2% |
| RequestReview | 158 | 6.9% |
| InformBlocked | 147 | 6.4% |
| InformProgress | 127 | 5.6% |
| ... | ... | ... |

## Format

ShareGPT format (compatible with Unsloth, Axolotl, LLaMA-Factory):

```json
{
  "conversations": [
    {"from": "system", "value": "You are an AI agent using Slipstream..."},
    {"from": "human", "value": "Tell bob to review my code"},
    {"from": "gpt", "value": "THOUGHT: ...\nQUANTIZE: ...\nSLIP: ..."}
  ]
}
```

## Usage

### With Hugging Face Datasets
```python
from datasets import load_dataset

dataset = load_dataset("anthony-maio/slipstream-tqt")
```

### With Unsloth
```python
from unsloth import FastLanguageModel
from datasets import load_dataset

dataset = load_dataset("anthony-maio/slipstream-tqt", split="train")
# ... finetune with SFTTrainer
```

## UCR Anchors

The Universal Concept Reference (UCR) defines 21 core anchors:

**Requests:** `RequestTask`, `RequestReview`, `RequestHelp`, `RequestPlan`

**Inform:** `InformComplete`, `InformProgress`, `InformBlocked`, `InformStatus`

**Propose:** `ProposePlan`, `ProposeChange`, `ProposeAlternative`

**Evaluate:** `EvalApprove`, `EvalReject`, `EvalNeedsWork`

**Meta:** `Accept`, `Reject`, `MetaAck`, `MetaHandoff`, `Fallback`

Each anchor occupies a position in a 4D semantic manifold:
- **ACTION**: observe, inform, ask, request, propose, commit, evaluate, meta
- **POLARITY**: negative → neutral → positive
- **DOMAIN**: task, plan, observation, evaluation, control, resource, error, general
- **URGENCY**: background → normal → critical

## Citation

```bibtex
@dataset{maio2025slipstream,
  title={Slipstream Think-Quantize-Transmit Dataset},
  author={Maio, Anthony},
  year={2025},
  publisher={Hugging Face},
  url={https://huggingface.co/datasets/anthony-maio/slipstream-tqt}
}
```

## Related Resources

- **Paper:** [Slipstream: Semantic Quantization for Efficient Multi-Agent Coordination](https://doi.org/10.5281/zenodo.18063451)
- **Code:** [github.com/anthony-maio/slipcore](https://github.com/anthony-maio/slipcore)
- **Model:** [huggingface.co/anthonym21/slipstream-glm-z1-9b](https://huggingface.co/anthonym21/slipstream-glm-z1-9b)

## License

Apache 2.0
