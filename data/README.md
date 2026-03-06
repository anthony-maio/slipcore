---
language:
- en
license: apache-2.0
task_categories:
- text-generation
- text2text-generation
tags:
- multi-agent
- semantic-quantization
- slipstream
- force-object
pretty_name: Slipstream Think-Quantize-Transmit Dataset
configs:
- config_name: default
  data_files:
  - split: train
    path: slipstream-tqt-v3.jsonl
---

# Slipstream Think-Quantize-Transmit Dataset

Training data for Slipstream v3 Force+Object protocol behavior.

## Files in this directory

- `slipstream-tqt-v3.jsonl`: current dataset (v3 format)
- `slipstream-tqt.jsonl`: legacy v1-format dataset retained for migration
- `DATASHEET.md`: dataset documentation

## Example Output Format

```text
THOUGHT: Need reviewer to check auth code.
QUANTIZE: Force=Request Object=Review
SLIP: SLIP v3 dev reviewer Request Review auth
```

## Usage

```python
from datasets import load_dataset

dataset = load_dataset("anthonym21/slipstream-tqt", split="train")
```

## Related

- Protocol: https://github.com/anthony-maio/slipcore
- Model: https://huggingface.co/anthonym21/slipstream-glm-z1-9b
- Paper: https://doi.org/10.5281/zenodo.18063451
