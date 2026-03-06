---
license: apache-2.0
base_model: zai-org/GLM-Z1-9B-0414
tags:
- slipstream
- multi-agent
- semantic-quantization
- force-object
datasets:
- anthonym21/slipstream-tqt
language:
- en
pipeline_tag: text-generation
library_name: peft
---

# Slipstream GLM-Z1-9B

Finetuned model for Slipstream v3 Force+Object wire generation.

## Protocol Target

Wire format:

```text
SLIP v3 <src> <dst> <Force> <Object> [payload...]
```

Example:

```text
SLIP v3 alice bob Request Review auth
```

## Training Summary

- Base model: `zai-org/GLM-Z1-9B-0414`
- Method: LoRA
- Dataset: `anthonym21/slipstream-tqt`
- Objective: Think -> Quantize -> Transmit in v3 Force+Object format

## Supported Artifacts

- LoRA adapter: [anthonym21/slipstream-glm-z1-9b](https://huggingface.co/anthonym21/slipstream-glm-z1-9b)
- Merged model: [anthonym21/slipstream-glm-z1-9b-merged](https://huggingface.co/anthonym21/slipstream-glm-z1-9b-merged)
- GGUF: [anthonym21/slipstream-glm-z1-9b-gguf](https://huggingface.co/anthonym21/slipstream-glm-z1-9b-gguf)

## Usage (Transformers + PEFT)

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base_model = AutoModelForCausalLM.from_pretrained("zai-org/GLM-Z1-9B-0414")
model = PeftModel.from_pretrained(base_model, "anthonym21/slipstream-glm-z1-9b")
tokenizer = AutoTokenizer.from_pretrained("anthonym21/slipstream-glm-z1-9b")
```

## Related Resources

- Protocol repo: [github.com/anthony-maio/slipcore](https://github.com/anthony-maio/slipcore)
- Dataset: [anthonym21/slipstream-tqt](https://huggingface.co/datasets/anthonym21/slipstream-tqt)
- Paper: [doi.org/10.5281/zenodo.18063451](https://doi.org/10.5281/zenodo.18063451)

## License

Apache 2.0.
