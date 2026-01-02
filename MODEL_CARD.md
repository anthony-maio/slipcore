---
license: apache-2.0
base_model: zai-org/GLM-Z1-9B-0414
tags:
- slipstream
- multi-agent
- semantic-quantization
- agent-communication
- think-quantize-transmit
- lora
- unsloth
datasets:
- anthonym21/slipstream-tqt
language:
- en
pipeline_tag: text-generation
library_name: peft
---

# Slipstream GLM-Z1-9B

A finetuned version of [GLM-Z1-9B-0414](https://huggingface.co/zai-org/GLM-Z1-9B-0414) trained on the **Slipstream protocol** - a semantic quantization system that achieves **82% token reduction** in multi-agent AI communication.

## Model Description

This model has learned the **Think-Quantize-Transmit (TQT)** cognitive pattern:

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

## Training Details

| Parameter | Value |
|-----------|-------|
| Base Model | zai-org/GLM-Z1-9B-0414 |
| Method | LoRA (rank=16, alpha=16) |
| Epochs | 2 |
| Learning Rate | 2e-4 |
| Batch Size | 16 (4 × 4 grad accum) |
| Sequence Length | 2048 |
| Training Examples | 2,283 |
| Hardware | Google Colab (A100/V100) |
| Framework | Unsloth + TRL |

### LoRA Target Modules
- Attention: `q_proj`, `k_proj`, `v_proj`, `o_proj`
- MLP: `gate_proj`, `up_proj`, `down_proj`

## Available Formats

| Format | Repository | Use Case |
|--------|------------|----------|
| LoRA Adapter | [slipstream-glm-z1-9b](https://huggingface.co/anthonym21/slipstream-glm-z1-9b) | Merge with base model |
| Merged 16-bit | [slipstream-glm-z1-9b-merged](https://huggingface.co/anthonym21/slipstream-glm-z1-9b-merged) | Direct loading |
| GGUF Q4_K_M | [slipstream-glm-z1-9b-gguf](https://huggingface.co/anthonym21/slipstream-glm-z1-9b-gguf) | Ollama / llama.cpp |
| GGUF Q8_0 | [slipstream-glm-z1-9b-gguf](https://huggingface.co/anthonym21/slipstream-glm-z1-9b-gguf) | Higher quality local |

## Usage

### With Transformers + PEFT

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base_model = AutoModelForCausalLM.from_pretrained("zai-org/GLM-Z1-9B-0414")
model = PeftModel.from_pretrained(base_model, "anthonym21/slipstream-glm-z1-9b")
tokenizer = AutoTokenizer.from_pretrained("anthonym21/slipstream-glm-z1-9b")
```

### With Ollama

```bash
# Download GGUF
wget https://huggingface.co/anthonym21/slipstream-glm-z1-9b-gguf/resolve/main/slipstream-q4_k_m.gguf

# Create Modelfile
cat > Modelfile <<EOF
FROM ./slipstream-q4_k_m.gguf
SYSTEM "You are an AI agent using the Slipstream protocol for efficient multi-agent communication."
EOF

# Run
ollama create slipstream -f Modelfile
ollama run slipstream "Tell bob to review my code"
```

### With Unsloth (for inference)

```python
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    "anthonym21/slipstream-glm-z1-9b",
    max_seq_length=2048,
    load_in_4bit=True,
)
FastLanguageModel.for_inference(model)
```

## UCR Anchors

The model understands 21 core anchors:

| Category | Anchors |
|----------|---------|
| Requests | `RequestTask`, `RequestReview`, `RequestHelp`, `RequestPlan` |
| Inform | `InformComplete`, `InformProgress`, `InformBlocked`, `InformStatus` |
| Propose | `ProposePlan`, `ProposeChange`, `ProposeAlternative` |
| Evaluate | `EvalApprove`, `EvalReject`, `EvalNeedsWork` |
| Meta | `Accept`, `Reject`, `MetaAck`, `MetaHandoff`, `Fallback` |

## Wire Format

```
SLIP v1 <src> <dst> <anchor> [payload...]
```

Example: `SLIP v1 alice bob RequestReview auth_module`

## Related Resources

- **Protocol Spec**: [github.com/anthony-maio/slipcore](https://github.com/anthony-maio/slipcore)
- **Training Dataset**: [hf.co/anthonym21/slipstream-tqt](https://huggingface.co/datasets/anthonym21/slipstream-tqt)
- **Paper**: [Slipstream: Semantic Quantization for Efficient Multi-Agent Coordination](https://doi.org/10.5281/zenodo.18063451)
- **PyPI**: `pip install slipcore`

## Citation

```bibtex
@misc{maio2025slipstream,
  title={Slipstream: Semantic Quantization for Efficient Multi-Agent Coordination},
  author={Maio, Anthony},
  year={2025},
  publisher={Hugging Face},
  url={https://huggingface.co/anthonym21/slipstream-glm-z1-9b}
}
```

## License

Apache 2.0
