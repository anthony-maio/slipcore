---
name: slipstream-finetune
description: Finetune LLMs to speak Slipstream v3 natively - complete guide
---

# Slipstream v3 Finetuning Guide

Train LLMs to communicate using the Slipstream v3 protocol natively with factorized Force-Object intents.

## Wire Format (v3)

```
SLIP v3 <src> <dst> <Force> <Object> [payload...]
```

Forces: Observe, Inform, Ask, Request, Propose, Commit, Eval, Meta, Accept, Reject, Error, Fallback

## Quick Start

### 1. Generate High-Quality Dataset

**Option A: Template-based (fast, free)**
```bash
python -m slipcore.finetune -n 1000 -f sharegpt_thought -o slipstream_train.jsonl
```

**Option B: LLM-enhanced (higher quality, requires API)**
```bash
# Using Claude API (recommended for quality)
export ANTHROPIC_API_KEY="your-key"
python -m slipcore.finetune_llm -n 1000 --provider anthropic -o slipstream_train.jsonl

# Using Gemini (good quality, cheap)
export GEMINI_API_KEY="your-key"
python -m slipcore.finetune_llm -n 1000 --provider gemini -o slipstream_train.jsonl
```

**Option C: Migrate existing v2 data to v3**
```bash
python scripts/migrate_v2_data.py data/slipstream-tqt.jsonl data/slipstream-tqt-v3.jsonl
```

### 2. Finetune with Unsloth

```python
from unsloth import FastLanguageModel
import torch

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="THUDM/GLM-4-9B-0414",
    max_seq_length=2048,
    dtype=None,
    load_in_4bit=True,
)

model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ],
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=42,
)

from datasets import load_dataset
dataset = load_dataset("json", data_files="slipstream_train.jsonl", split="train")

def format_glm4(example):
    convs = example["conversations"]
    text = ""
    for conv in convs:
        if conv["from"] == "system":
            text += f"[gMASK]<sop><|system|>\n{conv['value']}"
        elif conv["from"] == "human":
            text += f"<|user|>\n{conv['value']}"
        elif conv["from"] == "gpt":
            text += f"<|assistant|>\n{conv['value']}"
    return {"text": text}

dataset = dataset.map(format_glm4)

from trl import SFTTrainer
from transformers import TrainingArguments

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=2048,
    args=TrainingArguments(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=10,
        max_steps=200,
        learning_rate=2e-4,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=10,
        output_dir="slipstream_glm4",
        optim="adamw_8bit",
        seed=42,
    ),
)

trainer.train()
model.save_pretrained("slipstream_glm4_lora")
tokenizer.save_pretrained("slipstream_glm4_lora")
```

### 3. Test the Finetuned Model

```python
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="slipstream_glm4_lora",
    max_seq_length=2048,
    dtype=None,
    load_in_4bit=True,
)
FastLanguageModel.for_inference(model)

prompt = """[gMASK]<sop><|system|>
You communicate using the Slipstream v3 protocol.<|user|>
Tell the backend team to review the authentication code<|assistant|>
"""

inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, max_new_tokens=64)
print(tokenizer.decode(outputs[0]))
# Expected: SLIP v3 agent backend Request Review auth
```

### 4. Export and Release

```python
# GGUF for Ollama/llama.cpp
model.save_pretrained_gguf(
    "slipstream_glm4_gguf",
    tokenizer,
    quantization_method="q4_k_m",
)
```

## Dataset Formats

| Format | Description | Use Case |
|--------|-------------|----------|
| `sharegpt_thought` | THOUGHT + SLIP wire | Recommended for TQT |
| `sharegpt_semantics` | THOUGHT + QUANTIZE + SLIP | Maximum supervision |
| `sharegpt` | Direct SLIP wire | Simple classification |
| `chat` | OpenAI chat format | API-compatible |
| `alpaca` | Instruction/output | Alpaca-style training |

## Training Tips

1. **Dataset size**: 500-2000 examples is usually sufficient
2. **Quality > Quantity**: LLM-generated data beats templates
3. **Epochs**: 1-2 epochs, watch for overfitting
4. **Combine sources**: Template + LLM + migrated v2 data for diversity
5. **Validate**: Check output contains `SLIP v3` with valid Force-Object pairs

## Using with Ollama

```bash
cat > Modelfile << 'EOF'
FROM ./slipstream_glm4_gguf/slipstream-glm4-9b-Q4_K_M.gguf

SYSTEM "You communicate using the Slipstream v3 protocol. Wire format: SLIP v3 <src> <dst> <Force> <Object> [payload...]. Forces: Observe, Inform, Ask, Request, Propose, Commit, Eval, Meta, Accept, Reject, Error, Fallback."

TEMPLATE """[gMASK]<sop><|system|>
{{ .System }}<|user|>
{{ .Prompt }}<|assistant|>
{{ .Response }}"""
EOF

ollama create slipstream -f Modelfile
ollama run slipstream "Tell alice to review the API code"
# -> SLIP v3 agent alice Request Review api
```
