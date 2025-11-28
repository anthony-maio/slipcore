# Finetuning Guide for SLIPCore

This guide shows how to finetune a model to reliably speak the nSLIP protocol using **Unsloth** (fast LoRA/QLoRA).

## Prerequisites

```bash
# Install Unsloth (requires CUDA GPU)
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"

# Or for specific CUDA version
pip install "unsloth[cu121-ampere-torch240]"  # For RTX 30xx/40xx
```

## Quick Start

### 1. Load Model with Unsloth

```python
from unsloth import FastLanguageModel
import torch

# Load base model with 4-bit quantization
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Qwen2.5-7B-bnb-4bit",  # or "unsloth/Llama-3.2-3B-bnb-4bit"
    max_seq_length=2048,
    dtype=None,  # Auto-detect
    load_in_4bit=True,
)

# Add LoRA adapters
model = FastLanguageModel.get_peft_model(
    model,
    r=16,  # LoRA rank
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=42,
)
```

### 2. Prepare Dataset

```python
from datasets import load_dataset

# Load SLIPCore training data
dataset = load_dataset("json", data_files="data/finetune/train.jsonl", split="train")

# Format for instruction tuning
def format_prompt(example):
    return {
        "text": f"""### Instruction:
{example['input']}

### Response:
{example['target']}"""
    }

dataset = dataset.map(format_prompt)
```

### 3. Train

```python
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
        warmup_steps=5,
        max_steps=100,  # Increase for full training
        learning_rate=2e-4,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=10,
        output_dir="outputs",
        optim="adamw_8bit",
    ),
)

trainer.train()
```

### 4. Save & Export

```python
# Save LoRA adapter
model.save_pretrained("slipcore-lora")
tokenizer.save_pretrained("slipcore-lora")

# Merge and export to GGUF for Ollama
model.save_pretrained_gguf("slipcore-gguf", tokenizer, quantization_method="q4_k_m")
```

### 5. Use with Ollama

```bash
# Create Modelfile
cat > Modelfile << 'EOF'
FROM ./slipcore-gguf/unsloth.Q4_K_M.gguf
TEMPLATE "### Instruction:\n{{ .Prompt }}\n\n### Response:\n"
PARAMETER stop "###"
EOF

# Create Ollama model
ollama create slipcore -f Modelfile

# Test it
ollama run slipcore "You are a planner agent. Goal[1]: Add rate limiting. Respond with nSLIP."
```

## Full Training Script

See `scripts/train_unsloth.py` for a complete training script.

## Hardware Requirements

| Model | VRAM | Training Time |
|-------|------|---------------|
| Qwen2.5-3B | ~8GB | ~30 min |
| Qwen2.5-7B | ~12GB | ~1 hour |
| Llama-3.2-3B | ~8GB | ~30 min |
| Mistral-7B | ~12GB | ~1 hour |

With QLoRA (4-bit), you can train 7B models on a single RTX 3090/4090.

## Tips

1. **Start small**: Train for 50-100 steps first to verify it's working
2. **Check outputs**: The model should output valid nSLIP like `@a4|f1|c1|S1|d0|T2|g1|k1#`
3. **Increase data**: More diverse goals = better generalization
4. **Role-specific models**: Consider training separate models for planner/executor/critic

## Expected Results

After finetuning, the model should:
- Consistently output valid nSLIP wire format
- Correctly increment turn numbers
- Use appropriate acts (REQUEST→PROPOSE→INFORM→EVAL)
- Include relevant slots (goal_id, task_id, etc.)
