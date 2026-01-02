"""
Slipstream Finetuning Script with Unsloth

Finetunes GLM-Z1-9B-0414 on the Slipstream Think-Quantize-Transmit dataset.
Optimized for consumer GPUs (8-24GB VRAM) using 4-bit quantization + LoRA.

Usage:
    # Basic training
    python scripts/train_slipstream.py

    # With custom parameters
    python scripts/train_slipstream.py --epochs 3 --batch-size 4 --lr 2e-4

    # Resume from checkpoint
    python scripts/train_slipstream.py --resume checkpoints/checkpoint-500

Requirements:
    pip install unsloth
    pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
    pip install --no-deps trl peft accelerate bitsandbytes

Note: On Windows, you may need to install torch with CUDA first:
    pip install torch --index-url https://download.pytorch.org/whl/cu121
"""

import os
import json
import argparse
from pathlib import Path
from datetime import datetime


def setup_environment():
    """Configure environment for optimal training."""
    # Reduce memory fragmentation
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    # Disable tokenizers parallelism warning
    os.environ["TOKENIZERS_PARALLELISM"] = "false"


def load_model(model_path: str, max_seq_length: int = 2048, load_in_4bit: bool = True):
    """
    Load GLM-Z1-9B-0414 with Unsloth optimizations.

    Args:
        model_path: Path to model or HuggingFace repo ID
        max_seq_length: Maximum sequence length for training
        load_in_4bit: Use 4-bit quantization (recommended for <24GB VRAM)
    """
    from unsloth import FastLanguageModel

    print(f"Loading model from: {model_path}")
    print(f"Max sequence length: {max_seq_length}")
    print(f"4-bit quantization: {load_in_4bit}")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_path,
        max_seq_length=max_seq_length,
        dtype=None,  # Auto-detect (float16 for most GPUs)
        load_in_4bit=load_in_4bit,
        trust_remote_code=True,  # Required for GLM
        # token="hf_...",  # Add your HF token if needed
    )

    return model, tokenizer


def apply_lora(model, r: int = 16, lora_alpha: int = 16, lora_dropout: float = 0):
    """
    Apply LoRA adapters to the model.

    Args:
        model: The base model
        r: LoRA rank (higher = more capacity but slower)
        lora_alpha: LoRA scaling factor
        lora_dropout: Dropout for regularization
    """
    from unsloth import FastLanguageModel

    print(f"Applying LoRA: r={r}, alpha={lora_alpha}, dropout={lora_dropout}")

    model = FastLanguageModel.get_peft_model(
        model,
        r=r,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",  # Attention
            "gate_proj", "up_proj", "down_proj",     # MLP
        ],
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        use_gradient_checkpointing="unsloth",  # Memory optimization
        random_state=42,
        use_rslora=False,  # Rank-stabilized LoRA (optional)
        loftq_config=None,
    )

    # Print trainable parameters
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

    return model


def load_dataset(data_path: str, tokenizer, max_seq_length: int = 2048):
    """
    Load and format the Slipstream dataset.

    Expects ShareGPT format with conversations:
    {
        "conversations": [
            {"from": "system", "value": "..."},
            {"from": "human", "value": "..."},
            {"from": "gpt", "value": "THOUGHT: ...\nQUANTIZE: ...\nSLIP: ..."}
        ]
    }
    """
    from datasets import load_dataset
    from unsloth.chat_templates import get_chat_template, standardize_sharegpt

    print(f"Loading dataset from: {data_path}")

    # Load JSONL file
    dataset = load_dataset("json", data_files=data_path, split="train")
    print(f"Loaded {len(dataset)} examples")

    # Standardize to ShareGPT format
    dataset = standardize_sharegpt(dataset)

    # Apply chat template
    tokenizer = get_chat_template(
        tokenizer,
        chat_template="chatml",  # Works well for most models
    )

    def formatting_func(examples):
        convos = examples["conversations"]
        texts = []
        for convo in convos:
            text = tokenizer.apply_chat_template(
                convo,
                tokenize=False,
                add_generation_prompt=False
            )
            texts.append(text)
        return {"text": texts}

    dataset = dataset.map(formatting_func, batched=True)

    # Show sample
    print("\n--- Sample formatted example ---")
    print(dataset[0]["text"][:500] + "...")

    return dataset, tokenizer


def create_trainer(
    model,
    tokenizer,
    dataset,
    output_dir: str = "./checkpoints",
    num_epochs: int = 2,
    batch_size: int = 2,
    gradient_accumulation_steps: int = 4,
    learning_rate: float = 2e-4,
    warmup_steps: int = 50,
    logging_steps: int = 10,
    save_steps: int = 100,
    max_seq_length: int = 2048,
):
    """
    Create the SFTTrainer for finetuning.
    """
    from trl import SFTTrainer
    from transformers import TrainingArguments
    from unsloth import is_bfloat16_supported

    print(f"\nTraining configuration:")
    print(f"  Output dir: {output_dir}")
    print(f"  Epochs: {num_epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Gradient accumulation: {gradient_accumulation_steps}")
    print(f"  Effective batch size: {batch_size * gradient_accumulation_steps}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Max sequence length: {max_seq_length}")

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        dataset_num_proc=2,
        packing=False,  # Can enable for efficiency, but may affect quality
        args=TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            learning_rate=learning_rate,
            lr_scheduler_type="cosine",
            warmup_steps=warmup_steps,
            logging_steps=logging_steps,
            save_steps=save_steps,
            save_total_limit=3,
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
            optim="adamw_8bit",
            weight_decay=0.01,
            seed=42,
            report_to="none",  # Set to "wandb" for tracking
        ),
    )

    return trainer


def export_model(model, tokenizer, output_dir: str, export_gguf: bool = True):
    """
    Export the finetuned model in multiple formats.
    """
    from pathlib import Path

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # 1. Save LoRA adapter only (smallest, for inference with base model)
    lora_path = output_path / "slipstream-lora"
    print(f"\n1. Saving LoRA adapter to: {lora_path}")
    model.save_pretrained(lora_path)
    tokenizer.save_pretrained(lora_path)

    # 2. Save merged model (full weights)
    merged_path = output_path / "slipstream-merged"
    print(f"2. Saving merged model to: {merged_path}")
    model.save_pretrained_merged(
        merged_path,
        tokenizer,
        save_method="merged_16bit",  # or "merged_4bit" for smaller
    )

    # 3. Export to GGUF for llama.cpp / Ollama / SGLang
    if export_gguf:
        print(f"3. Exporting GGUF quantizations...")

        # Q4_K_M - Good balance of size/quality
        gguf_path_q4 = output_path / "slipstream-q4_k_m.gguf"
        print(f"   - Q4_K_M: {gguf_path_q4}")
        model.save_pretrained_gguf(
            str(output_path / "slipstream"),
            tokenizer,
            quantization_method="q4_k_m",
        )

        # Q8_0 - Higher quality
        gguf_path_q8 = output_path / "slipstream-q8_0.gguf"
        print(f"   - Q8_0: {gguf_path_q8}")
        model.save_pretrained_gguf(
            str(output_path / "slipstream"),
            tokenizer,
            quantization_method="q8_0",
        )

    print(f"\nExport complete! Files in: {output_path}")
    return output_path


def push_to_hub(model, tokenizer, repo_id: str, private: bool = False):
    """
    Push the model to HuggingFace Hub.
    """
    print(f"\nPushing to HuggingFace Hub: {repo_id}")

    # Push LoRA adapter
    model.push_to_hub(
        repo_id,
        tokenizer=tokenizer,
        private=private,
        token=os.environ.get("HF_TOKEN"),
    )

    # Also push merged version
    model.push_to_hub_merged(
        f"{repo_id}-merged",
        tokenizer,
        save_method="merged_16bit",
        private=private,
        token=os.environ.get("HF_TOKEN"),
    )

    print(f"Pushed to: https://huggingface.co/{repo_id}")


def main():
    parser = argparse.ArgumentParser(description="Finetune GLM-Z1-9B on Slipstream")

    # Model args
    parser.add_argument("--model", type=str, default="./models/GLM-Z1-9B-0414",
                        help="Path to model or HuggingFace repo ID")
    parser.add_argument("--max-seq-length", type=int, default=2048,
                        help="Maximum sequence length")
    parser.add_argument("--no-4bit", action="store_true",
                        help="Disable 4-bit quantization (needs more VRAM)")

    # LoRA args
    parser.add_argument("--lora-r", type=int, default=16,
                        help="LoRA rank")
    parser.add_argument("--lora-alpha", type=int, default=16,
                        help="LoRA alpha")

    # Training args
    parser.add_argument("--data", type=str, default="./data/slipstream_train_combined.jsonl",
                        help="Path to training data")
    parser.add_argument("--output", type=str, default="./output",
                        help="Output directory")
    parser.add_argument("--epochs", type=int, default=2,
                        help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=2,
                        help="Per-device batch size")
    parser.add_argument("--grad-accum", type=int, default=4,
                        help="Gradient accumulation steps")
    parser.add_argument("--lr", type=float, default=2e-4,
                        help="Learning rate")
    parser.add_argument("--resume", type=str, default=None,
                        help="Resume from checkpoint")

    # Export args
    parser.add_argument("--no-gguf", action="store_true",
                        help="Skip GGUF export")
    parser.add_argument("--push-to-hub", type=str, default=None,
                        help="Push to HuggingFace Hub (repo ID)")

    args = parser.parse_args()

    # Setup
    setup_environment()

    print("=" * 60)
    print("SLIPSTREAM FINETUNING")
    print("=" * 60)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Load model
    model, tokenizer = load_model(
        args.model,
        max_seq_length=args.max_seq_length,
        load_in_4bit=not args.no_4bit,
    )

    # Apply LoRA
    model = apply_lora(model, r=args.lora_r, lora_alpha=args.lora_alpha)

    # Load dataset
    dataset, tokenizer = load_dataset(
        args.data,
        tokenizer,
        max_seq_length=args.max_seq_length,
    )

    # Create trainer
    trainer = create_trainer(
        model=model,
        tokenizer=tokenizer,
        dataset=dataset,
        output_dir=f"{args.output}/checkpoints",
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        max_seq_length=args.max_seq_length,
    )

    # Train
    print("\n" + "=" * 60)
    print("TRAINING")
    print("=" * 60)

    if args.resume:
        print(f"Resuming from: {args.resume}")
        trainer.train(resume_from_checkpoint=args.resume)
    else:
        trainer.train()

    # Export
    print("\n" + "=" * 60)
    print("EXPORTING")
    print("=" * 60)

    export_model(
        model,
        tokenizer,
        output_dir=args.output,
        export_gguf=not args.no_gguf,
    )

    # Push to Hub
    if args.push_to_hub:
        push_to_hub(model, tokenizer, args.push_to_hub)

    print("\n" + "=" * 60)
    print("COMPLETE!")
    print("=" * 60)
    print(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\nNext steps:")
    print(f"  1. Test: python scripts/test_slipstream.py")
    print(f"  2. Run with Ollama: ollama create slipstream -f {args.output}/Modelfile")
    print(f"  3. Push to HF: python scripts/train_slipstream.py --push-to-hub YOUR_USERNAME/slipstream-glm")


if __name__ == "__main__":
    main()
