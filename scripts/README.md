# Slipstream Training Scripts

Complete pipeline for finetuning and deploying Slipstream models.

## Quick Start

```bash
# 1. Install dependencies
pip install -r scripts/requirements-training.txt

# 2. Download base model
python scripts/download_model.py --download
# OR: huggingface-cli download zai-org/GLM-Z1-9B-0414 --local-dir ./models/GLM-Z1-9B-0414

# 3. Train
python scripts/train_slipstream.py

# 4. Test
python scripts/test_slipstream.py

# 5. Deploy with Ollama
cp output/Modelfile.template output/Modelfile
ollama create slipstream -f output/Modelfile
ollama run slipstream "Tell bob to review my code"
```

## Scripts

| Script | Purpose |
|--------|---------|
| `download_model.py` | Download GLM-Z1-9B-0414 from HuggingFace |
| `train_slipstream.py` | Finetune with Unsloth + LoRA |
| `test_slipstream.py` | Test the finetuned model |
| `requirements-training.txt` | Python dependencies |

## Training Options

```bash
# Basic training (2 epochs, default settings)
python scripts/train_slipstream.py

# Custom settings
python scripts/train_slipstream.py \
    --epochs 3 \
    --batch-size 4 \
    --lr 1e-4 \
    --lora-r 32

# Resume from checkpoint
python scripts/train_slipstream.py --resume ./output/checkpoints/checkpoint-500

# Push to HuggingFace Hub
python scripts/train_slipstream.py --push-to-hub YOUR_USERNAME/slipstream-glm
```

## Output Files

After training, you'll have:

```
output/
├── checkpoints/           # Training checkpoints
├── slipstream-lora/       # LoRA adapter only (~50MB)
├── slipstream-merged/     # Full merged model (~18GB)
├── slipstream-q4_k_m.gguf # Quantized for Ollama (~5GB)
├── slipstream-q8_0.gguf   # Higher quality quantized (~9GB)
└── Modelfile              # For Ollama deployment
```

## Hardware Requirements

| VRAM | Configuration |
|------|---------------|
| 8GB | 4-bit, batch_size=1, grad_accum=8 |
| 16GB | 4-bit, batch_size=2, grad_accum=4 |
| 24GB | 4-bit, batch_size=4, grad_accum=2 |
| 48GB+ | 16-bit, batch_size=8 |

## Troubleshooting

### CUDA Out of Memory
- Reduce `--batch-size` to 1
- Increase `--grad-accum` to compensate
- Ensure `--no-4bit` is NOT set

### Model Loading Errors
- Make sure `trust_remote_code=True` is set
- Check that the model path is correct
- Try using the HuggingFace repo ID directly: `zai-org/GLM-Z1-9B-0414`

### GGUF Export Fails
- Ensure llama.cpp dependencies are installed
- Try exporting merged model only: `--no-gguf`
- Export GGUF separately with llama.cpp tools
