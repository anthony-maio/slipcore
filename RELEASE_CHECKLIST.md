# Slipstream v2.0.0 Release Checklist

## Pre-Release

### 1. Academic Paper
- [ ] Locate existing draft in project
- [ ] Have Gemini/Claude review and polish
- [ ] Add proper citations and references
- [ ] Target: arXiv or workshop submission

### 2. Content & Marketing
- [ ] **LinkedIn Post** - Short announcement
  - Key value prop: 80% token reduction
  - Tag relevant people/orgs (AAIF, LangChain, etc.)
  - Include demo GIF or screenshot

- [ ] **Medium Post** - Deep dive
  - Problem: Agent coordination overhead
  - Solution: Semantic quantization
  - Visualizations: UCR manifold, token comparison
  - Code examples
  - Call to action: GitHub stars, community Discord

### 3. Code Quality
- [ ] Run linter: `ruff check src/`
- [ ] Verify all tests pass
- [ ] Update version in `pyproject.toml` (currently 2.0.0)
- [ ] Update CHANGELOG.md (create if needed)

---

## GitHub Release

### 4. Create Release
```bash
# Tag the release
git tag -a v2.0.0 -m "Slipstream v2.0.0 - Semantic Quantization"
git push origin v2.0.0

# Create release on GitHub UI or:
gh release create v2.0.0 --title "Slipstream v2.0.0" --notes "See CHANGELOG.md"
```

### 5. CI/CD Auto-Publish
The GitHub Action (`.github/workflows/publish.yml`) will auto-publish to PyPI.

**First-time setup required:**
1. Go to https://pypi.org/manage/project/slipcore/settings/publishing/
2. Add GitHub as trusted publisher:
   - Owner: `anthony-maio`
   - Repository: `slipcore`
   - Workflow: `publish.yml`

---

## Model & Dataset Release

### 6. Generate Dataset
```bash
# High-quality with Claude API (~$0.75 for 1500)
export ANTHROPIC_API_KEY="sk-..."
python -m slipcore.finetune_llm -n 1500 --provider anthropic -o slipstream_train.jsonl

# Or cheaper with DeepSeek (~$0.03 for 1500)
export DEEPSEEK_API_KEY="sk-..."
python -m slipcore.finetune_llm -n 1500 --provider deepseek -o slipstream_train.jsonl
```

### 7. Finetune Model
```bash
# On machine with GPU (8GB+ VRAM)
# See .claude/skills/slipstream-finetune.md for full code

# Quick version:
pip install unsloth transformers datasets trl
python finetune_glm4.py  # Create this from the skill
```

### 8. Export Models

**LoRA Adapter (~200MB):**
```python
model.save_pretrained("slipstream_glm4_lora")
model.push_to_hub("anthony-maio/slipstream-glm4-9b-lora")
```

**Merged Model (~18GB):**
```python
merged = model.merge_and_unload()
merged.push_to_hub("anthony-maio/slipstream-glm4-9b")
```

**GGUF for Ollama (~5GB):**
```python
model.save_pretrained_gguf(
    "slipstream_gguf",
    tokenizer,
    quantization_method="q4_k_m"
)
model.push_to_hub_gguf(
    "anthony-maio/slipstream-glm4-9b-gguf",
    tokenizer,
    quantization_method=["q4_k_m", "q8_0"]
)
```

### 9. Release Dataset

**HuggingFace:**
```python
from datasets import Dataset
import json

with open("slipstream_train.jsonl") as f:
    data = [json.loads(line) for line in f]

dataset = Dataset.from_list(data)
dataset.push_to_hub("anthony-maio/slipstream-training-data")
```

**Kaggle:**
```bash
# Create dataset-metadata.json first
kaggle datasets create -p ./data -u
```

### 10. Ollama Registry
```bash
# Create Modelfile
cat > Modelfile << 'EOF'
FROM ./slipstream-glm4-9b-Q4_K_M.gguf
SYSTEM "You communicate using the Slipstream protocol."
EOF

# Test locally
ollama create slipstream -f Modelfile
ollama run slipstream "Tell bob to review the code"

# Push to Ollama registry (requires account)
ollama push anthony-maio/slipstream
```

---

## Post-Release

### 11. Update README
- [ ] Add HuggingFace model links
- [ ] Add Ollama install command
- [ ] Add dataset links
- [ ] Update badges (PyPI version, downloads, etc.)

### 12. AAIF Submission
- [ ] Check AAIF project requirements: https://lfaidata.foundation/
- [ ] Prepare technical overview document
- [ ] Demonstrate community traction (GitHub stars, downloads)
- [ ] Submit proposal

### 13. Community Building
- [ ] Create Discord server or GitHub Discussions
- [ ] Respond to issues/PRs
- [ ] Share in relevant communities:
  - r/LocalLLaMA
  - r/MachineLearning
  - LangChain Discord
  - AI Twitter/X

---

## Quick Commands Reference

```bash
# Build package
python -m build

# Test locally
pip install -e .
python -c "from slipcore import slip; print(slip('a','b','RequestReview'))"

# Publish to PyPI (manual, if CI fails)
twine upload dist/*

# Generate dataset
python -m slipcore.finetune_llm -n 1500 --provider anthropic -o train.jsonl

# Run local tests
python -c "from slipcore import slip, decode; assert decode(slip('a','b','RequestReview')).anchor.mnemonic == 'RequestReview'"
```

---

## Links (Update After Release)

- PyPI: https://pypi.org/project/slipcore/
- GitHub: https://github.com/anthony-maio/slipcore
- HuggingFace Model: (TBD)
- HuggingFace Dataset: (TBD)
- Ollama: (TBD)
- Paper: (TBD)
