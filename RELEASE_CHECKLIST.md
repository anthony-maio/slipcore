# Slipstream Release Checklist

## Pre-Release

### 1. Code Quality
- [ ] Run linter: `ruff check src/`
- [ ] Run type checker: `mypy src/slipcore/`
- [ ] Run full test suite: `pytest tests/ -v`
- [ ] Verify version is updated in `pyproject.toml` and `src/slipcore/__init__.py`
- [ ] Update CHANGELOG.md

### 2. Smoke Test
```bash
python -c "
from slipcore import format_slip, parse_slip, render_human, __version__
wire = format_slip('a', 'b', 'Request', 'Review')
msg = parse_slip(wire)
assert msg.force == 'Request' and msg.obj == 'Review'
print(f'slipcore {__version__} OK: {wire}')
"
```

### 3. Build
```bash
python -m build
twine check dist/*
```

---

## GitHub Release

### 4. Tag and Release
```bash
# Tag the release (use current version)
git tag -a v3.1.0 -m "Slipstream v3.1.0"
git push origin v3.1.0

# Create release on GitHub
gh release create v3.1.0 --title "Slipstream v3.1.0" --notes "See CHANGELOG.md"
```

### 5. CI/CD Auto-Publish
The GitHub Action (`.github/workflows/publish.yml`) auto-publishes to PyPI on release.

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
# Template-based (free, fast)
python -m slipcore.finetune -n 1000 -f sharegpt_thought -o train.jsonl

# LLM-enhanced (higher quality, requires API key)
python -m slipcore.finetune_llm -n 1500 --provider anthropic -o train_llm.jsonl
```

### 7. Finetune Model
See `.claude/skills/slipstream-finetune.md` for full instructions.

### 8. Release Dataset to HuggingFace
```python
from datasets import Dataset
import json

with open("train.jsonl") as f:
    data = [json.loads(line) for line in f]

dataset = Dataset.from_list(data)
dataset.push_to_hub("anthony-maio/slipstream-training-data")
```

---

## Post-Release

### 9. Verify
- [ ] Package installable: `pip install slipcore`
- [ ] Imports work: `python -c "from slipcore import format_slip; print('OK')"`
- [ ] PyPI page looks correct
- [ ] GitHub release page has correct notes

### 10. Announce
- [ ] Update README badges if needed
- [ ] Post to relevant communities

---

## Links

- PyPI: https://pypi.org/project/slipcore/
- GitHub: https://github.com/anthony-maio/slipcore
- HuggingFace Dataset: (TBD)
