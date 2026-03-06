# Slipstream Release Checklist

## 1. Pre-Release Quality Gate

- [ ] `PYTHONPATH=src ruff check src/`
- [ ] `PYTHONPATH=src mypy src/slipcore/`
- [ ] `PYTHONPATH=src pytest tests -v --tb=short`
- [ ] Version updated in `pyproject.toml` and `src/slipcore/__init__.py`
- [ ] `CHANGELOG.md` updated

## 2. Conformance / Migration Gate

- [ ] Conformance vectors updated for normative changes (`spec/conformance/*.jsonl`)
- [ ] Fallback strict behavior verified (`Fallback` requires 1-16 char ref)
- [ ] Legacy migration path validated (`parse_slip_legacy`)

## 3. Paper / Spec / Code Sync Gate

- [x] Spec updated (`spec/spec-00-invariants.md` and related docs)
- [x] Paper updated (`private/zenodo/slipstream-paper-v3.tex`)
- [x] Claim map updated (`docs/claim-map.md`)
- [x] Examples in paper/spec/docs match shipped behavior and versioning

## 4. Governance Gate

- [x] `CONTRIBUTING.md`
- [x] `CODE_OF_CONDUCT.md`
- [x] `SECURITY.md`
- [x] `GOVERNANCE.md`
- [x] `MAINTAINERS.md`
- [x] `CODEOWNERS`
- [x] Issue and PR templates present in `.github/`

## 5. Build and Install Gate

```bash
python -m build
python -m twine check dist/*
```

- [ ] Install from wheel in a clean venv and run smoke import

```bash
python -m venv .venv-smoke
source .venv-smoke/bin/activate  # Windows: .venv-smoke\Scripts\activate
pip install dist/slipcore-<version>-py3-none-any.whl
python -c "from slipcore import format_slip, parse_slip; print(parse_slip(format_slip('a','b','Request','Task')).wire)"
```

## 6. GitHub Release

```bash
VERSION=$(python -c "import slipcore; print(slipcore.__version__)")
git tag -a "v${VERSION}" -m "Slipstream v${VERSION}"
git push origin "v${VERSION}"
gh release create "v${VERSION}" --title "Slipstream v${VERSION}" --notes "See CHANGELOG.md"
```

## 7. Model and Dataset Consistency

- [ ] `MODEL_CARD.md` aligned with v3 Force+Object wire format
- [ ] `data/README.md`, `data/DATASHEET.md`, `data/dataset-metadata.json`, `data/.zenodo.json` aligned
- [ ] Hugging Face namespace links resolve (`anthonym21/...`)

## 8. Publish Targets

- PyPI: https://pypi.org/project/slipcore/
- GitHub: https://github.com/anthony-maio/slipcore
- HF Dataset: https://huggingface.co/datasets/anthonym21/slipstream-tqt
- HF Model: https://huggingface.co/anthonym21/slipstream-glm-z1-9b
