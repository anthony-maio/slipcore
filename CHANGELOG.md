# Changelog

All notable changes to slipcore are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/).

## [3.1.0] - 2026-03-06

### Changed
- Enforced strict fallback parsing: `Fallback` wires now require a 1-16 char ref token
- Enforced closed Force vocabulary for UCR extensions
- Locked core UCR anchor mutation after base UCR construction/load
- Development status upgraded from Alpha to Beta
- Publish workflow now runs full test suite instead of inline v2 smoke test
- Updated package metadata and docs for Python-first adoption flow
- Release checklist updated for v3 API

### Added
- Explicit legacy migration helper: `parse_slip_legacy()` for permissive fallback wire migration
- Governance package: `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`, `SECURITY.md`, `GOVERNANCE.md`, `MAINTAINERS.md`, `CODEOWNERS`, and PR/issue templates
- Paper/spec/code release parity artifact: `docs/claim-map.md`
- Canonical tracked paper source for parity: `docs/paper/slipstream-v3.1.md`
- New adoption docs: `docs/start-here.md` and `docs/migration-v3-1.md`
- LangGraph transport adapter (`slipcore.langgraph`) with encode/decode node and router helpers
- LangGraph adoption guide and runnable integration example
- Release checklist gate script now validates sections 1/2/3/4/5/7 and can enforce full completion on release events
- Unit tests for `finetune.py` template generator
- Unit tests for `errors.py` exception hierarchy
- This changelog

### Removed
- `slipcore_tools` stub package (will return when CLI is built out)
- `tools` optional dependency group

### Fixed
- README quickstart used invalid `KeywordQuantizer.quantize(..., src=..., dst=...)` usage
- `publish.yml` used dead v2 API (`slip`/`decode`) - replaced with `pytest`
- `finetune.py` fallback examples used placeholder `refXXXX` - now generates realistic ref tokens
- Model/dataset metadata and links now consistently target the `anthonym21` Hugging Face namespace

## [3.0.0] - 2025-12-25

### Added
- Factorized Force+Object intent model replacing 46 flat v2 mnemonics
- `ForceToken` enum (12 closed vocabulary) and `ObjectToken` dataclass (30+ core objects)
- Token-aligned wire format: `SLIP v3 <src> <dst> <Force> <Object> [payload...]`
- `KeywordQuantizer` — stdlib-only two-stage classifier
- UCR registry with content hashing, anchor lifecycle (DRAFT -> ACTIVE -> DEPRECATED)
- Conformance test vectors (`valid.jsonl`, `invalid.jsonl`, `roundtrip.jsonl`)
- ABNF grammar specification (`slipstream-v3.abnf`)
- LLM-enhanced dataset generator with 7 provider support (Anthropic, Gemini, OpenAI, Together, Fireworks, DeepSeek, OpenRouter)
- SDK guide and A2A adapter documentation
- UCR semantic reference document (77KB)
- Optional ML package (`slipcore_ml`) with semantic quantizer, coords inferer, clustering
- v2-to-v3 migration script and mapping dict

### Changed
- Wire format version `v2` -> `v3`
- Intent classification reduced from 46-way to 12-way + 30-way factorized problem

## [2.4.0] - 2025-10-15

### Added
- Training dataset (2,283 examples)
- Token-efficient hybrid mnemonics

## [2.0.0] - 2025-09-01

### Added
- Initial semantic quantization protocol
- UCR manifold with 4D coordinate system
- Template-based finetuning dataset generator

[3.1.0]: https://github.com/anthony-maio/slipcore/compare/v2.4.0...v3.1.0
[3.0.0]: https://github.com/anthony-maio/slipcore/compare/v2.4.0...v3.0.0
[2.4.0]: https://github.com/anthony-maio/slipcore/compare/v2.0.0...v2.4.0
[2.0.0]: https://github.com/anthony-maio/slipcore/releases/tag/v2.0.0
