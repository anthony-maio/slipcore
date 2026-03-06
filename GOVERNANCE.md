# Governance

Slipstream uses a maintainer-led model with lightweight RFC process.

## Roles

- Maintainers: merge rights, release authority, security triage, final decisions.
- Contributors: submit issues/PRs/RFCs.

See `MAINTAINERS.md` for current maintainers.

## Decision Process

- Routine fixes/docs: maintainer review + merge.
- Normative protocol changes: require RFC issue + claim-map updates + paper/spec/code/test sync.
- Breaking changes: major version only, with migration notes.

## RFC-lite Process

For protocol/object/governance changes:

1. Open issue labeled `rfc`.
2. Include problem statement, proposal, alternatives, compatibility impact, and security implications.
3. At least one maintainer approval required.
4. Merge only when spec, conformance vectors, code, tests, and paper are updated together.

## Extension Governance Defaults

Extension object growth must be controlled:

- Minimum repeated fallback evidence before proposing new object tokens.
- Rate-limit extension creation in production deployments.
- Prefer human approval before activating new extensions in production.
- Record provenance (`created_by`, timestamps, evidence source).

## Deprecation Policy

- Mark deprecated behavior in docs/changelog first.
- Keep migration helper for one major release window when feasible.
- Remove only in a major release.

## Release Authority

Maintainers control release tags and PyPI publication.
Release checklist in `RELEASE_CHECKLIST.md` is mandatory.

## Paper Source of Truth

- Canonical tracked paper source: `docs/paper/slipstream-v3.1.md`.
- Optional publication artifacts (LaTeX/PDF) may be generated elsewhere, but normative content must first land in the tracked markdown source in this repository.
