# Security Policy

## Supported Versions

Security fixes are applied to the latest major release branch.

| Version | Supported |
|---|---|
| 4.x | yes |
| <4.0 | no |

## Reporting a Vulnerability

Do not open public issues for unpatched vulnerabilities.

Report privately to: anthony@making-minds.ai

Include:

- Affected version(s)
- Reproduction steps
- Impact assessment
- Suggested mitigation (if known)

## Response Targets

- Initial acknowledgment: within 3 business days
- Triage decision: within 7 business days
- Patch timeline: communicated after triage

## Scope Notes

Slipstream wire payloads are untrusted input. Always validate and isolate fallback-store content in downstream systems.
