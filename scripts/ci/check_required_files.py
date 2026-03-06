"""Fail if required governance and parity files are missing."""

from __future__ import annotations

from pathlib import Path

REQUIRED_FILES = [
    "CONTRIBUTING.md",
    "CODE_OF_CONDUCT.md",
    "SECURITY.md",
    "GOVERNANCE.md",
    "MAINTAINERS.md",
    "CODEOWNERS",
    ".github/PULL_REQUEST_TEMPLATE.md",
    ".github/ISSUE_TEMPLATE/bug_report.md",
    ".github/ISSUE_TEMPLATE/feature_request.md",
    "docs/claim-map.md",
    "docs/paper/slipstream-v3.1.md",
    "docs/start-here.md",
    "docs/langgraph-guide.md",
    "docs/migration-v3-1.md",
]


def main() -> int:
    root = Path(__file__).resolve().parents[2]
    missing = [p for p in REQUIRED_FILES if not (root / p).exists()]

    if missing:
        print("Missing required release files:")
        for item in missing:
            print(f"- {item}")
        return 1

    print("All required release files are present.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
