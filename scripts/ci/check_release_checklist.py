"""Validate release checklist structure and completion for gated sections."""

from __future__ import annotations

import argparse
from pathlib import Path

SECTION_HEADINGS = {
    "1": "## 1. Pre-Release Quality Gate",
    "2": "## 2. Conformance / Migration Gate",
    "3": "## 3. Paper / Spec / Code Sync Gate",
    "4": "## 4. Governance Gate",
    "5": "## 5. Build and Install Gate",
    "7": "## 7. Model and Dataset Consistency",
}


def _collect_section_items(lines: list[str], heading: str) -> list[str]:
    in_section = False
    items: list[str] = []

    for line in lines:
        if line.strip() == heading:
            in_section = True
            continue
        if in_section and line.startswith("## "):
            break
        if in_section and line.strip().startswith("- ["):
            items.append(line.strip())

    return items


def _is_checked(item: str) -> bool:
    return item.startswith("- [x]") or item.startswith("- [X]")


def _parse_sections(raw: str) -> list[str]:
    items = [part.strip() for part in raw.split(",") if part.strip()]
    invalid = [item for item in items if item not in SECTION_HEADINGS]
    if invalid:
        raise ValueError(f"Unknown checklist section id(s): {', '.join(invalid)}")
    return items


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sections",
        default="1,2,3,4,5,7",
        help="Comma-separated checklist section numbers to validate (default: 1,2,3,4,5,7).",
    )
    parser.add_argument(
        "--require-checked",
        action="store_true",
        help="Fail if any checklist item in selected sections is unchecked.",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[2]
    checklist = root / "RELEASE_CHECKLIST.md"
    if not checklist.exists():
        print("Missing RELEASE_CHECKLIST.md")
        return 1

    lines = checklist.read_text(encoding="utf-8").splitlines()
    failures: list[str] = []
    try:
        section_ids = _parse_sections(args.sections)
    except ValueError as exc:
        print(str(exc))
        return 1

    for section_id in section_ids:
        heading = SECTION_HEADINGS[section_id]
        items = _collect_section_items(lines, heading)
        if not items:
            failures.append(f"Section {section_id} missing or has no checklist items.")
            continue

        if args.require_checked:
            unchecked = [item for item in items if not _is_checked(item)]
            if unchecked:
                failures.append(f"Section {section_id} has unchecked items:")
                failures.extend(f"  {item}" for item in unchecked)

    if failures:
        print("Release checklist validation failed:")
        for failure in failures:
            print(f"- {failure}")
        return 1

    if args.require_checked:
        print(f"Release checklist sections {', '.join(section_ids)} are complete.")
    else:
        print(f"Release checklist sections {', '.join(section_ids)} are structurally valid.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
