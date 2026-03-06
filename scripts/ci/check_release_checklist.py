"""Validate release checklist structure and completion for gated sections."""

from __future__ import annotations

import argparse
from pathlib import Path

SECTION_HEADINGS = {
    "3": "## 3. Paper / Spec / Code Sync Gate",
    "4": "## 4. Governance Gate",
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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--require-checked",
        action="store_true",
        help="Fail if any checklist item in sections 3 and 4 is unchecked.",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[2]
    checklist = root / "RELEASE_CHECKLIST.md"
    if not checklist.exists():
        print("Missing RELEASE_CHECKLIST.md")
        return 1

    lines = checklist.read_text(encoding="utf-8").splitlines()
    failures: list[str] = []

    for section_id, heading in SECTION_HEADINGS.items():
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
        print("Release checklist sections 3 and 4 are complete.")
    else:
        print("Release checklist sections 3 and 4 are structurally valid.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
