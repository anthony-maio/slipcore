"""Minimal external link checker for release docs."""

from __future__ import annotations

import re
import urllib.error
import urllib.request
from pathlib import Path

FILES = [
    "README.md",
    "MODEL_CARD.md",
    "data/README.md",
    "RELEASE_CHECKLIST.md",
]

URL_RE = re.compile(r"\[[^\]]+\]\((https?://[^)]+)\)")
ACCEPTABLE_STATUS = {200, 201, 202, 203, 204, 301, 302, 303, 307, 308, 403, 429}


def request_status(url: str) -> int:
    req = urllib.request.Request(url, method="HEAD")
    try:
        with urllib.request.urlopen(req, timeout=15) as response:  # noqa: S310
            return int(response.status)
    except urllib.error.HTTPError as err:
        if err.code == 405:
            get_req = urllib.request.Request(url, method="GET")
            with urllib.request.urlopen(get_req, timeout=15) as response:  # noqa: S310
                return int(response.status)
        return int(err.code)


def main() -> int:
    root = Path(__file__).resolve().parents[2]
    urls: set[str] = set()

    for rel in FILES:
        text = (root / rel).read_text(encoding="utf-8")
        urls.update(URL_RE.findall(text))

    failures: list[tuple[str, int]] = []
    for url in sorted(urls):
        status = request_status(url)
        print(f"{status} {url}")
        if status not in ACCEPTABLE_STATUS:
            failures.append((url, status))

    if failures:
        print("\nBroken/unexpected link responses:")
        for url, status in failures:
            print(f"- {status} {url}")
        return 1

    print("All checked links returned acceptable status codes.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
