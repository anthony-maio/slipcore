"""Smoke tests for README/docs usage snippets."""

from __future__ import annotations

from slipcore import (
    format_fallback,
    format_slip,
    parse_slip,
    parse_slip_legacy,
    quantize,
)


def main() -> int:
    wire = format_slip("alice", "bob", "Request", "Review", ["auth"])
    assert wire == "SLIP v3 alice bob Request Review auth"

    msg = parse_slip(wire)
    assert msg.force == "Request"
    assert msg.obj == "Review"
    assert msg.payload == ("auth",)

    q_wire = quantize("Please review the authentication code", src="dev", dst="reviewer")
    assert q_wire.startswith("SLIP v3 dev reviewer ")

    f_wire = format_fallback("qa", "planner", "ref7f3a")
    f_msg = parse_slip(f_wire)
    assert f_msg.fallback_ref == "ref7f3a"

    legacy = parse_slip_legacy("SLIP v3 qa planner Fallback Generic")
    assert legacy.fallback_ref == "reflegacy"

    print("Docs smoke tests passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
