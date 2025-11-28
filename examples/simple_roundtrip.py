#!/usr/bin/env python
"""Simple roundtrip example: encode and decode a SLIP message."""
from slipcore.protocol import (
    SlipMessage, Act, FrameType, Slot,
    encode_message, decode_message,
)


def main():
    msg = SlipMessage(
        conv_id=1,
        turn=0,
        src=0,
        dst=1,
        act=Act.REQUEST,
        frame=FrameType.TASK,
        slots={
            Slot.GOAL_ID: 1,
            Slot.TASK_ID: 1,
            Slot.PRIORITY: 1,
            Slot.TAG: "hello_world",
        },
    )

    wire = encode_message(msg)
    print("Wire:", wire)

    parsed = decode_message(wire)
    print("Parsed:", parsed)


if __name__ == "__main__":
    main()
