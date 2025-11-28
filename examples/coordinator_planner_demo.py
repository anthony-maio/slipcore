#!/usr/bin/env python
"""Very simple coordinator <-> planner handshake demo (no LangGraph, just structure)."""
from slipcore.protocol import (
    SlipMessage, Act, FrameType, Slot,
    encode_message, decode_message,
)


def coordinator_create_request(goal_id: int, task_id: int) -> str:
    msg = SlipMessage(
        conv_id=10,
        turn=1,
        src=0,  # coord
        dst=1,  # planner
        act=Act.REQUEST,
        frame=FrameType.TASK,
        slots={
            Slot.GOAL_ID: goal_id,
            Slot.TASK_ID: task_id,
            Slot.PRIORITY: 2,
            Slot.TAG: "demo_task",
        },
    )
    return encode_message(msg)


def planner_reply(request_wire: str) -> str:
    req = decode_message(request_wire)
    # In reality you would call the LLM here.
    # For demo, just turn REQUEST/TASK into PROPOSE/PLAN.
    proposal = SlipMessage(
        conv_id=req.conv_id,
        turn=req.turn + 1,
        src=req.dst,
        dst=req.src,
        act=Act.PROPOSE,
        frame=FrameType.PLAN,
        slots={
            Slot.GOAL_ID: req.slots[Slot.GOAL_ID],
            Slot.TASK_ID: req.slots[Slot.TASK_ID],
            Slot.PRIORITY: req.slots.get(Slot.PRIORITY, 0),
            Slot.TAG: "plan_v1",
        },
    )
    return encode_message(proposal)


def main():
    wire_req = coordinator_create_request(goal_id=17, task_id=42)
    print("Coordinator -> Planner:", wire_req)

    wire_prop = planner_reply(wire_req)
    print("Planner -> Coordinator:", wire_prop)

    prop = decode_message(wire_prop)
    print("Decoded proposal:", prop)


if __name__ == "__main__":
    main()
