# Create nSLIP Message

Generate an nSLIP wire format message based on the description: $ARGUMENTS

Use the slipcore Python library to create and encode the message. Parse the user's description to determine:
- Act type (REQUEST, PROPOSE, INFORM, EVAL, etc.)
- Frame type (TASK, PLAN, OBSERVATION, EVALUATION)
- Source and destination agents
- Relevant slots (goal_id, task_id, priority, status, tag, etc.)

Common agent IDs:
- 0 = Coordinator
- 1 = Planner
- 2 = Executor
- 3 = Critic

Example for "coordinator requests planner to implement auth, priority 2":
```python
from slipcore import SlipMessage, Act, FrameType, Slot, encode_message

msg = SlipMessage(
    conv_id=1, turn=1, src=0, dst=1,
    act=Act.REQUEST, frame=FrameType.TASK,
    slots={Slot.GOAL_ID: 1, Slot.TASK_ID: 1, Slot.PRIORITY: 2, Slot.TAG: "implement_auth"}
)
print(encode_message(msg))
```
