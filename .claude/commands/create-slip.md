# Create Slipstream Message

Generate a Slipstream v3 wire format message based on the description: $ARGUMENTS

Use the slipcore Python library to create the message. Parse the user's description to determine:
- Source and destination agents
- The appropriate Force token (action verb)
- The appropriate Object token (domain noun)
- Any payload content

Force tokens (12 closed vocabulary):
Observe, Inform, Ask, Request, Propose, Commit, Eval, Meta, Accept, Reject, Error, Fallback

Common Object tokens:
Task, Plan, Review, Help, Status, Complete, Blocked, Progress, Approve, NeedsWork, Ack, Sync, Handoff, Escalate, Generic

Example for "alice asks bob to review the auth code":
```python
from slipcore import format_slip

wire = format_slip("alice", "bob", "Request", "Review", ["auth"])
print(wire)
# -> SLIP v3 alice bob Request Review auth
```

Example for "tell the team the deployment is done":
```python
from slipcore import format_slip

wire = format_slip("devops", "team", "Inform", "Complete", ["deployment"])
print(wire)
# -> SLIP v3 devops team Inform Complete deployment
```

For complex/unquantizable content, use fallback:
```python
from slipcore import format_fallback, FallbackStore

store = FallbackStore()
ref = store.store("check kubernetes pods for memory leaks")
wire = format_fallback("devops", "sre", ref)
print(wire)
# -> SLIP v3 devops sre Fallback Generic ref...
```
