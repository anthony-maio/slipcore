# Create Slipstream Message

Generate a Slipstream wire format message based on the description: $ARGUMENTS

Use the slipcore Python library to create the message. Parse the user's description to determine:
- Source and destination agents
- The appropriate semantic anchor (intent)
- Any payload content

Common anchors:
- RequestTask, RequestReview, RequestHelp, RequestPlan
- InformComplete, InformProgress, InformBlocked, InformStatus
- ProposePlan, ProposeChange, ProposeAlternative
- EvalApprove, EvalReject, EvalNeedsWork
- Accept, Reject, MetaAck, MetaHandoff

Example for "alice asks bob to review the auth code":
```python
from slipcore import slip

wire = slip("alice", "bob", "RequestReview", ["auth_code"])
print(wire)
# -> SLIP v1 alice bob RequestReview auth_code
```

Example for "tell the team the deployment is done":
```python
from slipcore import slip

wire = slip("devops", "team", "InformComplete", ["deployment"])
print(wire)
# -> SLIP v1 devops team InformComplete deployment
```

For complex/unquantizable content, use fallback:
```python
from slipcore import fallback

wire = fallback("alice", "bob", "check kubernetes pods for memory leaks")
print(wire)
# -> SLIP v1 alice bob Fallback check kubernetes pods for memory leaks
```
