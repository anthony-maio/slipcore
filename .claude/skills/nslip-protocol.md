# nSLIP Protocol Skill

Use this skill when working with SLIPCore messages, parsing nSLIP wire format, or communicating between agents in the SLIP protocol.

## Protocol Overview

nSLIP (nano-SLIP) is a compact wire format for agent-to-agent messaging. Messages look like:
```
@a3|f0|c1|S0|d1|T1|g1|k1|q2|t"refactor_auth"#
```

## Message Structure

Every message has:
- `@` start marker, `#` end marker
- `|` field separator
- Header fields + slot fields

### Header Fields (required)

| Prefix | Field | Description |
|--------|-------|-------------|
| `a` | act | Speech act (0-10) |
| `f` | frame | Frame type (0-4) |
| `c` | conv_id | Conversation/session ID |
| `S` | src | Source agent ID |
| `d` | dst | Destination agent ID |
| `T` | turn | Turn number |

### Acts (speech acts)

| Code | Name | Use |
|------|------|-----|
| 0 | OBSERVE | Report new info about world/state |
| 1 | INFORM | Derived info / belief |
| 2 | ASK | Request information |
| 3 | REQUEST | Request a task/operation |
| 4 | PROPOSE | Propose a plan/option |
| 5 | COMMIT | Commit to a plan/task |
| 6 | ACCEPT | Accept a plan/request |
| 7 | REJECT | Reject a plan/request |
| 8 | EVAL | Evaluate a plan/result |
| 9 | ERROR | Report an error |
| 10 | META | Protocol/capability update |

### Frame Types

| Code | Name | Use |
|------|------|-----|
| 0 | TASK | Task definition/status |
| 1 | PLAN | Plan with steps |
| 2 | OBSERVATION | Observation about environment |
| 3 | EVALUATION | Evaluation of something |
| 4 | CONTROL | Meta/protocol level |

### Slot Fields (optional)

| Prefix | Slot | Type |
|--------|------|------|
| `g` | GOAL_ID | int |
| `k` | TASK_ID | int |
| `p` | PARENT_TASK_ID | int |
| `r` | RESULT_ID | int |
| `q` | PRIORITY | int (1-3) |
| `s` | SCORE | int (0-10) |
| `u` | STATUS | string |
| `e` | ERROR_CODE | int |
| `t` | TAG | string |

## Value Encoding

### Integers: Base62
```
0-9 → 0-9
10-35 → A-Z
36-61 → a-z
62 → 10, 63 → 11, etc.
```

Examples: `0`=0, `A`=10, `z`=61, `10`=62, `H`=17, `g`=42

### Strings: Quoted
```
"some_tag"
"with \"escaped\" quotes"
```

## Common Patterns

### Coordinator → Planner (REQUEST/TASK)
```
@a3|f0|c1|S0|d1|T1|g1|k1|q2|t"implement_auth"#
```
- act=3 (REQUEST), frame=0 (TASK)
- src=0 (coordinator), dst=1 (planner)
- goal_id=1, task_id=1, priority=2

### Planner → Coordinator (PROPOSE/PLAN)
```
@a4|f1|c1|S1|d0|T2|g1|k1|q2|t"plan_v1"#
```
- act=4 (PROPOSE), frame=1 (PLAN)
- src=1 (planner), dst=0 (coordinator)

### Executor → Coordinator (INFORM/OBSERVATION)
```
@a1|f2|c1|S2|d0|T3|g1|k1|r1|u"done"|t"exec_status"#
```
- act=1 (INFORM), frame=2 (OBSERVATION)
- result_id=1, status="done"

### Critic → Coordinator (EVAL/EVALUATION)
```
@a8|f3|c1|S3|d0|T4|g1|k1|s8|t"ok"#
```
- act=8 (EVAL), frame=3 (EVALUATION)
- score=8, tag="ok"

## Agent IDs (convention)

| ID | Agent |
|----|-------|
| 0 | Coordinator |
| 1 | Planner |
| 2 | Executor |
| 3 | Critic |

## Parsing Tips

1. Strip `@` prefix and `#` suffix
2. Split by `|`
3. First char of each token is the field prefix
4. Rest is the value (base62 int or quoted string)
5. Header fields use uppercase `S` (src) and `T` (turn) to avoid conflicts with slots

## Python Usage

```python
from slipcore import SlipMessage, Act, FrameType, Slot, encode_message, decode_message

# Create a message
msg = SlipMessage(
    conv_id=1, turn=1, src=0, dst=1,
    act=Act.REQUEST, frame=FrameType.TASK,
    slots={Slot.GOAL_ID: 1, Slot.TASK_ID: 1, Slot.TAG: "my_task"}
)

# Encode to wire format
wire = encode_message(msg)  # "@a3|f0|c1|S0|d1|T1|g1|k1|t\"my_task\"#"

# Decode from wire format
parsed = decode_message(wire)
```

## When to Use This Skill

- Parsing or generating nSLIP messages
- Debugging agent communication
- Understanding message flows between agents
- Writing code that integrates with SLIPCore
