# Parse nSLIP Message

Parse and explain the given nSLIP wire format message: $ARGUMENTS

Use the slipcore Python library to decode the message and provide a human-readable explanation including:
1. The speech act and frame type
2. Source and destination agents
3. Conversation context (conv_id, turn)
4. All slot values with their meanings

Run this Python code to parse:
```python
from slipcore import decode_message, Slot

wire = "$ARGUMENTS"
msg = decode_message(wire)

print(f"Act: {msg.act.name} - {['report info', 'derived belief', 'ask for info', 'request task', 'propose plan', 'commit to plan', 'accept', 'reject', 'evaluate', 'error', 'meta'][msg.act.value]}")
print(f"Frame: {msg.frame.name}")
print(f"From: Agent {msg.src} → Agent {msg.dst}")
print(f"Context: Conv {msg.conv_id}, Turn {msg.turn}")
print("Slots:")
for slot, val in msg.slots.items():
    print(f"  {slot.name}: {val}")
```
