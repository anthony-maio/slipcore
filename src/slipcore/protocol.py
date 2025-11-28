from __future__ import annotations
from dataclasses import dataclass
from enum import IntEnum
from typing import Dict, Any

# ========= Base62 integer encoding =========

_BASE62_ALPHABET = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
_BASE62_INDEX = {ch: i for i, ch in enumerate(_BASE62_ALPHABET)}


def encode_int_base62(n: int) -> str:
    if n < 0:
        raise ValueError("encode_int_base62 only supports non-negative integers")
    if n == 0:
        return _BASE62_ALPHABET[0]
    out = []
    while n > 0:
        n, rem = divmod(n, 62)
        out.append(_BASE62_ALPHABET[rem])
    return "".join(reversed(out))


def decode_int_base62(s: str) -> int:
    n = 0
    for ch in s:
        n = n * 62 + _BASE62_INDEX[ch]
    return n


# ========= Protocol enums =========

class Act(IntEnum):
    OBSERVE = 0
    INFORM = 1
    ASK = 2
    REQUEST = 3
    PROPOSE = 4
    COMMIT = 5
    ACCEPT = 6
    REJECT = 7
    EVAL = 8
    ERROR = 9
    META = 10


class FrameType(IntEnum):
    TASK = 0
    PLAN = 1
    OBSERVATION = 2
    EVALUATION = 3
    CONTROL = 4


class Slot(IntEnum):
    # IDs / references
    GOAL_ID = 0         # g
    TASK_ID = 1         # k
    PARENT_TASK_ID = 2  # p
    RESULT_ID = 3       # r

    # scalar/meta
    PRIORITY = 10       # q
    SCORE = 11          # s
    STATUS = 12         # u
    ERROR_CODE = 13     # e

    # optional short comment / tag (kept small, not full text)
    TAG = 20            # t


# Map slot → prefix char for encoding
_SLOT_PREFIX: Dict[Slot, str] = {
    Slot.GOAL_ID: "g",
    Slot.TASK_ID: "k",
    Slot.PARENT_TASK_ID: "p",
    Slot.RESULT_ID: "r",
    Slot.PRIORITY: "q",
    Slot.SCORE: "s",
    Slot.STATUS: "u",
    Slot.ERROR_CODE: "e",
    Slot.TAG: "t",
}
_PREFIX_SLOT = {v: k for k, v in _SLOT_PREFIX.items()}


@dataclass
class SlipMessage:
    conv_id: int
    turn: int
    src: int
    dst: int
    act: Act
    frame: FrameType
    slots: Dict[Slot, Any]

    def __repr__(self) -> str:
        fields = (
            f"conv={self.conv_id}",
            f"turn={self.turn}",
            f"src={self.src}",
            f"dst={self.dst}",
            f"act={self.act.name}",
            f"frame={self.frame.name}",
        )
        return f"<SlipMessage {' '.join(fields)} slots={self.slots}>"


# ========= nSLIP encoding / decoding =========

_FIELD_PREFIXES = {
    "act": "a",
    "frame": "f",
    "conv": "c",
    "src": "S",  # uppercase to avoid conflict with SCORE slot
    "dst": "d",
    "turn": "T",  # uppercase to avoid conflict with TAG slot
}
_PREFIX_TO_FIELD = {v: k for k, v in _FIELD_PREFIXES.items()}

START_MARKER = "@"
END_MARKER = "#"
SEPARATOR = "|"


def _encode_value(v: Any) -> str:
    """
    For internal agent protocol we mostly care about ints (IDs, small enums).
    Strings are allowed but should remain short (tags/status labels).
    """
    if isinstance(v, int):
        return encode_int_base62(v)
    if isinstance(v, str):
        escaped = v.replace("\\", "\\\\").replace('"', '\\"')
        return f'"{escaped}"'
    raise TypeError(f"Unsupported slot value type: {type(v).__name__}")


def _decode_value(s: str) -> Any:
    if not s:
        return ""
    if s[0] == '"' and s[-1] == '"':
        inner = s[1:-1]
        return inner.replace('\\"', '"').replace("\\\\", "\\")
    # treat as base62 int
    return decode_int_base62(s)


def encode_message(msg: SlipMessage) -> str:
    """
    Encode SlipMessage into a compact nSLIP string:
      @a<act>|f<frame>|c<conv>|s<src>|d<dst>|t<turn>|<slot-prefix><value>|...#
    """
    parts = []
    parts.append(_FIELD_PREFIXES["act"] + encode_int_base62(int(msg.act)))
    parts.append(_FIELD_PREFIXES["frame"] + encode_int_base62(int(msg.frame)))
    parts.append(_FIELD_PREFIXES["conv"] + encode_int_base62(msg.conv_id))
    parts.append(_FIELD_PREFIXES["src"] + encode_int_base62(msg.src))
    parts.append(_FIELD_PREFIXES["dst"] + encode_int_base62(msg.dst))
    parts.append(_FIELD_PREFIXES["turn"] + encode_int_base62(msg.turn))

    for slot, value in msg.slots.items():
        prefix = _SLOT_PREFIX[slot]
        parts.append(prefix + _encode_value(value))

    return START_MARKER + SEPARATOR.join(parts) + END_MARKER


def decode_message(s: str) -> SlipMessage:
    """
    Decode nSLIP string back into SlipMessage.
    """
    s = s.strip()
    if not s.startswith(START_MARKER) or not s.endswith(END_MARKER):
        raise ValueError("Invalid nSLIP message framing")

    core = s[len(START_MARKER):-len(END_MARKER)]
    if not core:
        raise ValueError("Empty nSLIP message")

    tokens = core.split(SEPARATOR)
    header: Dict[str, int] = {}
    slots: Dict[Slot, Any] = {}

    for tok in tokens:
        if not tok:
            continue
        key_char = tok[0]
        payload = tok[1:]

        if key_char in _PREFIX_TO_FIELD:
            field_name = _PREFIX_TO_FIELD[key_char]
            header[field_name] = decode_int_base62(payload)
        elif key_char in _PREFIX_SLOT:
            slot = _PREFIX_SLOT[key_char]
            slots[slot] = _decode_value(payload)
        else:
            raise ValueError(f"Unknown prefix '{key_char}' in token '{tok}'")

    required = ("act", "frame", "conv", "src", "dst", "turn")
    for r in required:
        if r not in header:
            raise ValueError(f"Missing required header field '{r}'")

    return SlipMessage(
        conv_id=header["conv"],
        turn=header["turn"],
        src=header["src"],
        dst=header["dst"],
        act=Act(header["act"]),
        frame=FrameType(header["frame"]),
        slots=slots,
    )


if __name__ == "__main__":
    # Basic smoke test
    msg = SlipMessage(
        conv_id=3,
        turn=1,
        src=0,
        dst=1,
        act=Act.REQUEST,
        frame=FrameType.TASK,
        slots={
            Slot.GOAL_ID: 17,
            Slot.TASK_ID: 42,
            Slot.PRIORITY: 2,
            Slot.TAG: "refactor_auth",
        },
    )

    wire = encode_message(msg)
    print("Wire:", wire)

    parsed = decode_message(wire)
    print("Parsed:", parsed)
