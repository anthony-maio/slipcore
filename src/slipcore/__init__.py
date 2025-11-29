"""SLIPCore - Streamlined Intragent Protocol for LLM agent communication."""

from .protocol import (
    SlipMessage,
    Act,
    FrameType,
    Slot,
    encode_message,
    decode_message,
)

__version__ = "0.1.0"

__all__ = [
    "SlipMessage",
    "Act",
    "FrameType",
    "Slot",
    "encode_message",
    "decode_message",
    "__version__",
]
