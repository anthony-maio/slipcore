"""Slipcore: Token-efficient agent-to-agent coordination protocol.

Usage:
    >>> import slipcore
    >>> wire = slipcore.format_slip("planner", "exec", "Request", "Plan")
    >>> wire
    'SLIP v3 planner exec Request Plan'
    >>> msg = slipcore.parse_slip(wire)
    >>> msg.force, msg.obj
    ('Request', 'Plan')
    >>> slipcore.render_human(msg)
    '[planner -> exec] Request Plan: "Request plan creation"'
"""

__version__ = "4.0.0"

from .errors import (
    AnchorNotFoundError,
    MissingExtraError,
    PolicyViolationError,
    SlipError,
    UCRError,
    UCRVersionMismatchError,
    WireParseError,
    WireValidationError,
)
from .events import (
    ExtensionProposed,
    FallbackEmitted,
    FallbackStore,
    Quantized,
    WireReceived,
    WireSent,
    clear_sinks,
    emit,
    register_sink,
    unregister_sink,
)
from .extensions import ExtensionManager, FallbackTracker
from .intent import (
    CORE_OBJECTS,
    V2_TO_V3,
    ForceToken,
    Intent,
    ObjectToken,
    from_v2_mnemonic,
    resolve_intent,
)
from .quantizer import (
    KeywordQuantizer,
    QuantizeResult,
    get_default_quantizer,
    quantize,
    think_quantize_transmit,
)
from .render import render_human, render_log_line
from .ucr import (
    UCR,
    AnchorState,
    UCRAnchor,
    UCRAuthority,
    create_base_ucr,
)
from .wire import (
    WIRE_VERSION,
    SlipMessage,
    format_fallback,
    format_slip,
    parse_slip,
    parse_slip_legacy,
    validate_wire,
)

__all__ = [
    "__version__",
    # Wire
    "SlipMessage",
    "format_slip",
    "format_fallback",
    "parse_slip",
    "parse_slip_legacy",
    "validate_wire",
    "WIRE_VERSION",
    # Intent
    "ForceToken",
    "ObjectToken",
    "Intent",
    "CORE_OBJECTS",
    "resolve_intent",
    "from_v2_mnemonic",
    "V2_TO_V3",
    # UCR
    "UCR",
    "UCRAnchor",
    "UCRAuthority",
    "AnchorState",
    "create_base_ucr",
    # Render
    "render_human",
    "render_log_line",
    # Events
    "emit",
    "register_sink",
    "unregister_sink",
    "clear_sinks",
    "FallbackStore",
    "WireSent",
    "WireReceived",
    "Quantized",
    "FallbackEmitted",
    "ExtensionProposed",
    # Quantizer
    "KeywordQuantizer",
    "QuantizeResult",
    "quantize",
    "think_quantize_transmit",
    "get_default_quantizer",
    # Extensions
    "ExtensionManager",
    "FallbackTracker",
    # Errors
    "SlipError",
    "WireParseError",
    "WireValidationError",
    "UCRError",
    "AnchorNotFoundError",
    "UCRVersionMismatchError",
    "PolicyViolationError",
    "MissingExtraError",
]
