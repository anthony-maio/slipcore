"""
Slipstream (SLIP) - Semantic Quantization for Efficient Multi-Agent Coordination

A protocol that achieves token efficiency not through syntactic compression,
but through Semantic Quantization - transmitting pointers to concepts in a
shared Universal Concept Reference (UCR) rather than raw text.

Key components:
- UCR: Universal Concept Reference - the semantic manifold
- Protocol: Token-aligned wire format (no special characters)
- Quantizer: Think-Quantize-Transmit engine
- Extensions: Dynamic local anchor learning

Quick start:
    >>> from slipcore import slip, decode, quantize
    >>>
    >>> # Direct message creation
    >>> wire = slip("alice", "bob", "RequestReview")
    >>> print(wire)
    'SLIP v1 alice bob RequestReview'
    >>>
    >>> # Think-Quantize-Transmit pattern
    >>> from slipcore import think_quantize_transmit
    >>> wire = think_quantize_transmit(
    ...     "Please review the auth code for security issues",
    ...     src="dev", dst="reviewer"
    ... )
"""

__version__ = "2.0.0"

# Core UCR components
from .ucr import (
    UCR,
    UCRAnchor,
    Dimension,
    LEVELS_PER_DIM,
    CORE_RANGE_END,
    create_base_ucr,
    get_default_ucr,
    set_default_ucr,
)

# Protocol - wire format
from .protocol import (
    SlipMessage,
    encode,
    decode,
    slip,
    fallback,
    MessageBuilder,
    PROTOCOL_MARKER,
    DEFAULT_VERSION,
)

# Quantizer - Think-Quantize-Transmit
from .quantizer import (
    QuantizeResult,
    KeywordQuantizer,
    quantize,
    think_quantize_transmit,
    create_quantizer,
)

# Extensions - dynamic local anchors
from .extensions import (
    ExtensionManager,
    FallbackTracker,
    get_extension_manager,
)

# Finetuning - dataset generation for training agents
from .finetune import (
    generate_training_examples,
    generate_dataset,
    TrainingExample,
    SYSTEM_PROMPT_BASIC,
    SYSTEM_PROMPT_DETAILED,
)

# LLM-enhanced dataset generation (requires httpx)
try:
    from .finetune_llm import (
        generate_dataset_llm,
        LLMExample,
        PROVIDERS as LLM_PROVIDERS,
    )
    _HAS_LLM_FINETUNE = True
except ImportError:
    _HAS_LLM_FINETUNE = False

__all__ = [
    # Version
    "__version__",
    # UCR
    "UCR",
    "UCRAnchor",
    "Dimension",
    "LEVELS_PER_DIM",
    "CORE_RANGE_END",
    "create_base_ucr",
    "get_default_ucr",
    "set_default_ucr",
    # Protocol
    "SlipMessage",
    "encode",
    "decode",
    "slip",
    "fallback",
    "MessageBuilder",
    "PROTOCOL_MARKER",
    "DEFAULT_VERSION",
    # Quantizer
    "QuantizeResult",
    "KeywordQuantizer",
    "quantize",
    "think_quantize_transmit",
    "create_quantizer",
    # Extensions
    "ExtensionManager",
    "FallbackTracker",
    "get_extension_manager",
    # Finetuning
    "generate_training_examples",
    "generate_dataset",
    "TrainingExample",
    "SYSTEM_PROMPT_BASIC",
    "SYSTEM_PROMPT_DETAILED",
    # LLM-enhanced finetuning (optional)
    "generate_dataset_llm",
    "LLMExample",
    "LLM_PROVIDERS",
]
