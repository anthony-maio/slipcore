"""
Semantic Quantizer - The Think-Quantize-Transmit Engine

Maps agent thoughts (natural language) to UCR anchors.
Supports two modes:
1. Keyword-based (fast, no dependencies)
2. Embedding-based (accurate, requires sentence-transformers)

Also handles:
- Fallback detection (when confidence is too low)
- Usage tracking (for UCR evolution)
- Extension anchor learning
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Callable
from collections import Counter
import re

from .ucr import UCR, UCRAnchor, get_default_ucr, CORE_RANGE_END


# ============ Quantization Result ============

@dataclass
class QuantizeResult:
    """
    Result of quantizing a thought to a UCR anchor.

    Attributes:
        anchor: The matched UCR anchor
        confidence: How well the thought matches (0.0-1.0)
        method: How the match was made ("keyword", "embedding", "fallback")
        alternatives: Other possible matches with their scores
    """
    anchor: UCRAnchor
    confidence: float
    method: str
    alternatives: list[tuple[UCRAnchor, float]] = field(default_factory=list)

    @property
    def is_fallback(self) -> bool:
        return self.anchor.mnemonic == "Fallback"

    @property
    def is_high_confidence(self) -> bool:
        return self.confidence >= 0.7


# ============ Keyword-Based Quantizer ============

# Keyword patterns for each anchor category
_KEYWORD_PATTERNS: dict[str, list[str]] = {
    # Observations
    "ObserveState": ["state", "current", "status", "environment", "system state"],
    "ObserveChange": ["changed", "detected", "noticed", "updated", "modified"],
    "ObserveError": ["error", "exception", "failed", "crash", "bug"],

    # Information
    "InformResult": ["result", "output", "computed", "calculated", "returns"],
    "InformStatus": ["status", "update", "progress", "currently"],
    "InformComplete": ["complete", "finished", "done", "completed", "success"],
    "InformBlocked": ["blocked", "waiting", "stuck", "depends on", "need"],
    "InformProgress": ["progress", "working on", "making progress", "underway"],

    # Questions
    "AskClarify": ["clarify", "what do you mean", "unclear", "confused", "explain"],
    "AskStatus": ["what is the status", "how is", "progress on", "update on"],
    "AskPermission": ["can i", "may i", "permission", "allowed", "okay to"],
    "AskResource": ["available", "resource", "capacity", "do we have"],

    # Requests
    "RequestTask": ["please do", "execute", "perform", "run", "implement"],
    "RequestPlan": ["create a plan", "plan for", "how should we", "strategy"],
    "RequestReview": ["review", "check", "look at", "evaluate", "feedback"],
    "RequestHelp": ["help", "assist", "support", "guidance", "advice"],
    "RequestCancel": ["cancel", "abort", "stop", "nevermind", "forget"],
    "RequestPriority": ["priority", "urgent", "expedite", "escalate"],
    "RequestResource": ["allocate", "provision", "need resource", "require"],

    # Proposals
    "ProposePlan": ["propose", "suggest", "recommendation", "i think we should"],
    "ProposeChange": ["change", "modify", "alter", "adjust"],
    "ProposeAlternative": ["alternative", "instead", "another approach", "option"],
    "ProposeRollback": ["rollback", "revert", "undo", "go back"],

    # Commitments
    "CommitTask": ["i will", "i'll do", "on it", "taking this", "i commit"],
    "CommitDeadline": ["by", "deadline", "eta", "deliver by"],
    "CommitResource": ["allocating", "providing", "assigning"],

    # Evaluations
    "EvalApprove": ["approved", "lgtm", "looks good", "accept", "ship it"],
    "EvalReject": ["rejected", "no", "denied", "not acceptable", "wrong"],
    "EvalNeedsWork": ["needs work", "revise", "changes needed", "almost"],
    "EvalComplete": ["complete", "done", "finished", "all good"],
    "EvalBlocked": ["blocked", "cannot proceed", "impediment"],

    # Meta
    "MetaAck": ["ack", "acknowledged", "got it", "received", "understood"],
    "MetaSync": ["sync", "ping", "alive", "heartbeat"],
    "MetaHandoff": ["handoff", "transfer", "passing to", "your turn"],
    "MetaEscalate": ["escalate", "raise", "need manager", "above my paygrade"],
    "MetaAbort": ["abort", "emergency stop", "halt", "critical failure"],

    # Accept/Reject
    "Accept": ["yes", "accept", "agreed", "confirmed", "affirmative"],
    "Reject": ["no", "reject", "disagree", "refuse", "decline"],
    "AcceptWithCondition": ["yes but", "if", "conditional", "provided that"],
    "Defer": ["later", "defer", "postpone", "not now", "revisit"],

    # Errors
    "ErrorGeneric": ["error", "failed", "exception"],
    "ErrorTimeout": ["timeout", "timed out", "too slow"],
    "ErrorResource": ["resource unavailable", "out of", "exhausted"],
    "ErrorPermission": ["permission denied", "unauthorized", "forbidden"],
    "ErrorValidation": ["invalid", "validation failed", "bad input"],
}


def _keyword_score(thought: str, patterns: list[str]) -> float:
    """Score how well a thought matches keyword patterns."""
    thought_lower = thought.lower()
    matches = 0
    for pattern in patterns:
        if pattern.lower() in thought_lower:
            # Longer patterns are stronger signals
            matches += len(pattern.split())
    # Normalize to 0-1 range (cap at 1.0)
    return min(1.0, matches / 3.0)


class KeywordQuantizer:
    """
    Simple keyword-based quantizer. No ML dependencies.
    Good for bootstrapping and low-latency scenarios.
    """

    def __init__(self, ucr: Optional[UCR] = None, fallback_threshold: float = 0.2):
        self.ucr = ucr or get_default_ucr()
        self.fallback_threshold = fallback_threshold
        self._usage_stats: Counter = Counter()

    def quantize(self, thought: str) -> QuantizeResult:
        """
        Map a natural language thought to the best UCR anchor.

        Args:
            thought: The agent's thought/intent in natural language

        Returns:
            QuantizeResult with the best anchor and confidence score
        """
        scores: list[tuple[UCRAnchor, float]] = []

        for mnemonic, patterns in _KEYWORD_PATTERNS.items():
            anchor = self.ucr.get_by_mnemonic(mnemonic)
            if anchor:
                score = _keyword_score(thought, patterns)
                if score > 0:
                    scores.append((anchor, score))

        # Sort by score descending
        scores.sort(key=lambda x: x[1], reverse=True)

        if not scores or scores[0][1] < self.fallback_threshold:
            # Use fallback
            fallback = self.ucr.get_by_mnemonic("Fallback")
            self._usage_stats["_fallback"] += 1
            return QuantizeResult(
                anchor=fallback,
                confidence=0.0,
                method="fallback",
                alternatives=scores[:3],
            )

        best_anchor, best_score = scores[0]
        self._usage_stats[best_anchor.mnemonic] += 1

        return QuantizeResult(
            anchor=best_anchor,
            confidence=best_score,
            method="keyword",
            alternatives=scores[1:4],
        )

    def get_usage_stats(self) -> dict[str, int]:
        """Get usage statistics for UCR evolution analysis."""
        return dict(self._usage_stats)

    def get_fallback_rate(self) -> float:
        """Get the rate of fallback usage (indicates UCR coverage gaps)."""
        total = sum(self._usage_stats.values())
        if total == 0:
            return 0.0
        return self._usage_stats["_fallback"] / total


# ============ Embedding-Based Quantizer (Optional) ============

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False


class EmbeddingQuantizer:
    """
    Embedding-based quantizer using sentence-transformers.
    More accurate than keyword matching but requires ML dependencies.

    Usage:
        quantizer = EmbeddingQuantizer()
        result = quantizer.quantize("I need someone to review this code")
    """

    def __init__(
        self,
        ucr: Optional[UCR] = None,
        model_name: str = "all-MiniLM-L6-v2",
        fallback_threshold: float = 0.5,
    ):
        if not HAS_NUMPY:
            raise ImportError("numpy is required for EmbeddingQuantizer")

        self.ucr = ucr or get_default_ucr()
        self.fallback_threshold = fallback_threshold
        self._usage_stats: Counter = Counter()

        # Lazy load sentence-transformers
        self._model = None
        self._model_name = model_name
        self._anchor_embeddings: Optional[np.ndarray] = None
        self._anchor_list: list[UCRAnchor] = []

    def _ensure_model(self):
        """Lazy load the embedding model and pre-compute anchor embeddings."""
        if self._model is not None:
            return

        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError(
                "sentence-transformers is required for EmbeddingQuantizer. "
                "Install with: pip install sentence-transformers"
            )

        self._model = SentenceTransformer(self._model_name)

        # Pre-compute embeddings for all anchor canonical texts
        self._anchor_list = list(self.ucr.anchors.values())
        canonical_texts = [a.canonical for a in self._anchor_list]
        self._anchor_embeddings = self._model.encode(canonical_texts, normalize_embeddings=True)

    def quantize(self, thought: str) -> QuantizeResult:
        """
        Map a natural language thought to the best UCR anchor using embeddings.

        Args:
            thought: The agent's thought/intent in natural language

        Returns:
            QuantizeResult with the best anchor and confidence score
        """
        self._ensure_model()

        # Embed the thought
        thought_embedding = self._model.encode([thought], normalize_embeddings=True)[0]

        # Compute cosine similarities (embeddings are normalized, so dot product = cosine)
        similarities = np.dot(self._anchor_embeddings, thought_embedding)

        # Get top matches
        top_indices = np.argsort(similarities)[::-1][:5]
        scores = [(self._anchor_list[i], float(similarities[i])) for i in top_indices]

        best_anchor, best_score = scores[0]

        if best_score < self.fallback_threshold:
            fallback = self.ucr.get_by_mnemonic("Fallback")
            self._usage_stats["_fallback"] += 1
            return QuantizeResult(
                anchor=fallback,
                confidence=best_score,
                method="fallback",
                alternatives=scores[:3],
            )

        self._usage_stats[best_anchor.mnemonic] += 1

        return QuantizeResult(
            anchor=best_anchor,
            confidence=best_score,
            method="embedding",
            alternatives=scores[1:4],
        )

    def get_usage_stats(self) -> dict[str, int]:
        """Get usage statistics for UCR evolution analysis."""
        return dict(self._usage_stats)

    def get_fallback_rate(self) -> float:
        """Get the rate of fallback usage."""
        total = sum(self._usage_stats.values())
        if total == 0:
            return 0.0
        return self._usage_stats["_fallback"] / total


# ============ Auto-selecting Quantizer ============

def create_quantizer(
    ucr: Optional[UCR] = None,
    prefer_embeddings: bool = True,
    fallback_threshold: float = 0.3,
) -> KeywordQuantizer | EmbeddingQuantizer:
    """
    Create the best available quantizer.

    Args:
        ucr: UCR instance to use
        prefer_embeddings: Try to use embedding quantizer if available
        fallback_threshold: Confidence threshold for fallback

    Returns:
        EmbeddingQuantizer if available and preferred, else KeywordQuantizer
    """
    if prefer_embeddings:
        try:
            return EmbeddingQuantizer(ucr=ucr, fallback_threshold=fallback_threshold)
        except ImportError:
            pass

    return KeywordQuantizer(ucr=ucr, fallback_threshold=fallback_threshold)


# ============ High-Level API ============

_default_quantizer: Optional[KeywordQuantizer | EmbeddingQuantizer] = None


def quantize(thought: str) -> QuantizeResult:
    """
    Quantize a thought to a UCR anchor using the default quantizer.

    This is the main entry point for the Think-Quantize-Transmit pattern.

    Example:
        >>> result = quantize("Please review the authentication code")
        >>> result.anchor.mnemonic
        'RequestReview'
        >>> result.confidence
        0.67
    """
    global _default_quantizer
    if _default_quantizer is None:
        _default_quantizer = create_quantizer(prefer_embeddings=False)
    return _default_quantizer.quantize(thought)


def think_quantize_transmit(
    thought: str,
    src: str,
    dst: str,
    ucr: Optional[UCR] = None,
) -> str:
    """
    The complete Think-Quantize-Transmit flow.

    Takes a natural language thought and produces a wire-ready SLIP message.

    Args:
        thought: Natural language intent
        src: Source agent identifier
        dst: Destination agent identifier
        ucr: Optional UCR instance

    Returns:
        Wire-format SLIP message string

    Example:
        >>> wire = think_quantize_transmit(
        ...     "I need someone to check this code for security issues",
        ...     src="developer",
        ...     dst="reviewer"
        ... )
        >>> wire
        'SLIP v1 developer reviewer RequestReview'
    """
    from .protocol import slip, fallback as slip_fallback

    result = quantize(thought)

    if result.is_fallback:
        return slip_fallback(src, dst, thought, ucr)
    else:
        return slip(src, dst, result.anchor.mnemonic, ucr=ucr)


# ============ Smoke Test ============

if __name__ == "__main__":
    print("=== Semantic Quantizer Demo ===\n")

    test_thoughts = [
        "Please review the authentication module for security issues",
        "I've finished implementing the feature",
        "What's the current status of the deployment?",
        "I propose we use Redis for caching instead of Memcached",
        "Yes, that looks good to me",
        "There's an error in the payment processing code",
        "I'm blocked waiting for the API credentials",
        "Check the auth logs for timing anomalies in the OAuth flow",  # Should fallback
    ]

    quantizer = KeywordQuantizer()

    for thought in test_thoughts:
        result = quantizer.quantize(thought)
        status = "FALLBACK" if result.is_fallback else f"{result.confidence:.2f}"
        print(f"Thought: {thought[:50]}...")
        print(f"  → {result.anchor.mnemonic} ({status})")
        if result.alternatives:
            alt_str = ", ".join(f"{a.mnemonic}:{s:.2f}" for a, s in result.alternatives[:2])
            print(f"  Alternatives: {alt_str}")
        print()

    print(f"Fallback rate: {quantizer.get_fallback_rate():.1%}")
    print(f"\nUsage stats: {quantizer.get_usage_stats()}")
