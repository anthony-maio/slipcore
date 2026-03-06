"""Keyword-based quantizer (stdlib-only).

Two-stage classifier: keywords -> Force, then keywords -> Object.
All embedding/ML code has been moved to slipcore_ml.
"""

from __future__ import annotations

import hashlib
from collections import Counter
from dataclasses import dataclass

from .events import FallbackEmitted, FallbackStore, Quantized, emit
from .wire import format_fallback, format_slip

# ============ Two-Stage Keyword Patterns ============

# Stage 1: Force classification keywords
FORCE_KEYWORDS: dict[str, list[str]] = {
    "Observe": [
        "i noticed", "i see", "detected", "observed", "state is",
        "current state", "environment", "system state", "changed to",
    ],
    "Inform": [
        "result", "output", "status", "update", "progress",
        "complete", "finished", "done", "completed", "success",
        "blocked", "waiting", "stuck", "working on", "currently",
        "fyi", "heads up", "let you know",
    ],
    "Ask": [
        "what is", "how is", "can i", "may i", "could you tell",
        "what do you mean", "unclear", "clarify", "explain",
        "is it okay", "permission", "allowed", "available",
    ],
    "Request": [
        "please do", "execute", "perform", "run", "implement",
        "review", "check", "look at", "feedback", "help",
        "assist", "support", "cancel", "abort", "stop",
        "allocate", "provision", "create a plan", "plan for",
    ],
    "Propose": [
        "propose", "suggest", "recommendation", "i think we should",
        "how about", "we could", "alternative", "instead",
        "i suggest", "i propose", "option",
    ],
    "Commit": [
        "i will", "i'll do", "on it", "taking this", "i commit",
        "allocating", "providing", "assigning", "by deadline",
    ],
    "Eval": [
        "approved", "lgtm", "looks good", "ship it",
        "needs work", "revise", "changes needed",
        "rejected", "not acceptable",
    ],
    "Meta": [
        "ack", "acknowledged", "got it", "received", "understood",
        "sync", "ping", "heartbeat", "handoff", "transfer",
        "passing to", "your turn", "escalate", "raise",
        "abort", "emergency", "halt",
    ],
    "Accept": [
        "yes", "accept", "agreed", "confirmed", "affirmative",
        "yes but", "conditional", "provided that",
    ],
    "Reject": [
        "no", "reject", "disagree", "refuse", "decline",
    ],
    "Error": [
        "error occurred", "failed", "exception", "timeout",
        "timed out", "resource unavailable", "permission denied",
        "invalid", "validation failed",
    ],
}

# Stage 2: Object classification keywords
OBJECT_KEYWORDS: dict[str, list[str]] = {
    "State": ["state", "environment", "system state", "current"],
    "Change": ["changed", "detected", "modified", "updated"],
    "Error": ["error", "exception", "crash", "bug", "failed"],
    "Result": ["result", "output", "computed", "calculated", "returns"],
    "Status": ["status", "update", "progress on", "currently"],
    "Complete": ["complete", "finished", "done", "completed", "success"],
    "Blocked": ["blocked", "waiting", "stuck", "depends on", "need"],
    "Progress": ["progress", "working on", "making progress", "underway"],
    "Clarify": ["clarify", "what do you mean", "unclear", "confused", "explain"],
    "Permission": ["permission", "can i", "may i", "allowed", "okay to"],
    "Resource": ["resource", "capacity", "available", "allocate", "provision"],
    "Task": ["task", "execute", "perform", "run", "implement", "do"],
    "Plan": ["plan", "strategy", "how should we", "create a plan"],
    "Review": ["review", "check", "look at", "evaluate", "feedback", "pull request"],
    "Help": ["help", "assist", "support", "guidance", "advice"],
    "Cancel": ["cancel", "abort", "stop", "nevermind"],
    "Priority": ["priority", "urgent", "expedite"],
    "Alternative": ["alternative", "instead", "another approach", "option"],
    "Rollback": ["rollback", "revert", "undo", "go back"],
    "Deadline": ["deadline", "eta", "deliver by", "by when"],
    "Approve": ["approved", "lgtm", "looks good", "accept", "ship it"],
    "NeedsWork": ["needs work", "revise", "changes needed", "almost"],
    "Ack": ["ack", "acknowledged", "got it", "received", "understood"],
    "Sync": ["sync", "ping", "alive", "heartbeat"],
    "Handoff": ["handoff", "transfer", "passing to", "your turn"],
    "Escalate": ["escalate", "raise", "need manager", "above my paygrade"],
    "Abort": ["abort", "emergency stop", "halt", "critical failure"],
    "Condition": ["yes but", "if", "conditional", "provided that"],
    "Defer": ["later", "defer", "postpone", "not now", "revisit"],
    "Timeout": ["timeout", "timed out", "too slow"],
    "Validation": ["invalid", "validation", "bad input"],
    "Generic": [],
}


def _keyword_score(text: str, patterns: list[str]) -> float:
    """Score how well text matches keyword patterns."""
    text_lower = text.lower()
    matches = 0
    for pattern in patterns:
        if pattern.lower() in text_lower:
            matches += len(pattern.split())
    return min(1.0, matches / 3.0)


@dataclass
class QuantizeResult:
    """Result of quantizing a thought."""

    force: str
    obj: str
    confidence: float
    method: str
    is_fallback: bool = False


class KeywordQuantizer:
    """Two-stage keyword-based quantizer. No ML dependencies."""

    def __init__(self, fallback_threshold: float = 0.2) -> None:
        self.fallback_threshold = fallback_threshold
        self._usage_stats: Counter[str] = Counter()
        self.fallback_store = FallbackStore()

    def quantize(self, thought: str) -> QuantizeResult:
        """Map a thought to Force + Object using keyword matching."""
        # Stage 1: Classify Force
        force_scores: list[tuple[str, float]] = []
        for force, patterns in FORCE_KEYWORDS.items():
            score = _keyword_score(thought, patterns)
            if score > 0:
                force_scores.append((force, score))

        force_scores.sort(key=lambda x: x[1], reverse=True)

        if not force_scores or force_scores[0][1] < self.fallback_threshold:
            self._usage_stats["_fallback"] += 1
            return QuantizeResult(
                force="Fallback",
                obj="Generic",
                confidence=0.0,
                method="fallback",
                is_fallback=True,
            )

        best_force = force_scores[0][0]
        force_confidence = force_scores[0][1]

        # Stage 2: Classify Object
        obj_scores: list[tuple[str, float]] = []
        for obj_name, patterns in OBJECT_KEYWORDS.items():
            score = _keyword_score(thought, patterns)
            if score > 0:
                obj_scores.append((obj_name, score))

        obj_scores.sort(key=lambda x: x[1], reverse=True)

        if obj_scores:
            best_obj = obj_scores[0][0]
            obj_confidence = obj_scores[0][1]
        else:
            best_obj = "Generic"
            obj_confidence = 0.1

        combined_confidence = (force_confidence + obj_confidence) / 2.0
        self._usage_stats[f"{best_force}.{best_obj}"] += 1

        return QuantizeResult(
            force=best_force,
            obj=best_obj,
            confidence=combined_confidence,
            method="keyword",
        )

    def get_usage_stats(self) -> dict[str, int]:
        return dict(self._usage_stats)

    def get_fallback_rate(self) -> float:
        total = sum(self._usage_stats.values())
        if total == 0:
            return 0.0
        return self._usage_stats["_fallback"] / total


_default_quantizer: KeywordQuantizer | None = None


def get_default_quantizer() -> KeywordQuantizer:
    """Return the module-level default quantizer (created on first use)."""
    global _default_quantizer
    if _default_quantizer is None:
        _default_quantizer = KeywordQuantizer()
    return _default_quantizer


def quantize(thought: str, src: str, dst: str) -> str:
    """Quantize a thought to a SLIP v3 wire message.

    Uses the keyword quantizer (stdlib-only, no ML dependencies).
    The default quantizer persists across calls, so fallback refs
    stored in its FallbackStore remain retrievable.
    For embedding-based quantization, use slipcore_ml.SemanticQuantizer.
    """
    q = get_default_quantizer()
    result = q.quantize(thought)

    input_hash = hashlib.sha256(thought.encode()).hexdigest()[:12]

    if result.is_fallback:
        ref = q.fallback_store.store(thought)
        emit(FallbackEmitted(
            ref=ref,
            reason="low_confidence",
            confidence=result.confidence,
            raw_text_hash=input_hash,
        ))
        return format_fallback(src, dst, ref)

    emit(Quantized(
        input_hash=input_hash,
        force=result.force,
        obj=result.obj,
        confidence=result.confidence,
        mode="keyword",
    ))

    return format_slip(src, dst, result.force, result.obj)


def think_quantize_transmit(thought: str, src: str, dst: str) -> str:
    """The complete Think-Quantize-Transmit flow.

    Takes a natural language thought and produces a wire-ready SLIP message.

    Example:
        >>> think_quantize_transmit(
        ...     "Please review the auth code",
        ...     src="dev", dst="reviewer"
        ... )
        'SLIP v3 dev reviewer Request Review'
    """
    return quantize(thought, src, dst)
