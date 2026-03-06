"""UCR Extension Layer -- Dynamic Local Anchors (v3).

Updated for the Force-Object factorized model.
ML-based clustering has been moved to slipcore_ml.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .events import ExtensionProposed, emit
from .ucr import UCR, Coords, UCRAnchor, create_base_ucr


@dataclass
class FallbackEvent:
    """Record of a fallback (unquantized) message."""

    thought: str
    src: str
    dst: str
    timestamp: float = 0.0
    nearest_force: Optional[str] = None
    nearest_obj: Optional[str] = None
    nearest_score: float = 0.0


class FallbackTracker:
    """Tracks fallback events to identify gaps in UCR coverage."""

    def __init__(self, max_events: int = 10000) -> None:
        self.max_events = max_events
        self.events: list[FallbackEvent] = []
        self._pattern_counts: Counter[str] = Counter()

    def record(self, event: FallbackEvent) -> None:
        self.events.append(event)
        if len(self.events) > self.max_events:
            self.events = self.events[-self.max_events:]
        self._extract_patterns(event.thought)

    def _extract_patterns(self, thought: str) -> None:
        words = thought.lower().split()
        for n in [2, 3]:
            for i in range(len(words) - n + 1):
                ngram = " ".join(words[i : i + n])
                self._pattern_counts[ngram] += 1

    def get_top_patterns(self, n: int = 20) -> list[tuple[str, int]]:
        return self._pattern_counts.most_common(n)

    def get_stats(self) -> dict:
        return {
            "total_events": len(self.events),
            "unique_patterns": len(self._pattern_counts),
            "top_patterns": self.get_top_patterns(10),
        }

    def clear(self) -> None:
        self.events.clear()
        self._pattern_counts.clear()


class ExtensionManager:
    """Manages the extension layer of a UCR installation (v3 Force-Object model)."""

    def __init__(self, ucr: Optional[UCR] = None) -> None:
        self.ucr = ucr if ucr is not None else create_base_ucr()
        self.fallback_tracker = FallbackTracker()

    def add_extension(
        self,
        force: str,
        obj: str,
        canonical: str,
        coords: Coords,
        created_by: str = "swarm",
    ) -> UCRAnchor:
        """Add a new extension anchor using Force-Object model."""
        anchor = self.ucr.propose_extension(
            force=force,
            obj=obj,
            canonical=canonical,
            coords=coords,
            created_by=created_by,
        )
        emit(ExtensionProposed(
            force=force,
            obj=obj,
            created_by=created_by,
        ))
        return anchor

    def record_fallback(
        self,
        thought: str,
        src: str,
        dst: str,
        nearest_force: Optional[str] = None,
        nearest_obj: Optional[str] = None,
        nearest_score: float = 0.0,
    ) -> None:
        event = FallbackEvent(
            thought=thought,
            src=src,
            dst=dst,
            nearest_force=nearest_force,
            nearest_obj=nearest_obj,
            nearest_score=nearest_score,
        )
        self.fallback_tracker.record(event)

    def suggest_extensions(self, min_count: int = 5) -> list[str]:
        suggestions = []
        for pattern, count in self.fallback_tracker.get_top_patterns(20):
            if count >= min_count:
                suggestions.append(f"Agent intent: {pattern}")
        return suggestions

    def export_extensions(self, path: Path) -> None:
        extensions = self.ucr.extension_anchors()
        data = {
            "ucr_version": self.ucr.authority.ucr_version,
            "extensions": [a.to_dict() for a in extensions],
            "fallback_stats": self.fallback_tracker.get_stats(),
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def get_stats(self) -> dict:
        return {
            "core_anchors": len(self.ucr.core_anchors()),
            "extension_anchors": len(self.ucr.extension_anchors()),
            "fallback_stats": self.fallback_tracker.get_stats(),
        }
