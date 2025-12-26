"""
UCR Extension Layer - Dynamic Local Anchors

The extension layer allows installations to:
1. Add custom anchors for domain-specific concepts (0x8000-0xFFFF range)
2. Track fallback usage to identify candidates for new anchors
3. Learn new anchors from repeated fallback patterns (K-means or greedy clustering)
4. Export extension data for potential promotion to core UCR

Architecture:
- Core UCR (0x0000-0x7FFF): Immutable standard
- Extension UCR (0x8000-0xFFFF): Installation-specific, evolvable

Enhanced with v3's K-means clustering for better extension learning.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, List, Callable
from collections import Counter
from pathlib import Path
import json
import hashlib

from .ucr import UCR, UCRAnchor, CORE_RANGE_END, get_default_ucr, Dimension, LEVELS_PER_DIM


# ============ Optional Dependencies ============

try:
    import numpy as np
    _HAS_NUMPY = True
except ImportError:
    np = None  # type: ignore
    _HAS_NUMPY = False

try:
    from sklearn.cluster import MiniBatchKMeans
    _HAS_SKLEARN = True
except ImportError:
    MiniBatchKMeans = None  # type: ignore
    _HAS_SKLEARN = False


# ============ Fallback Tracker ============

@dataclass
class FallbackEvent:
    """Record of a fallback (unquantized) message."""
    thought: str
    src: str
    dst: str
    timestamp: float = 0.0
    nearest_anchor: Optional[str] = None
    nearest_score: float = 0.0


class FallbackTracker:
    """
    Tracks fallback events to identify gaps in UCR coverage.
    Used to inform extension anchor creation and core UCR evolution.
    """

    def __init__(self, max_events: int = 10000):
        self.max_events = max_events
        self.events: list[FallbackEvent] = []
        self._pattern_counts: Counter = Counter()

    def record(self, event: FallbackEvent) -> None:
        """Record a fallback event."""
        self.events.append(event)
        if len(self.events) > self.max_events:
            self.events = self.events[-self.max_events:]

        # Extract key phrases for pattern detection
        self._extract_patterns(event.thought)

    def _extract_patterns(self, thought: str) -> None:
        """Extract recurring patterns from fallback thoughts."""
        words = thought.lower().split()
        # Track 2-grams and 3-grams
        for n in [2, 3]:
            for i in range(len(words) - n + 1):
                ngram = " ".join(words[i:i+n])
                self._pattern_counts[ngram] += 1

    def get_top_patterns(self, n: int = 20) -> list[tuple[str, int]]:
        """Get the most common fallback patterns (candidates for new anchors)."""
        return self._pattern_counts.most_common(n)

    def get_stats(self) -> dict:
        """Get fallback statistics."""
        return {
            "total_events": len(self.events),
            "unique_patterns": len(self._pattern_counts),
            "top_patterns": self.get_top_patterns(10),
        }

    def clear(self) -> None:
        """Clear all tracked events."""
        self.events.clear()
        self._pattern_counts.clear()


# ============ Extension Manager ============

def _generate_mnemonic(canonical: str) -> str:
    """Generate a CamelCase mnemonic from canonical text."""
    words = canonical.split()[:4]  # Max 4 words
    mnemonic = "".join(w.capitalize() for w in words if w.isalpha())
    # Ensure it starts with uppercase
    if mnemonic and not mnemonic[0].isupper():
        mnemonic = mnemonic.capitalize()
    return mnemonic or "CustomAnchor"


def _estimate_coords(canonical: str) -> tuple[int, ...]:
    """
    Estimate semantic coordinates from canonical text.
    This is a simple heuristic - real implementations would use embeddings.
    """
    text_lower = canonical.lower()

    # ACTION dimension (0-7)
    action = 3  # default: request
    if any(w in text_lower for w in ["observe", "see", "notice", "detect"]):
        action = 0
    elif any(w in text_lower for w in ["inform", "tell", "report", "update"]):
        action = 1
    elif any(w in text_lower for w in ["ask", "question", "what", "how", "why"]):
        action = 2
    elif any(w in text_lower for w in ["request", "please", "need", "want"]):
        action = 3
    elif any(w in text_lower for w in ["propose", "suggest", "recommend"]):
        action = 4
    elif any(w in text_lower for w in ["commit", "will", "promise", "guarantee"]):
        action = 5
    elif any(w in text_lower for w in ["evaluate", "review", "assess", "check"]):
        action = 6
    elif any(w in text_lower for w in ["meta", "sync", "ack", "handoff"]):
        action = 7

    # POLARITY dimension (0-7)
    polarity = 4  # neutral
    if any(w in text_lower for w in ["no", "reject", "deny", "refuse", "fail"]):
        polarity = 1
    elif any(w in text_lower for w in ["yes", "accept", "approve", "good"]):
        polarity = 6

    # DOMAIN dimension (0-7)
    domain = 7  # general
    if any(w in text_lower for w in ["task", "do", "execute", "implement"]):
        domain = 0
    elif any(w in text_lower for w in ["plan", "strategy", "approach"]):
        domain = 1
    elif any(w in text_lower for w in ["observe", "see", "state", "current"]):
        domain = 2
    elif any(w in text_lower for w in ["evaluate", "review", "quality"]):
        domain = 3
    elif any(w in text_lower for w in ["control", "meta", "system"]):
        domain = 4
    elif any(w in text_lower for w in ["resource", "memory", "cpu", "disk"]):
        domain = 5
    elif any(w in text_lower for w in ["error", "exception", "fail"]):
        domain = 6

    # URGENCY dimension (0-7)
    urgency = 4  # normal
    if any(w in text_lower for w in ["critical", "urgent", "emergency", "asap"]):
        urgency = 7
    elif any(w in text_lower for w in ["important", "priority", "soon"]):
        urgency = 5
    elif any(w in text_lower for w in ["low", "background", "when possible"]):
        urgency = 1

    return (action, polarity, domain, urgency)


class ExtensionManager:
    """
    Manages the extension layer of a UCR installation.

    Enhanced with v3's clustering capabilities:
    - K-means clustering when sklearn available
    - Greedy cosine clustering fallback
    - Centroid computation for new anchors

    Handles:
    - Adding custom local anchors
    - Learning from fallback patterns (ML-enhanced)
    - Exporting extensions for promotion consideration
    """

    def __init__(
        self,
        ucr: Optional[UCR] = None,
        embed_batch: Optional[Callable] = None,
        greedy_sim_threshold: float = 0.78,
    ):
        self.ucr = ucr or get_default_ucr()
        self.fallback_tracker = FallbackTracker()
        self._extension_file: Optional[Path] = None
        self._embed_batch = embed_batch  # Optional embedding function
        self.greedy_sim_threshold = greedy_sim_threshold

    def add_extension(
        self,
        canonical: str,
        mnemonic: Optional[str] = None,
        coords: Optional[tuple[int, ...]] = None,
        centroid: Optional[List[float]] = None,
    ) -> UCRAnchor:
        """
        Add a new extension anchor.

        Args:
            canonical: Human-readable description of the concept
            mnemonic: Wire-format token (auto-generated if not provided)
            coords: Semantic coordinates (auto-estimated if not provided)
            centroid: Embedding vector (computed if embed_batch available)

        Returns:
            The newly created anchor
        """
        if mnemonic is None:
            mnemonic = _generate_mnemonic(canonical)

        # Ensure mnemonic is unique
        base_mnemonic = mnemonic
        counter = 1
        while self.ucr.get_by_mnemonic(mnemonic) is not None:
            mnemonic = f"{base_mnemonic}{counter}"
            counter += 1

        if coords is None:
            coords = _estimate_coords(canonical)

        # Compute centroid if embedding function available
        if centroid is None and self._embed_batch is not None and _HAS_NUMPY:
            embeds = self._embed_batch([canonical])
            centroid = embeds[0].tolist() if len(embeds) > 0 else None

        index = self.ucr.next_extension_index()

        anchor = UCRAnchor(
            index=index,
            mnemonic=mnemonic,
            canonical=canonical,
            coords=coords,
            is_core=False,
            centroid=centroid,
        )

        self.ucr.add_anchor(anchor)
        return anchor

    def record_fallback(
        self,
        thought: str,
        src: str,
        dst: str,
        nearest_anchor: Optional[str] = None,
        nearest_score: float = 0.0,
    ) -> None:
        """Record a fallback event for pattern analysis."""
        event = FallbackEvent(
            thought=thought,
            src=src,
            dst=dst,
            nearest_anchor=nearest_anchor,
            nearest_score=nearest_score,
        )
        self.fallback_tracker.record(event)

    def suggest_extensions(self, min_count: int = 5) -> list[str]:
        """
        Suggest new extension anchors based on fallback patterns.

        Returns canonical descriptions for frequently occurring patterns.
        """
        suggestions = []
        for pattern, count in self.fallback_tracker.get_top_patterns(20):
            if count >= min_count:
                # Convert pattern to canonical description
                canonical = f"Agent intent: {pattern}"
                suggestions.append(canonical)
        return suggestions

    def auto_learn(self, min_count: int = 10) -> list[UCRAnchor]:
        """
        Automatically create extension anchors from frequent fallback patterns.

        Uses pattern-based learning (n-gram detection).
        For ML-based learning, use propose_extensions_ml().

        Args:
            min_count: Minimum occurrences before creating an anchor

        Returns:
            List of newly created anchors
        """
        new_anchors = []
        for pattern, count in self.fallback_tracker.get_top_patterns(10):
            if count >= min_count:
                canonical = f"Agent intent: {pattern}"
                anchor = self.add_extension(canonical)
                new_anchors.append(anchor)
        return new_anchors

    def propose_extensions_ml(
        self,
        fallback_texts: List[str],
        min_cluster_size: int = 3,
        max_new: int = 32,
    ) -> List[UCRAnchor]:
        """
        Learn new anchors from fallback traffic using ML clustering.

        This is the v3-style extension learning:
        - Embed all fallback texts
        - Cluster them (K-means if sklearn available, else greedy cosine)
        - For each sufficiently large cluster:
            - Choose exemplar closest to centroid
            - Infer coordinates
            - Mint new anchor

        Args:
            fallback_texts: List of fallback messages to cluster
            min_cluster_size: Minimum messages in a cluster to create anchor
            max_new: Maximum number of new anchors to create

        Returns:
            List of proposed new anchors (not yet added to UCR)
        """
        if not _HAS_NUMPY:
            return []

        buf = [t for t in fallback_texts if t and t.strip()]
        if len(buf) < min_cluster_size:
            return []

        if self._embed_batch is None:
            # No embeddings available, fall back to pattern-based
            return []

        # Embed all fallback texts
        embeds = self._embed_batch(buf)

        # Normalize rows
        def _norm_rows(mat):
            norms = np.linalg.norm(mat, axis=1, keepdims=True)
            return mat / (norms + 1e-12)

        embeds = _norm_rows(np.asarray(embeds, dtype=np.float32))

        # Cluster
        clusters: List[List[int]] = []

        if _HAS_SKLEARN and len(buf) >= (min_cluster_size * 2):
            # Use MiniBatchKMeans
            k = max(1, min(max_new, len(buf) // max(1, min_cluster_size)))
            km = MiniBatchKMeans(n_clusters=k, n_init="auto", random_state=42)
            km.fit(embeds)
            for i in range(k):
                idxs = np.where(km.labels_ == i)[0].tolist()
                if idxs:
                    clusters.append(idxs)
        else:
            # Greedy cosine clustering (no sklearn)
            centroids_list: List["np.ndarray"] = []
            for i, v in enumerate(embeds):
                placed = False
                for c_idx, c in enumerate(centroids_list):
                    if float(np.dot(v, c)) >= self.greedy_sim_threshold:
                        clusters[c_idx].append(i)
                        # Update centroid (mean then renormalize)
                        new_c = np.mean(embeds[clusters[c_idx]], axis=0)
                        n = np.linalg.norm(new_c)
                        centroids_list[c_idx] = new_c / (n + 1e-12) if n > 0 else new_c
                        placed = True
                        break
                if not placed:
                    clusters.append([i])
                    centroids_list.append(v.copy())

        # Generate anchors from clusters
        new_anchors: List[UCRAnchor] = []
        minted = 0

        for idxs in clusters:
            if len(idxs) < min_cluster_size:
                continue
            if minted >= max_new:
                break

            # Cluster centroid
            cluster_embeds = embeds[idxs]
            centroid = np.mean(cluster_embeds, axis=0)
            n = np.linalg.norm(centroid)
            centroid = centroid / (n + 1e-12) if n > 0 else centroid

            # Find exemplar (closest to centroid)
            dists = np.linalg.norm(cluster_embeds - centroid, axis=1)
            exemplar = buf[idxs[int(np.argmin(dists))]].strip()

            # Infer coordinates
            coords = _estimate_coords(exemplar)

            # Generate mnemonic
            mnemonic = _generate_mnemonic(exemplar)
            # Add EXT_ prefix for extension anchors
            mnemonic = f"EXT_{mnemonic}"

            # Check uniqueness
            base_mnemonic = mnemonic
            counter = 1
            while self.ucr.get_by_mnemonic(mnemonic) is not None:
                mnemonic = f"{base_mnemonic}{counter}"
                counter += 1

            # Get next index
            next_idx = self.ucr.next_extension_index()
            if next_idx > 0xFFFF:
                break  # Extension range exhausted

            anchor = UCRAnchor(
                index=next_idx,
                mnemonic=mnemonic,
                canonical=exemplar,
                coords=coords,
                is_core=False,
                centroid=centroid.tolist(),
            )

            new_anchors.append(anchor)
            minted += 1

        return new_anchors

    def learn_and_register(
        self,
        fallback_texts: List[str],
        min_cluster_size: int = 3,
        max_new: int = 32,
    ) -> List[UCRAnchor]:
        """
        Learn new anchors from fallbacks and register them to the UCR.

        Convenience method combining propose_extensions_ml + add to UCR.

        Args:
            fallback_texts: List of fallback messages
            min_cluster_size: Minimum cluster size
            max_new: Maximum new anchors

        Returns:
            List of newly registered anchors
        """
        proposed = self.propose_extensions_ml(fallback_texts, min_cluster_size, max_new)
        registered = []
        for anchor in proposed:
            try:
                self.ucr.add_anchor(anchor)
                registered.append(anchor)
            except ValueError:
                # Skip duplicates
                pass
        return registered

    def export_extensions(self, path: Path) -> None:
        """Export extension anchors for promotion consideration."""
        extensions = self.ucr.extension_anchors()
        data = {
            "ucr_version": self.ucr.version,
            "extensions": [a.to_dict() for a in extensions],
            "fallback_stats": self.fallback_tracker.get_stats(),
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def import_extensions(self, path: Path) -> int:
        """
        Import extension anchors from a file.

        Returns the number of anchors imported.
        """
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        count = 0
        for anchor_data in data.get("extensions", []):
            try:
                # Re-index to avoid collisions
                anchor_data["index"] = self.ucr.next_extension_index()
                anchor = UCRAnchor.from_dict(anchor_data)
                anchor = UCRAnchor(
                    index=anchor.index,
                    mnemonic=anchor.mnemonic,
                    canonical=anchor.canonical,
                    coords=anchor.coords,
                    is_core=False,
                )
                self.ucr.add_anchor(anchor)
                count += 1
            except ValueError:
                # Skip duplicates or invalid entries
                pass
        return count

    def get_stats(self) -> dict:
        """Get extension layer statistics."""
        return {
            "core_anchors": len(self.ucr.core_anchors()),
            "extension_anchors": len(self.ucr.extension_anchors()),
            "fallback_stats": self.fallback_tracker.get_stats(),
        }


# ============ Module-Level Convenience ============

_default_extension_manager: Optional[ExtensionManager] = None


def get_extension_manager() -> ExtensionManager:
    """Get or create the default extension manager."""
    global _default_extension_manager
    if _default_extension_manager is None:
        _default_extension_manager = ExtensionManager()
    return _default_extension_manager


# ============ Smoke Test ============

if __name__ == "__main__":
    print("=== Extension Layer Demo ===\n")

    manager = ExtensionManager()

    # Add a custom extension
    anchor = manager.add_extension(
        canonical="Request code review with security focus",
        mnemonic="RequestSecurityReview",
    )
    print(f"Added extension: {anchor.mnemonic} (0x{anchor.index:04X})")
    print(f"  Canonical: {anchor.canonical}")
    print(f"  Coords: {anchor.coords}\n")

    # Simulate some fallback events
    fallback_thoughts = [
        "check the kubernetes pod logs",
        "check the kubernetes pod logs",
        "check the kubernetes pod logs",
        "review the terraform configuration",
        "analyze memory leak in service",
        "check the kubernetes pod logs",
        "check the kubernetes pod logs",
    ]

    for thought in fallback_thoughts:
        manager.record_fallback(thought, "dev", "ops")

    print("Fallback patterns detected:")
    for pattern, count in manager.fallback_tracker.get_top_patterns(5):
        print(f"  '{pattern}': {count} occurrences")

    print(f"\nSuggested extensions: {manager.suggest_extensions(min_count=3)}")
    print(f"\nStats: {manager.get_stats()}")
