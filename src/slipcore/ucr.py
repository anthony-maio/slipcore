"""
Universal Concept Reference (UCR) - The Semantic Manifold

The UCR is a quantized semantic coordinate system for agent communication.
Instead of transmitting embeddings (model-specific, high-dimensional), agents
communicate via positions in a shared, low-dimensional semantic manifold.

Core concepts:
- Dimensions: Semantic axes (action, urgency, domain, polarity)
- Anchors: Named positions in the manifold (common agent intents)
- Quantization: Map agent thoughts to nearest anchor

Architecture:
- Core UCR (0x0000-0x7FFF): Standard anchors, immutable per version
- Extension UCR (0x8000-0xFFFF): Installation-specific, evolvable
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Optional
import json
from pathlib import Path


# ============ Semantic Dimensions ============
# The axes of our semantic manifold. Kept minimal per design.

class Dimension(IntEnum):
    """
    The semantic axes of the UCR manifold.
    Each dimension represents a fundamental aspect of agent communication.
    """
    ACTION = 0      # What type of action: observe, inform, request, propose, evaluate
    POLARITY = 1    # Direction: initiating vs responding, positive vs negative
    DOMAIN = 2      # Context: task, plan, observation, evaluation, control
    URGENCY = 3     # Priority: routine, elevated, critical


# Discrete levels per dimension (kept small for token efficiency)
LEVELS_PER_DIM = 8


# ============ UCR Entry (Anchor) ============

@dataclass
class UCRAnchor:
    """
    A named position in the semantic manifold.

    Attributes:
        index: Unique identifier (0x0000-0xFFFF)
        mnemonic: Single-token wire representation (e.g., "RequestReview")
        canonical: Human-readable description
        coords: Position in the manifold (one value per dimension)
        is_core: True if part of standard UCR, False if extension
    """
    index: int
    mnemonic: str
    canonical: str
    coords: tuple[int, ...]  # One int per dimension, each 0 to LEVELS_PER_DIM-1
    is_core: bool = True

    def __post_init__(self):
        if len(self.coords) != len(Dimension):
            raise ValueError(f"coords must have {len(Dimension)} values, got {len(self.coords)}")
        for i, c in enumerate(self.coords):
            if not (0 <= c < LEVELS_PER_DIM):
                raise ValueError(f"coord[{i}] must be 0-{LEVELS_PER_DIM-1}, got {c}")

    def to_dict(self) -> dict:
        return {
            "index": self.index,
            "mnemonic": self.mnemonic,
            "canonical": self.canonical,
            "coords": list(self.coords),
            "is_core": self.is_core,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "UCRAnchor":
        return cls(
            index=d["index"],
            mnemonic=d["mnemonic"],
            canonical=d["canonical"],
            coords=tuple(d["coords"]),
            is_core=d.get("is_core", True),
        )


# ============ UCR Registry ============

CORE_RANGE_END = 0x8000  # 0x0000-0x7FFF = core, 0x8000-0xFFFF = extensions


@dataclass
class UCR:
    """
    The Universal Concept Reference - a semantic manifold for agent communication.

    Contains both core (standard) anchors and extension (local) anchors.
    Provides lookup by index, mnemonic, and nearest-neighbor by coordinates.
    """
    version: str
    anchors: dict[int, UCRAnchor] = field(default_factory=dict)
    _mnemonic_index: dict[str, int] = field(default_factory=dict, repr=False)

    def add_anchor(self, anchor: UCRAnchor) -> None:
        """Add an anchor to the registry."""
        if anchor.index in self.anchors:
            raise ValueError(f"Anchor index {anchor.index:#06x} already exists")
        if anchor.mnemonic in self._mnemonic_index:
            raise ValueError(f"Anchor mnemonic '{anchor.mnemonic}' already exists")

        # Validate core vs extension range
        if anchor.is_core and anchor.index >= CORE_RANGE_END:
            raise ValueError(f"Core anchor index must be < {CORE_RANGE_END:#06x}")
        if not anchor.is_core and anchor.index < CORE_RANGE_END:
            raise ValueError(f"Extension anchor index must be >= {CORE_RANGE_END:#06x}")

        self.anchors[anchor.index] = anchor
        self._mnemonic_index[anchor.mnemonic] = anchor.index

    def get_by_index(self, index: int) -> Optional[UCRAnchor]:
        """Lookup anchor by numeric index."""
        return self.anchors.get(index)

    def get_by_mnemonic(self, mnemonic: str) -> Optional[UCRAnchor]:
        """Lookup anchor by mnemonic string."""
        idx = self._mnemonic_index.get(mnemonic)
        return self.anchors.get(idx) if idx is not None else None

    def find_nearest(self, coords: tuple[int, ...]) -> UCRAnchor:
        """
        Find the anchor nearest to the given coordinates.
        Uses Manhattan distance for simplicity and speed.
        """
        if not self.anchors:
            raise ValueError("UCR has no anchors")

        best_anchor = None
        best_distance = float('inf')

        for anchor in self.anchors.values():
            distance = sum(abs(a - b) for a, b in zip(anchor.coords, coords))
            if distance < best_distance:
                best_distance = distance
                best_anchor = anchor

        return best_anchor

    def core_anchors(self) -> list[UCRAnchor]:
        """Return all core (standard) anchors."""
        return [a for a in self.anchors.values() if a.is_core]

    def extension_anchors(self) -> list[UCRAnchor]:
        """Return all extension (local) anchors."""
        return [a for a in self.anchors.values() if not a.is_core]

    def next_extension_index(self) -> int:
        """Get the next available extension index."""
        ext_indices = [a.index for a in self.anchors.values() if not a.is_core]
        if not ext_indices:
            return CORE_RANGE_END
        return max(ext_indices) + 1

    def save(self, path: Path) -> None:
        """Save UCR to JSON file."""
        data = {
            "version": self.version,
            "anchors": [a.to_dict() for a in self.anchors.values()],
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path: Path) -> "UCR":
        """Load UCR from JSON file."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        ucr = cls(version=data["version"])
        for anchor_data in data["anchors"]:
            ucr.add_anchor(UCRAnchor.from_dict(anchor_data))
        return ucr

    def __len__(self) -> int:
        return len(self.anchors)


# ============ Base UCR Factory ============

def create_base_ucr() -> UCR:
    """
    Create the base UCR with core anchors for common agent intents.

    Coordinate scheme (4 dimensions, 8 levels each):
    - ACTION:   0=observe, 1=inform, 2=ask, 3=request, 4=propose, 5=commit, 6=evaluate, 7=meta
    - POLARITY: 0=negative, 1-3=declining, 4=neutral, 5-6=positive, 7=strong_positive
    - DOMAIN:   0=task, 1=plan, 2=observation, 3=evaluation, 4=control, 5=resource, 6=error, 7=general
    - URGENCY:  0=background, 1-2=low, 3-4=normal, 5-6=elevated, 7=critical
    """
    ucr = UCR(version="1.0.0")

    # Core coordination anchors
    # Format: (index, mnemonic, canonical, (action, polarity, domain, urgency))

    core_anchors = [
        # === Observations (ACTION=0) ===
        (0x0001, "ObserveState", "Report current system or environment state", (0, 4, 2, 3)),
        (0x0002, "ObserveChange", "Report a detected change", (0, 4, 2, 4)),
        (0x0003, "ObserveError", "Report an observed error condition", (0, 2, 6, 6)),

        # === Information (ACTION=1) ===
        (0x0010, "InformResult", "Share a computed or derived result", (1, 5, 2, 3)),
        (0x0011, "InformStatus", "Provide status update", (1, 4, 0, 3)),
        (0x0012, "InformComplete", "Report task completion", (1, 6, 0, 4)),
        (0x0013, "InformBlocked", "Report being blocked on something", (1, 2, 0, 5)),
        (0x0014, "InformProgress", "Share progress update", (1, 5, 0, 3)),

        # === Questions (ACTION=2) ===
        (0x0020, "AskClarify", "Request clarification on requirements", (2, 4, 1, 4)),
        (0x0021, "AskStatus", "Query current status", (2, 4, 0, 3)),
        (0x0022, "AskPermission", "Request permission to proceed", (2, 4, 4, 4)),
        (0x0023, "AskResource", "Query resource availability", (2, 4, 5, 3)),

        # === Requests (ACTION=3) ===
        (0x0030, "RequestTask", "Request execution of a task", (3, 4, 0, 4)),
        (0x0031, "RequestPlan", "Request creation of a plan", (3, 4, 1, 4)),
        (0x0032, "RequestReview", "Request review of work", (3, 4, 3, 3)),
        (0x0033, "RequestHelp", "Request assistance", (3, 4, 7, 5)),
        (0x0034, "RequestCancel", "Request cancellation", (3, 1, 4, 5)),
        (0x0035, "RequestPriority", "Request priority change", (3, 4, 4, 5)),
        (0x0036, "RequestResource", "Request allocation of resource", (3, 4, 5, 4)),

        # === Proposals (ACTION=4) ===
        (0x0040, "ProposePlan", "Propose a plan for consideration", (4, 5, 1, 4)),
        (0x0041, "ProposeChange", "Propose a modification", (4, 5, 0, 4)),
        (0x0042, "ProposeAlternative", "Propose an alternative approach", (4, 5, 1, 4)),
        (0x0043, "ProposeRollback", "Propose reverting changes", (4, 3, 4, 5)),

        # === Commitments (ACTION=5) ===
        (0x0050, "CommitTask", "Commit to performing a task", (5, 6, 0, 4)),
        (0x0051, "CommitDeadline", "Commit to a deadline", (5, 6, 0, 4)),
        (0x0052, "CommitResource", "Commit resources", (5, 6, 5, 4)),

        # === Evaluations (ACTION=6) ===
        (0x0060, "EvalApprove", "Evaluation: approved/positive", (6, 7, 3, 4)),
        (0x0061, "EvalReject", "Evaluation: rejected/negative", (6, 0, 3, 4)),
        (0x0062, "EvalNeedsWork", "Evaluation: needs revision", (6, 3, 3, 4)),
        (0x0063, "EvalComplete", "Evaluation: work is complete", (6, 6, 3, 4)),
        (0x0064, "EvalBlocked", "Evaluation: blocked by issue", (6, 2, 3, 5)),

        # === Meta/Control (ACTION=7) ===
        (0x0070, "MetaAck", "Acknowledge receipt", (7, 5, 4, 2)),
        (0x0071, "MetaSync", "Synchronization ping", (7, 4, 4, 3)),
        (0x0072, "MetaHandoff", "Hand off responsibility", (7, 4, 4, 4)),
        (0x0073, "MetaEscalate", "Escalate to higher authority", (7, 3, 4, 6)),
        (0x0074, "MetaAbort", "Abort current operation", (7, 0, 4, 7)),

        # === Accept/Reject responses ===
        (0x0080, "Accept", "Accept a proposal or request", (5, 7, 7, 3)),
        (0x0081, "Reject", "Reject a proposal or request", (5, 0, 7, 3)),
        (0x0082, "AcceptWithCondition", "Conditional acceptance", (5, 5, 7, 4)),
        (0x0083, "Defer", "Defer decision", (5, 4, 7, 2)),

        # === Error handling ===
        (0x0090, "ErrorGeneric", "Generic error occurred", (1, 1, 6, 5)),
        (0x0091, "ErrorTimeout", "Operation timed out", (1, 1, 6, 5)),
        (0x0092, "ErrorResource", "Resource unavailable", (1, 1, 6, 5)),
        (0x0093, "ErrorPermission", "Permission denied", (1, 0, 6, 5)),
        (0x0094, "ErrorValidation", "Validation failed", (1, 1, 6, 4)),

        # === Fallback ===
        (0x00FF, "Fallback", "Unquantizable - see payload for natural language", (7, 4, 7, 4)),
    ]

    for index, mnemonic, canonical, coords in core_anchors:
        ucr.add_anchor(UCRAnchor(
            index=index,
            mnemonic=mnemonic,
            canonical=canonical,
            coords=coords,
            is_core=True,
        ))

    return ucr


# Module-level default UCR instance
_default_ucr: Optional[UCR] = None


def get_default_ucr() -> UCR:
    """Get or create the default UCR instance."""
    global _default_ucr
    if _default_ucr is None:
        _default_ucr = create_base_ucr()
    return _default_ucr


def set_default_ucr(ucr: UCR) -> None:
    """Set the default UCR instance."""
    global _default_ucr
    _default_ucr = ucr
