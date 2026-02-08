"""Universal Concept Reference (UCR) -- semantic manifold registry.

Architecture:
    Core UCR  : 0x0000-0x7FFF -- standard anchors, immutable per version
    Extension : 0x8000-0xFFFF -- installation-specific, evolvable
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from enum import Enum, unique
from pathlib import Path
from typing import Any, Iterator, Optional

from .errors import AnchorNotFoundError, UCRError

CORE_RANGE_END = 0x8000
LEVELS_PER_DIM = 8
Coords = tuple[int, int, int, int]


@unique
class AnchorState(Enum):
    DRAFT = "draft"
    PROPOSED = "proposed"
    APPROVED = "approved"
    ACTIVE = "active"
    DEPRECATED = "deprecated"


@dataclass(slots=True)
class UCRAnchor:
    index: int
    force: str
    obj: str
    canonical: str
    coords: Coords
    is_core: bool = True
    state: AnchorState = AnchorState.ACTIVE
    created_by: str = "core"
    replaced_by: Optional[int] = None

    @property
    def mnemonic(self) -> str:
        return f"{self.force}{self.obj}"

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "index": self.index,
            "force": self.force,
            "obj": self.obj,
            "canonical": self.canonical,
            "coords": list(self.coords),
            "is_core": self.is_core,
            "state": self.state.value,
            "created_by": self.created_by,
        }
        if self.replaced_by is not None:
            d["replaced_by"] = self.replaced_by
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> UCRAnchor:
        return cls(
            index=d["index"],
            force=d["force"],
            obj=d["obj"],
            canonical=d["canonical"],
            coords=tuple(d["coords"]),  # type: ignore[arg-type]
            is_core=d.get("is_core", True),
            state=AnchorState(d.get("state", "active")),
            created_by=d.get("created_by", "core"),
            replaced_by=d.get("replaced_by"),
        )


@dataclass
class UCRAuthority:
    authority_id: str
    ucr_version: str
    ucr_hash: str = ""

    def to_dict(self) -> dict[str, str]:
        return {
            "authority_id": self.authority_id,
            "ucr_version": self.ucr_version,
            "ucr_hash": self.ucr_hash,
        }


@dataclass
class UCR:
    authority: UCRAuthority
    anchors: dict[int, UCRAnchor] = field(default_factory=dict)
    _force_obj_index: dict[tuple[str, str], int] = field(default_factory=dict, repr=False)

    def add_anchor(self, anchor: UCRAnchor) -> None:
        if anchor.index in self.anchors:
            raise UCRError(f"Index {anchor.index:#06x} exists")
        key = (anchor.force, anchor.obj)
        if key in self._force_obj_index:
            raise UCRError(f"Pair {key} exists")
        self.anchors[anchor.index] = anchor
        self._force_obj_index[key] = anchor.index

    def get_by_index(self, index: int) -> Optional[UCRAnchor]:
        return self.anchors.get(index)

    def get_by_force_obj(self, force: str, obj: str) -> Optional[UCRAnchor]:
        idx = self._force_obj_index.get((force, obj))
        return self.anchors.get(idx) if idx is not None else None

    def find_nearest(self, coords: Coords) -> UCRAnchor:
        if not self.anchors:
            raise UCRError("No anchors")
        best: Optional[UCRAnchor] = None
        best_dist = float("inf")
        for anchor in self.anchors.values():
            if anchor.state == AnchorState.DEPRECATED:
                continue
            dist = sum(abs(a - b) for a, b in zip(anchor.coords, coords))
            if dist < best_dist or (dist == best_dist and best is not None and anchor.index < best.index):
                best_dist = dist
                best = anchor
        if best is None:
            raise UCRError("All anchors are deprecated")
        return best

    def core_anchors(self) -> list[UCRAnchor]:
        return [a for a in self.anchors.values() if a.is_core]

    def extension_anchors(self) -> list[UCRAnchor]:
        return [a for a in self.anchors.values() if not a.is_core]

    def next_extension_index(self) -> int:
        ext = [a.index for a in self.anchors.values() if not a.is_core]
        next_idx = max(ext) + 1 if ext else CORE_RANGE_END
        if next_idx > 0xFFFF:
            raise UCRError("Extension address space exhausted (0x8000-0xFFFF)")
        return next_idx

    def propose_extension(
        self,
        force: str,
        obj: str,
        canonical: str,
        coords: Coords,
        created_by: str = "swarm",
    ) -> UCRAnchor:
        anchor = UCRAnchor(
            index=self.next_extension_index(),
            force=force,
            obj=obj,
            canonical=canonical,
            coords=coords,
            is_core=False,
            state=AnchorState.DRAFT,
            created_by=created_by,
        )
        self.add_anchor(anchor)
        return anchor

    def content_hash(self) -> str:
        items = [
            f"{a.index}:{a.force}:{a.obj}:{','.join(str(c) for c in a.coords)}"
            for a in sorted(self.anchors.values(), key=lambda x: x.index)
        ]
        content = "|".join(items)
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def save(self, path: Path) -> None:
        self.authority.ucr_hash = self.content_hash()
        data = {
            "authority": self.authority.to_dict(),
            "anchors": [a.to_dict() for a in self.anchors.values()],
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path: Path) -> UCR:
        with open(path) as f:
            data = json.load(f)
        authority = UCRAuthority(**data["authority"])
        ucr = cls(authority=authority)
        for ad in data["anchors"]:
            ucr.add_anchor(UCRAnchor.from_dict(ad))
        return ucr

    def __len__(self) -> int:
        return len(self.anchors)

    def __iter__(self) -> Iterator[UCRAnchor]:
        return iter(self.anchors.values())


def create_base_ucr(authority_id: str = "slipcore-default") -> UCR:
    """Create the base UCR with core Force-Object anchors."""
    ucr = UCR(authority=UCRAuthority(authority_id=authority_id, ucr_version="3.0.0"))

    core = [
        (0x0001, "Observe", "State", "Report current state", (0, 4, 2, 3)),
        (0x0002, "Observe", "Change", "Report detected change", (0, 4, 2, 4)),
        (0x0003, "Observe", "Error", "Report observed error", (0, 2, 6, 6)),
        (0x0010, "Inform", "Result", "Share computed result", (1, 5, 2, 3)),
        (0x0011, "Inform", "Status", "Provide status update", (1, 4, 0, 3)),
        (0x0012, "Inform", "Complete", "Report task completion", (1, 6, 0, 4)),
        (0x0013, "Inform", "Blocked", "Report being blocked", (1, 2, 0, 5)),
        (0x0014, "Inform", "Progress", "Share progress update", (1, 5, 0, 3)),
        (0x0020, "Ask", "Clarify", "Request clarification", (2, 4, 1, 4)),
        (0x0021, "Ask", "Status", "Query current status", (2, 4, 0, 3)),
        (0x0022, "Ask", "Permission", "Request permission", (2, 4, 4, 4)),
        (0x0023, "Ask", "Resource", "Query resource availability", (2, 4, 5, 3)),
        (0x0030, "Request", "Task", "Request task execution", (3, 4, 0, 4)),
        (0x0031, "Request", "Plan", "Request plan creation", (3, 4, 1, 4)),
        (0x0032, "Request", "Review", "Request work review", (3, 4, 3, 3)),
        (0x0033, "Request", "Help", "Request assistance", (3, 4, 7, 5)),
        (0x0034, "Request", "Cancel", "Request cancellation", (3, 1, 4, 5)),
        (0x0035, "Request", "Priority", "Request priority change", (3, 4, 4, 5)),
        (0x0036, "Request", "Resource", "Request resource allocation", (3, 4, 5, 4)),
        (0x0040, "Propose", "Plan", "Propose a plan", (4, 5, 1, 4)),
        (0x0041, "Propose", "Change", "Propose modification", (4, 5, 0, 4)),
        (0x0042, "Propose", "Alternative", "Propose alternative", (4, 5, 1, 4)),
        (0x0043, "Propose", "Rollback", "Propose reverting", (4, 3, 4, 5)),
        (0x0050, "Commit", "Task", "Commit to task", (5, 6, 0, 4)),
        (0x0051, "Commit", "Deadline", "Commit to deadline", (5, 6, 0, 4)),
        (0x0052, "Commit", "Resource", "Commit resources", (5, 6, 5, 4)),
        (0x0060, "Eval", "Approve", "Evaluation: approved", (6, 7, 3, 4)),
        (0x0061, "Eval", "Review", "Evaluation: under review", (6, 4, 3, 4)),
        (0x0062, "Eval", "NeedsWork", "Evaluation: needs revision", (6, 3, 3, 4)),
        (0x0063, "Eval", "Complete", "Evaluation: work complete", (6, 6, 3, 4)),
        (0x0070, "Meta", "Ack", "Acknowledge receipt", (7, 5, 4, 2)),
        (0x0071, "Meta", "Sync", "Synchronization ping", (7, 4, 4, 3)),
        (0x0072, "Meta", "Handoff", "Hand off responsibility", (7, 4, 4, 4)),
        (0x0073, "Meta", "Escalate", "Escalate to authority", (7, 3, 4, 6)),
        (0x0074, "Meta", "Abort", "Abort operation", (7, 0, 4, 7)),
        (0x0080, "Accept", "Generic", "Accept proposal/request", (5, 7, 7, 3)),
        (0x0081, "Reject", "Generic", "Reject proposal/request", (5, 0, 7, 3)),
        (0x0082, "Accept", "Condition", "Conditional acceptance", (5, 5, 7, 4)),
        (0x0083, "Meta", "Defer", "Defer decision", (5, 4, 7, 2)),
        (0x0090, "Error", "Generic", "Generic error", (1, 1, 6, 5)),
        (0x0091, "Error", "Timeout", "Operation timed out", (1, 1, 6, 5)),
        (0x0092, "Error", "Resource", "Resource unavailable", (1, 1, 6, 5)),
        (0x0093, "Error", "Permission", "Permission denied", (1, 0, 6, 5)),
        (0x0094, "Error", "Validation", "Validation failed", (1, 1, 6, 4)),
        (0x00FF, "Fallback", "Generic", "Unquantizable - see ref", (7, 4, 7, 4)),
    ]

    for index, force, obj, canonical, coords in core:
        ucr.add_anchor(
            UCRAnchor(index=index, force=force, obj=obj, canonical=canonical, coords=coords)
        )

    ucr.authority.ucr_hash = ucr.content_hash()
    return ucr
