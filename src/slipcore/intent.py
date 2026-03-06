"""Factorized intent model: Force + Object.

Wire format: SLIP v3 <src> <dst> <Force> <Object> [payload...]

Force vocabulary (closed, 12 tokens):
    Observe, Inform, Ask, Request, Propose, Commit, Eval, Meta,
    Accept, Reject, Error, Fallback

Object vocabulary (core + local extensions):
    ~30 core objects + installation-specific extensions
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, unique


@unique
class ForceToken(Enum):
    """Action verbs -- closed vocabulary."""

    OBSERVE = "Observe"
    INFORM = "Inform"
    ASK = "Ask"
    REQUEST = "Request"
    PROPOSE = "Propose"
    COMMIT = "Commit"
    EVAL = "Eval"
    META = "Meta"
    ACCEPT = "Accept"
    REJECT = "Reject"
    ERROR = "Error"
    FALLBACK = "Fallback"

    @property
    def action_coord(self) -> int:
        """Map to ACTION dimension (0-7)."""
        return _FORCE_TO_ACTION.get(self, 3)


_FORCE_TO_ACTION: dict[ForceToken, int] = {
    ForceToken.OBSERVE: 0,
    ForceToken.INFORM: 1,
    ForceToken.ASK: 2,
    ForceToken.REQUEST: 3,
    ForceToken.PROPOSE: 4,
    ForceToken.COMMIT: 5,
    ForceToken.EVAL: 6,
    ForceToken.META: 7,
    ForceToken.ACCEPT: 5,
    ForceToken.REJECT: 5,
    ForceToken.ERROR: 1,
    ForceToken.FALLBACK: 7,
}

FORCE_VALUES: frozenset[str] = frozenset(f.value for f in ForceToken)


@dataclass(frozen=True, slots=True)
class ObjectToken:
    """Domain noun -- core + extensible."""

    mnemonic: str
    canonical: str
    domain_coord: int
    polarity_coord: int
    urgency_coord: int
    is_core: bool = True


# Core object registry
CORE_OBJECTS: dict[str, ObjectToken] = {}


def _reg(mnemonic: str, canonical: str, domain: int, polarity: int, urgency: int) -> None:
    CORE_OBJECTS[mnemonic] = ObjectToken(
        mnemonic=mnemonic,
        canonical=canonical,
        domain_coord=domain,
        polarity_coord=polarity,
        urgency_coord=urgency,
    )


# Register core objects
_reg("State", "Current system state", 2, 4, 3)
_reg("Change", "Detected change", 2, 4, 4)
_reg("Error", "Error condition", 6, 1, 5)
_reg("Result", "Computed result", 2, 5, 3)
_reg("Status", "Status update", 0, 4, 3)
_reg("Complete", "Task completion", 0, 6, 4)
_reg("Blocked", "Blocked on dependency", 0, 2, 5)
_reg("Progress", "Progress update", 0, 5, 3)
_reg("Clarify", "Clarification request", 1, 4, 4)
_reg("Permission", "Permission to proceed", 4, 4, 4)
_reg("Resource", "Resource allocation", 5, 4, 3)
_reg("Task", "Task execution", 0, 4, 4)
_reg("Plan", "Plan creation", 1, 4, 4)
_reg("Review", "Work review", 3, 4, 3)
_reg("Help", "Assistance", 7, 4, 5)
_reg("Cancel", "Cancellation", 4, 1, 5)
_reg("Priority", "Priority change", 4, 4, 5)
_reg("Alternative", "Alternative approach", 1, 5, 4)
_reg("Rollback", "Revert changes", 4, 3, 5)
_reg("Deadline", "Deadline commitment", 0, 6, 4)
_reg("Approve", "Approved evaluation", 3, 7, 4)
_reg("NeedsWork", "Needs revision", 3, 3, 4)
_reg("Ack", "Acknowledge receipt", 4, 5, 2)
_reg("Sync", "Synchronization ping", 4, 4, 3)
_reg("Handoff", "Hand off responsibility", 4, 4, 4)
_reg("Escalate", "Escalate to authority", 4, 3, 6)
_reg("Abort", "Abort operation", 4, 0, 7)
_reg("Condition", "Conditional acceptance", 7, 5, 4)
_reg("Defer", "Deferred decision", 7, 4, 2)
_reg("Timeout", "Operation timed out", 6, 1, 5)
_reg("Validation", "Validation failed", 6, 1, 4)
_reg("Generic", "Generic/unclassified", 7, 4, 4)

Coords = tuple[int, int, int, int]


@dataclass(frozen=True, slots=True)
class Intent:
    """A factorized intent: Force + Object."""

    force: ForceToken
    obj: ObjectToken

    @property
    def mnemonic(self) -> str:
        return f"{self.force.value}{self.obj.mnemonic}"

    @property
    def coords(self) -> Coords:
        action = self.force.action_coord
        domain = self.obj.domain_coord
        polarity = self.obj.polarity_coord
        urgency = self.obj.urgency_coord
        return (action, polarity, domain, urgency)

    @property
    def wire_tokens(self) -> tuple[str, str]:
        return (self.force.value, self.obj.mnemonic)


def resolve_intent(force_str: str, obj_str: str) -> Intent:
    """Resolve Force and Object strings into a validated Intent."""
    try:
        force = ForceToken(force_str)
    except ValueError:
        raise ValueError(f"Unknown force: {force_str!r}")

    obj = CORE_OBJECTS.get(obj_str)
    if obj is None:
        raise ValueError(f"Unknown object: {obj_str!r}")

    return Intent(force=force, obj=obj)


# V2 to V3 migration mapping
V2_TO_V3: dict[str, tuple[str, str]] = {
    "ObserveState": ("Observe", "State"),
    "ObserveChange": ("Observe", "Change"),
    "ObserveError": ("Observe", "Error"),
    "InformResult": ("Inform", "Result"),
    "InformStatus": ("Inform", "Status"),
    "InformComplete": ("Inform", "Complete"),
    "InformBlocked": ("Inform", "Blocked"),
    "InformProgress": ("Inform", "Progress"),
    "AskClarify": ("Ask", "Clarify"),
    "AskStatus": ("Ask", "Status"),
    "AskPermission": ("Ask", "Permission"),
    "AskResource": ("Ask", "Resource"),
    "RequestTask": ("Request", "Task"),
    "RequestPlan": ("Request", "Plan"),
    "RequestReview": ("Request", "Review"),
    "RequestHelp": ("Request", "Help"),
    "RequestCancel": ("Request", "Cancel"),
    "RequestPriority": ("Request", "Priority"),
    "RequestResource": ("Request", "Resource"),
    "ProposePlan": ("Propose", "Plan"),
    "ProposeChange": ("Propose", "Change"),
    "ProposeAlternative": ("Propose", "Alternative"),
    "ProposeRollback": ("Propose", "Rollback"),
    "CommitTask": ("Commit", "Task"),
    "CommitDeadline": ("Commit", "Deadline"),
    "CommitResource": ("Commit", "Resource"),
    "EvalApprove": ("Eval", "Approve"),
    "EvalNeedsWork": ("Eval", "NeedsWork"),
    "MetaAck": ("Meta", "Ack"),
    "MetaSync": ("Meta", "Sync"),
    "MetaHandoff": ("Meta", "Handoff"),
    "MetaEscalate": ("Meta", "Escalate"),
    "MetaAbort": ("Meta", "Abort"),
    "Accept": ("Accept", "Generic"),
    "Reject": ("Reject", "Generic"),
    "AcceptWithCondition": ("Accept", "Condition"),
    "Defer": ("Meta", "Defer"),
    "ErrorGeneric": ("Error", "Generic"),
    "ErrorTimeout": ("Error", "Timeout"),
    "ErrorResource": ("Error", "Resource"),
    "ErrorPermission": ("Error", "Permission"),
    "ErrorValidation": ("Error", "Validation"),
    "Fallback": ("Fallback", "Generic"),
    # v2 had these with different casing too
    "EvalReject": ("Eval", "Review"),
    "EvalComplete": ("Eval", "Complete"),
    "EvalBlocked": ("Eval", "Blocked"),
}


def from_v2_mnemonic(mnemonic: str) -> Intent:
    """Convert a v2 flat mnemonic to a v3 Intent."""
    pair = V2_TO_V3.get(mnemonic)
    if pair is None:
        raise ValueError(f"Unknown v2 mnemonic: {mnemonic!r}")
    return resolve_intent(*pair)
