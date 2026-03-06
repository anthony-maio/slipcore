"""Coordinate inference using embeddings (requires slipcore[ml]).

Ported from v2's CoordsInferer with prototype embedding similarity.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Optional

import numpy as np


@dataclass(frozen=True)
class SemanticCoords:
    """4D logical position of an intent in the semantic manifold."""

    action: str
    polarity: int
    domain: str
    urgency: int


ACTION_MAP = {
    "OBS": 0, "INF": 1, "ASK": 2, "REQ": 3,
    "PROP": 4, "COMMIT": 5, "EVAL": 6, "META": 7, "CMD": 3,
}
DOMAIN_MAP = {
    "TASK": 0, "PLAN": 1, "OBS": 2, "EVAL": 3, "CTRL": 4,
    "RES": 5, "ERR": 6, "GEN": 7, "QA": 3, "INFRA": 5,
    "AUTH": 4, "DOC": 1, "META": 4,
}


def semantic_coords_to_tuple(sc: SemanticCoords) -> tuple[int, int, int, int]:
    """Convert SemanticCoords to UCR numeric tuple."""
    action = ACTION_MAP.get(sc.action.upper(), 3)
    domain = DOMAIN_MAP.get(sc.domain.upper(), 7)
    polarity = {-1: 1, 0: 4, 1: 6}.get(sc.polarity, 4)
    urgency = min(7, max(0, sc.urgency * 2 + 1))
    return (action, polarity, domain, urgency)


class CoordsInferer:
    """Assigns (Action, Polarity, Domain, Urgency) to text.

    Hybrid approach: heuristics + optional prototype embedding similarity.
    """

    def __init__(self, embed_batch: Optional[Callable] = None) -> None:
        self._embed_batch = embed_batch
        self._proto_action: Dict[str, np.ndarray] = {}
        self._proto_domain: Dict[str, np.ndarray] = {}

        self._action_phrases = {
            "REQ": [
                "Please do this task.",
                "Can you help with this request?",
                "I need you to do something.",
            ],
            "INF": [
                "FYI, here is a status update.",
                "I finished the task.",
                "This is an informational update.",
            ],
            "EVAL": [
                "Please review and evaluate this.",
                "Assess the quality of this work.",
                "Give a critique of this design.",
            ],
            "CMD": [
                "Do this immediately.",
                "Execute this command.",
                "Run the operation now.",
            ],
            "OBS": [
                "I noticed something changed.",
                "The current state is...",
                "I observed an issue.",
            ],
            "PROP": [
                "I suggest we do this.",
                "Here's my proposal.",
                "We could try this approach.",
            ],
        }
        self._domain_phrases = {
            "TASK": ["Assign a task ticket.", "Work item status update."],
            "QA": ["Request code review.", "Review pull request."],
            "INFRA": ["Scale the Kubernetes cluster.", "Deploy infrastructure change."],
            "AUTH": ["OAuth login issue.", "Authentication and authorization."],
            "ERR": ["System error occurred.", "Critical failure and outage."],
            "DOC": ["Update documentation.", "Write technical docs."],
            "META": ["Discuss process and coordination.", "Team protocol and planning."],
            "GEN": ["General conversation.", "Generic request or update."],
        }

    def prime(self) -> None:
        """Compute prototype embeddings (if embedder available)."""
        if not self._embed_batch:
            return

        def _norm(vec: np.ndarray) -> np.ndarray:
            n = np.linalg.norm(vec)
            return vec / (n + 1e-12) if n > 0 else vec

        for a, phrases in self._action_phrases.items():
            vecs = self._embed_batch([" ".join(phrases)])
            self._proto_action[a] = _norm(np.asarray(vecs[0], dtype=np.float32))

        for d, phrases in self._domain_phrases.items():
            vecs = self._embed_batch([" ".join(phrases)])
            self._proto_domain[d] = _norm(np.asarray(vecs[0], dtype=np.float32))

    def infer(self, text: str, vec: Optional[np.ndarray] = None) -> SemanticCoords:
        """Infer semantic coordinates from text."""
        low = text.strip().lower()

        # Urgency heuristic
        urgency = 0
        if any(k in low for k in ("critical", "sev1", "p0", "immediately")):
            urgency = 3
        elif any(k in low for k in ("urgent", "asap", "blocker")):
            urgency = 2
        elif any(k in low for k in ("soon", "priority", "important")):
            urgency = 1

        # Polarity heuristic
        polarity = 0
        if any(k in low for k in ("error", "failed", "crash", "broken", "bug")):
            polarity = -1
        elif any(k in low for k in ("fixed", "resolved", "success", "completed", "done")):
            polarity = 1

        # Action heuristic
        action = "INF"
        if text.strip().endswith("?") or low.startswith(("can you", "could you", "please")):
            action = "REQ"
        if any(k in low for k in ("review", "evaluate", "assess")):
            action = "EVAL"
        if any(k in low for k in ("i noticed", "detected", "observed")):
            action = "OBS"
        if any(k in low for k in ("i suggest", "i propose", "we could")):
            action = "PROP"

        # Domain heuristic
        domain = "GEN"
        if any(k in low for k in ("kubernetes", "deploy", "docker", "infra")):
            domain = "INFRA"
        elif any(k in low for k in ("auth", "oauth", "login", "jwt")):
            domain = "AUTH"
        elif any(k in low for k in ("review", "pull request", "pr ", "qa", "test")):
            domain = "QA"
        elif any(k in low for k in ("task", "ticket", "jira")):
            domain = "TASK"
        elif any(k in low for k in ("error", "exception", "stacktrace", "failure")):
            domain = "ERR"

        # Optional embedding refinement
        if vec is not None and self._proto_action and self._proto_domain:
            v = vec.astype(np.float32, copy=False)
            n = np.linalg.norm(v)
            if n > 0:
                v = v / n

            a_best, a_score = action, -1.0
            for a, pv in self._proto_action.items():
                s = float(np.dot(v, pv))
                if s > a_score:
                    a_best, a_score = a, s
            if a_score >= 0.40:
                action = a_best

            d_best, d_score = domain, -1.0
            for d, pv in self._proto_domain.items():
                s = float(np.dot(v, pv))
                if s > d_score:
                    d_best, d_score = d, s
            if d_score >= 0.35:
                domain = d_best

        return SemanticCoords(action=action, polarity=polarity, domain=domain, urgency=urgency)
