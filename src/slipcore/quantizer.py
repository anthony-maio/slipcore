"""
Semantic Quantizer - The Think-Quantize-Transmit Engine

Maps agent thoughts (natural language) to UCR anchors.
Supports three modes:
1. Keyword-based (fast, no dependencies)
2. Embedding-based with centroids (accurate, requires sentence-transformers)
3. Hybrid with CoordsInferer (prototype similarity + heuristics)

Also handles:
- Fallback detection (when confidence is too low)
- Usage tracking (for UCR evolution)
- Coordinate inference for new anchors
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Callable, Dict, List, Tuple
from collections import Counter
import re

from .ucr import UCR, UCRAnchor, get_default_ucr, CORE_RANGE_END


# ============ Optional Dependencies ============

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    np = None  # type: ignore
    HAS_NUMPY = False


# ============ Semantic Coordinates ============

@dataclass(frozen=True)
class SemanticCoords:
    """
    4D logical position of an intent in the semantic manifold.

    This is a human-readable representation used during coordinate inference.
    Maps to UCR's numeric coords via ACTION_MAP, DOMAIN_MAP, etc.
    """
    action: str    # REQ, INF, EVAL, CMD, OBS, PROP, META
    polarity: int  # -1 (negative), 0 (neutral), 1 (positive)
    domain: str    # TASK, QA, INFRA, AUTH, ERR, DOC, META, GEN
    urgency: int   # 0 (routine) to 3 (critical)


# Map v3-style string coords to v2 numeric coords
ACTION_MAP = {"OBS": 0, "INF": 1, "ASK": 2, "REQ": 3, "PROP": 4, "COMMIT": 5, "EVAL": 6, "META": 7, "CMD": 3}
DOMAIN_MAP = {"TASK": 0, "PLAN": 1, "OBS": 2, "EVAL": 3, "CTRL": 4, "RES": 5, "ERR": 6, "GEN": 7,
              "QA": 3, "INFRA": 5, "AUTH": 4, "DOC": 1, "META": 4}


def semantic_coords_to_tuple(sc: SemanticCoords) -> tuple[int, ...]:
    """Convert SemanticCoords to UCR numeric tuple."""
    action = ACTION_MAP.get(sc.action.upper(), 3)  # default REQ
    domain = DOMAIN_MAP.get(sc.domain.upper(), 7)  # default GEN
    # Map polarity: -1->1, 0->4, 1->6
    polarity = {-1: 1, 0: 4, 1: 6}.get(sc.polarity, 4)
    # Map urgency 0-3 to 0-7: 0->1, 1->3, 2->5, 3->7
    urgency = min(7, max(0, sc.urgency * 2 + 1))
    return (action, polarity, domain, urgency)


# ============ Coordinate Inference (from v3) ============

class CoordsInferer:
    """
    Assigns (Action, Polarity, Domain, Urgency) to text.

    Hybrid approach:
    - Heuristics for urgency and polarity (reliable, fast)
    - Optional prototype embedding similarity for action/domain refinement

    This is ported from v3's sophisticated coordinate inference system.
    """

    def __init__(self, embed_batch: Optional[Callable] = None):
        self._embed_batch = embed_batch
        self._proto_action: Dict[str, "np.ndarray"] = {}
        self._proto_domain: Dict[str, "np.ndarray"] = {}

        # Prototype phrases (short sentences > single tokens for embeddings)
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
        if not self._embed_batch or not HAS_NUMPY:
            return

        def _norm(vec):
            n = np.linalg.norm(vec)
            return vec / (n + 1e-12) if n > 0 else vec

        # Actions
        action_labels = list(self._action_phrases.keys())
        action_texts = [" ".join(self._action_phrases[a]) for a in action_labels]
        action_vecs = self._embed_batch(action_texts)
        for a, v in zip(action_labels, action_vecs):
            self._proto_action[a] = _norm(np.asarray(v, dtype=np.float32))

        # Domains
        domain_labels = list(self._domain_phrases.keys())
        domain_texts = [" ".join(self._domain_phrases[d]) for d in domain_labels]
        domain_vecs = self._embed_batch(domain_texts)
        for d, v in zip(domain_labels, domain_vecs):
            self._proto_domain[d] = _norm(np.asarray(v, dtype=np.float32))

    def infer(self, text: str, vec: Optional["np.ndarray"] = None) -> SemanticCoords:
        """
        Infer semantic coordinates from text.

        Args:
            text: The input text to analyze
            vec: Optional pre-computed embedding vector for refinement

        Returns:
            SemanticCoords with inferred action, polarity, domain, urgency
        """
        t = text.strip()
        low = t.lower()

        # --- urgency (heuristic) ---
        urgency = 0
        if any(k in low for k in ("critical", "sev1", "sev-1", "p0", "immediately", "right now")):
            urgency = 3
        elif any(k in low for k in ("urgent", "asap", "high priority", "blocker")):
            urgency = 2
        elif any(k in low for k in ("soon", "priority", "important")):
            urgency = 1

        # --- polarity (heuristic) ---
        polarity = 0
        if any(k in low for k in ("error", "failed", "failure", "crash", "broken", "outage", "bug", "can't", "cannot")):
            polarity = -1
        elif any(k in low for k in ("fixed", "resolved", "success", "completed", "done", "working now", "all good")):
            polarity = 1

        # --- action (heuristic) ---
        action = "INF"
        if t.endswith("?") or low.startswith(("can you", "could you", "would you", "please")):
            action = "REQ"
        if any(k in low for k in ("review", "critique", "evaluate", "assess")):
            action = "EVAL"
        if any(k in low for k in ("do this", "run ", "execute", "deploy", "scale ", "restart")) and urgency >= 2:
            action = "CMD"
        if any(k in low for k in ("i noticed", "i see", "detected", "observed")):
            action = "OBS"
        if any(k in low for k in ("i suggest", "i propose", "we could", "how about")):
            action = "PROP"

        # --- domain (heuristic) ---
        domain = "GEN"
        if any(k in low for k in ("kubernetes", "k8s", "cluster", "deploy", "terraform", "docker", "infra", "server", "latency")):
            domain = "INFRA"
        elif any(k in low for k in ("auth", "oauth", "login", "jwt", "sso", "permission")):
            domain = "AUTH"
        elif any(k in low for k in ("review", "pull request", "pr ", "qa", "test")):
            domain = "QA"
        elif any(k in low for k in ("task", "ticket", "jira", "backlog")):
            domain = "TASK"
        elif any(k in low for k in ("error", "exception", "stacktrace", "failed", "failure", "outage")):
            domain = "ERR"
        elif any(k in low for k in ("doc", "documentation", "readme", "spec", "paper")):
            domain = "DOC"
        elif any(k in low for k in ("protocol", "manifold", "coordination", "orchestrator")):
            domain = "META"

        # Optional refinement via prototype similarity
        if vec is not None and HAS_NUMPY and self._proto_action and self._proto_domain:
            def _norm(v):
                n = np.linalg.norm(v)
                return v / (n + 1e-12) if n > 0 else v

            v = _norm(vec.astype(np.float32, copy=False))

            # Action refine
            a_best, a_score = action, -1.0
            for a, pv in self._proto_action.items():
                s = float(np.dot(v, pv))
                if s > a_score:
                    a_best, a_score = a, s
            if a_score >= 0.40:
                action = a_best

            # Domain refine
            d_best, d_score = domain, -1.0
            for d, pv in self._proto_domain.items():
                s = float(np.dot(v, pv))
                if s > d_score:
                    d_best, d_score = d, s
            if d_score >= 0.35:
                domain = d_best

        # If action is REQ and urgency not set, default to 1
        if action == "REQ" and urgency == 0:
            urgency = 1

        return SemanticCoords(action=action, polarity=polarity, domain=domain, urgency=urgency)


# Global coords inferer instance
_coords_inferer: Optional[CoordsInferer] = None


def get_coords_inferer() -> CoordsInferer:
    """Get or create the default CoordsInferer."""
    global _coords_inferer
    if _coords_inferer is None:
        _coords_inferer = CoordsInferer()
    return _coords_inferer


def infer_coords(text: str, vec: Optional["np.ndarray"] = None) -> tuple[int, ...]:
    """
    Infer UCR-compatible coordinates from text.

    Returns a tuple of 4 integers suitable for UCRAnchor.coords.
    """
    inferer = get_coords_inferer()
    sc = inferer.infer(text, vec)
    return semantic_coords_to_tuple(sc)


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


# ============ Embedding-Based Quantizer (Enhanced) ============

class EmbeddingQuantizer:
    """
    Embedding-based quantizer using sentence-transformers.

    Enhanced with v3's centroid matrix approach:
    - Pre-computes normalized centroid matrix for fast similarity search
    - Supports anchor centroids (when available) or on-the-fly embedding
    - Includes CoordsInferer for prototype refinement

    Usage:
        quantizer = EmbeddingQuantizer()
        result = quantizer.quantize("I need someone to review this code")
    """

    def __init__(
        self,
        ucr: Optional[UCR] = None,
        model_name: str = "all-MiniLM-L6-v2",
        fallback_threshold: float = 0.55,
    ):
        if not HAS_NUMPY:
            raise ImportError("numpy is required for EmbeddingQuantizer")

        self.ucr = ucr or get_default_ucr()
        self.fallback_threshold = fallback_threshold
        self._usage_stats: Counter = Counter()
        self._fallback_buffer: List[str] = []  # Track low-confidence messages

        # Lazy load sentence-transformers
        self._model = None
        self._model_name = model_name

        # Centroid matrix (normalized) for fast similarity
        self._centroids_matrix: Optional["np.ndarray"] = None
        self._anchor_indices: List[int] = []  # Maps matrix row to anchor index
        self._embed_dim: Optional[int] = None

        # Coords inferer with prototype refinement
        self._coords_inferer: Optional[CoordsInferer] = None

    def _ensure_model(self):
        """Lazy load the embedding model and build centroid matrix."""
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
        self._rebuild_index()

        # Initialize coords inferer with embeddings
        self._coords_inferer = CoordsInferer(embed_batch=self._embed_batch)
        self._coords_inferer.prime()

    def _embed_batch(self, texts: List[str]) -> "np.ndarray":
        """Embed a batch of texts and return normalized vectors."""
        if not self._model:
            self._ensure_model()
        vecs = self._model.encode(texts, convert_to_numpy=True)
        vecs = np.asarray(vecs, dtype=np.float32)
        self._embed_dim = vecs.shape[1]
        # Normalize rows
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        return vecs / (norms + 1e-12)

    def _embed_one(self, text: str) -> "np.ndarray":
        """Embed a single text and return normalized vector."""
        return self._embed_batch([text])[0]

    def _rebuild_index(self):
        """Build/rebuild the centroid matrix from UCR anchors."""
        if not self.ucr.anchors:
            self._anchor_indices = []
            self._centroids_matrix = None
            return

        self._anchor_indices = sorted(self.ucr.anchors.keys())
        anchors = [self.ucr.anchors[idx] for idx in self._anchor_indices]

        # Check if anchors have pre-computed centroids
        has_centroids = all(a.centroid is not None for a in anchors)

        if has_centroids:
            # Use pre-computed centroids
            mat = np.asarray([a.centroid for a in anchors], dtype=np.float32)
        else:
            # Compute centroids from canonical texts
            canonical_texts = [a.canonical for a in anchors]
            mat = self._embed_batch(canonical_texts)
            # Optionally store centroids back to anchors
            for anchor, vec in zip(anchors, mat):
                anchor.centroid = vec.tolist()

        # Normalize rows
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        self._centroids_matrix = mat / (norms + 1e-12)

    def quantize(self, thought: str) -> QuantizeResult:
        """
        Map a natural language thought to the best UCR anchor using embeddings.

        Uses normalized cosine similarity against centroid matrix for fast lookup.

        Args:
            thought: The agent's thought/intent in natural language

        Returns:
            QuantizeResult with the best anchor and confidence score
        """
        self._ensure_model()

        if self._centroids_matrix is None or len(self._anchor_indices) == 0:
            fallback = self.ucr.get_by_mnemonic("Fallback")
            self._fallback_buffer.append(thought)
            return QuantizeResult(
                anchor=fallback,
                confidence=0.0,
                method="fallback",
                alternatives=[],
            )

        # Embed the thought (normalized)
        thought_vec = self._embed_one(thought)

        # Compute cosine similarities (dot product of normalized vectors)
        similarities = np.dot(self._centroids_matrix, thought_vec)

        # Get top matches
        top_locs = np.argsort(similarities)[::-1][:5]
        scores = []
        for loc in top_locs:
            anchor_idx = self._anchor_indices[loc]
            anchor = self.ucr.anchors[anchor_idx]
            scores.append((anchor, float(similarities[loc])))

        best_anchor, best_score = scores[0]

        if best_score < self.fallback_threshold:
            fallback = self.ucr.get_by_mnemonic("Fallback")
            self._usage_stats["_fallback"] += 1
            self._fallback_buffer.append(thought)
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

    def get_fallback_buffer(self) -> List[str]:
        """Get the buffer of low-confidence messages for extension learning."""
        return self._fallback_buffer.copy()

    def clear_fallback_buffer(self) -> None:
        """Clear the fallback buffer after extension learning."""
        self._fallback_buffer.clear()

    def infer_coords_for_text(self, text: str) -> tuple[int, ...]:
        """
        Infer semantic coordinates for a text using the enhanced CoordsInferer.

        Returns UCR-compatible coordinate tuple.
        """
        self._ensure_model()
        vec = self._embed_one(text)
        if self._coords_inferer:
            sc = self._coords_inferer.infer(text, vec)
            return semantic_coords_to_tuple(sc)
        return infer_coords(text, vec)

    def compute_centroid(self, texts: List[str]) -> List[float]:
        """
        Compute the centroid embedding for a cluster of texts.

        Useful for creating new extension anchors.
        """
        self._ensure_model()
        embeds = self._embed_batch(texts)
        centroid = np.mean(embeds, axis=0)
        # Normalize
        norm = np.linalg.norm(centroid)
        if norm > 0:
            centroid = centroid / norm
        return centroid.tolist()

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
