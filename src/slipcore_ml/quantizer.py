"""ML-based quantization (requires slipcore[ml]).

Two-stage embedding quantizer: Force prototypes + Object prototypes.
"""

from __future__ import annotations

import hashlib
import uuid
from typing import Optional

import numpy as np
from sentence_transformers import SentenceTransformer

from slipcore import (
    FallbackEmitted,
    Quantized,
    UCR,
    create_base_ucr,
    emit,
    format_fallback,
    format_slip,
)


class SemanticQuantizer:
    """Embedding-based quantizer for Think-Quantize-Transmit.

    Uses sentence-transformers to embed thoughts and match against
    UCR anchor centroids via cosine similarity.
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        ucr: Optional[UCR] = None,
        fallback_threshold: float = 0.40,
    ) -> None:
        self.model = SentenceTransformer(model_name)
        self.ucr = ucr or create_base_ucr()
        self.fallback_threshold = fallback_threshold
        self._centroids: dict[tuple[str, str], np.ndarray] = {}
        self._build_centroids()

    def _build_centroids(self) -> None:
        """Build embedding centroids for each anchor."""
        for anchor in self.ucr:
            text = anchor.canonical
            emb = self.model.encode(text, normalize_embeddings=True)
            self._centroids[(anchor.force, anchor.obj)] = emb

    def quantize(self, text: str, src: str, dst: str) -> str:
        """Quantize natural language to SLIP wire format.

        Args:
            text: Natural language intent
            src: Source agent ID
            dst: Destination agent ID

        Returns:
            Wire-format SLIP message string
        """
        emb = self.model.encode(text, normalize_embeddings=True)

        best_pair: Optional[tuple[str, str]] = None
        best_score = -1.0

        for (force, obj), centroid in self._centroids.items():
            score = float(np.dot(emb, centroid))
            if score > best_score:
                best_score = score
                best_pair = (force, obj)

        input_hash = hashlib.sha256(text.encode()).hexdigest()[:12]

        if best_score < self.fallback_threshold or best_pair is None:
            ref = f"ref{uuid.uuid4().hex[:8]}"
            emit(FallbackEmitted(
                ref=ref,
                reason="low_confidence",
                confidence=best_score,
                raw_text_hash=input_hash,
            ))
            return format_fallback(src, dst, ref)

        force, obj = best_pair
        emit(Quantized(
            input_hash=input_hash,
            force=force,
            obj=obj,
            confidence=best_score,
            mode="embedding",
        ))

        return format_slip(src, dst, force, obj)
