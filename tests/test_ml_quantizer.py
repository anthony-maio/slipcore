"""Tests for ML quantizer (skipped when ML deps not installed)."""

from __future__ import annotations

import pytest

try:
    import numpy as np
    from sentence_transformers import SentenceTransformer
    HAS_ML = True
except ImportError:
    HAS_ML = False

pytestmark = pytest.mark.skipif(not HAS_ML, reason="ML dependencies not installed")


class TestSemanticQuantizer:
    def test_import(self):
        from slipcore_ml.quantizer import SemanticQuantizer
        assert SemanticQuantizer is not None

    @pytest.mark.slow
    def test_quantize_basic(self):
        from slipcore_ml.quantizer import SemanticQuantizer

        q = SemanticQuantizer()
        wire = q.quantize("Please review the authentication code", "dev", "reviewer")
        assert wire.startswith("SLIP v3")
        assert "dev" in wire
        assert "reviewer" in wire

    @pytest.mark.slow
    def test_quantize_fallback(self):
        from slipcore_ml.quantizer import SemanticQuantizer

        q = SemanticQuantizer(fallback_threshold=0.99)  # Very high threshold
        wire = q.quantize("xyzzy foobar gibberish", "a", "b")
        assert "Fallback" in wire


class TestCoordsInferer:
    def test_import(self):
        from slipcore_ml.coords import CoordsInferer, SemanticCoords
        assert CoordsInferer is not None

    def test_infer_basic(self):
        from slipcore_ml.coords import CoordsInferer

        inferer = CoordsInferer()
        sc = inferer.infer("Please review this code")
        assert sc.action in ("REQ", "EVAL", "INF", "OBS", "PROP", "CMD")
        assert sc.polarity in (-1, 0, 1)
        assert sc.urgency in (0, 1, 2, 3)

    def test_infer_urgency(self):
        from slipcore_ml.coords import CoordsInferer

        inferer = CoordsInferer()
        sc = inferer.infer("This is critical and needs immediate attention")
        assert sc.urgency == 3

    def test_semantic_coords_to_tuple(self):
        from slipcore_ml.coords import SemanticCoords, semantic_coords_to_tuple

        sc = SemanticCoords(action="REQ", polarity=0, domain="TASK", urgency=1)
        coords = semantic_coords_to_tuple(sc)
        assert len(coords) == 4
        assert all(0 <= c <= 7 for c in coords)
