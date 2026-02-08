"""Tests for the Universal Concept Reference (UCR) module."""

from __future__ import annotations

from pathlib import Path

import pytest

from slipcore.errors import UCRError
from slipcore.ucr import (
    UCR,
    UCRAnchor,
    UCRAuthority,
    AnchorState,
    CORE_RANGE_END,
    Coords,
    create_base_ucr,
)


# ---------------------------------------------------------------------------
# UCRAnchor
# ---------------------------------------------------------------------------


class TestUCRAnchor:
    """Tests for UCRAnchor dataclass construction and methods."""

    def test_construction_defaults(self) -> None:
        """Anchor constructed with positional args gets correct defaults."""
        anchor = UCRAnchor(
            index=1,
            force="Request",
            obj="Plan",
            canonical="Request plan creation",
            coords=(3, 4, 1, 4),
        )
        assert anchor.index == 1
        assert anchor.force == "Request"
        assert anchor.obj == "Plan"
        assert anchor.canonical == "Request plan creation"
        assert anchor.coords == (3, 4, 1, 4)
        assert anchor.is_core is True
        assert anchor.state is AnchorState.ACTIVE
        assert anchor.created_by == "core"
        assert anchor.replaced_by is None

    def test_construction_overrides(self) -> None:
        """Keyword args override default values."""
        anchor = UCRAnchor(
            index=0x8000,
            force="Custom",
            obj="Widget",
            canonical="Custom widget",
            coords=(0, 0, 0, 0),
            is_core=False,
            state=AnchorState.DRAFT,
            created_by="agent-x",
            replaced_by=0x8001,
        )
        assert anchor.is_core is False
        assert anchor.state is AnchorState.DRAFT
        assert anchor.created_by == "agent-x"
        assert anchor.replaced_by == 0x8001

    def test_mnemonic_property(self) -> None:
        """Mnemonic is force + obj concatenation."""
        anchor = UCRAnchor(
            index=0x0031,
            force="Request",
            obj="Plan",
            canonical="Request plan creation",
            coords=(3, 4, 1, 4),
        )
        assert anchor.mnemonic == "RequestPlan"

    def test_mnemonic_single_word(self) -> None:
        """Mnemonic works with single-character force/obj."""
        anchor = UCRAnchor(
            index=1, force="A", obj="B", canonical="test", coords=(0, 0, 0, 0)
        )
        assert anchor.mnemonic == "AB"

    def test_to_dict(self) -> None:
        """to_dict produces expected keys and values."""
        anchor = UCRAnchor(
            index=0x0031,
            force="Request",
            obj="Plan",
            canonical="Request plan creation",
            coords=(3, 4, 1, 4),
        )
        d = anchor.to_dict()
        assert d["index"] == 0x0031
        assert d["force"] == "Request"
        assert d["obj"] == "Plan"
        assert d["canonical"] == "Request plan creation"
        assert d["coords"] == [3, 4, 1, 4]  # tuple -> list
        assert d["is_core"] is True
        assert d["state"] == "active"
        assert d["created_by"] == "core"

    def test_from_dict(self) -> None:
        """from_dict reconstructs anchor from dict."""
        d = {
            "index": 0x0031,
            "force": "Request",
            "obj": "Plan",
            "canonical": "Request plan creation",
            "coords": [3, 4, 1, 4],
            "is_core": True,
            "state": "active",
            "created_by": "core",
        }
        anchor = UCRAnchor.from_dict(d)
        assert anchor.index == 0x0031
        assert anchor.force == "Request"
        assert anchor.coords == (3, 4, 1, 4)
        assert anchor.state is AnchorState.ACTIVE

    def test_to_dict_from_dict_roundtrip(self) -> None:
        """Roundtrip through to_dict/from_dict preserves all fields."""
        original = UCRAnchor(
            index=0x8001,
            force="Custom",
            obj="Action",
            canonical="A custom action",
            coords=(2, 5, 3, 7),
            is_core=False,
            state=AnchorState.PROPOSED,
            created_by="agent-z",
        )
        restored = UCRAnchor.from_dict(original.to_dict())
        assert restored.index == original.index
        assert restored.force == original.force
        assert restored.obj == original.obj
        assert restored.canonical == original.canonical
        assert restored.coords == original.coords
        assert restored.is_core == original.is_core
        assert restored.state == original.state
        assert restored.created_by == original.created_by

    def test_from_dict_defaults(self) -> None:
        """from_dict applies defaults for missing optional keys."""
        d = {
            "index": 1,
            "force": "F",
            "obj": "O",
            "canonical": "c",
            "coords": [0, 0, 0, 0],
        }
        anchor = UCRAnchor.from_dict(d)
        assert anchor.is_core is True
        assert anchor.state is AnchorState.ACTIVE
        assert anchor.created_by == "core"


class TestAnchorState:
    """Tests for the AnchorState enum."""

    def test_all_states_exist(self) -> None:
        """All expected lifecycle states are defined."""
        expected = {"DRAFT", "PROPOSED", "APPROVED", "ACTIVE", "DEPRECATED"}
        assert {s.name for s in AnchorState} == expected

    def test_values(self) -> None:
        """Enum values are lowercase strings."""
        assert AnchorState.DRAFT.value == "draft"
        assert AnchorState.ACTIVE.value == "active"
        assert AnchorState.DEPRECATED.value == "deprecated"

    def test_from_string(self) -> None:
        """AnchorState can be constructed from its string value."""
        assert AnchorState("proposed") is AnchorState.PROPOSED


# ---------------------------------------------------------------------------
# UCRAuthority
# ---------------------------------------------------------------------------


class TestUCRAuthority:
    """Tests for UCRAuthority dataclass."""

    def test_construction(self, authority: UCRAuthority) -> None:
        """Authority fixture has expected values."""
        assert authority.authority_id == "test"
        assert authority.ucr_version == "3.0.0"
        assert authority.ucr_hash == ""

    def test_to_dict(self, authority: UCRAuthority) -> None:
        """to_dict includes all fields."""
        d = authority.to_dict()
        assert d == {
            "authority_id": "test",
            "ucr_version": "3.0.0",
            "ucr_hash": "",
        }

    def test_to_dict_with_hash(self) -> None:
        """to_dict reflects a non-empty hash when set."""
        auth = UCRAuthority(
            authority_id="x", ucr_version="1.0", ucr_hash="abc123"
        )
        assert auth.to_dict()["ucr_hash"] == "abc123"


# ---------------------------------------------------------------------------
# UCR registry
# ---------------------------------------------------------------------------


class TestUCR:
    """Tests for the UCR registry class."""

    # -- add_anchor ----------------------------------------------------------

    def test_add_anchor_success(self, authority: UCRAuthority) -> None:
        """Adding a new anchor succeeds and is retrievable."""
        ucr = UCR(authority=authority)
        anchor = UCRAnchor(
            index=1, force="F", obj="O", canonical="c", coords=(0, 0, 0, 0)
        )
        ucr.add_anchor(anchor)
        assert len(ucr) == 1
        assert ucr.get_by_index(1) is anchor

    def test_add_anchor_duplicate_index(self, authority: UCRAuthority) -> None:
        """Adding anchor with duplicate index raises UCRError."""
        ucr = UCR(authority=authority)
        a1 = UCRAnchor(
            index=1, force="F1", obj="O1", canonical="c1", coords=(0, 0, 0, 0)
        )
        a2 = UCRAnchor(
            index=1, force="F2", obj="O2", canonical="c2", coords=(1, 1, 1, 1)
        )
        ucr.add_anchor(a1)
        with pytest.raises(UCRError, match="exists"):
            ucr.add_anchor(a2)

    def test_add_anchor_duplicate_pair(self, authority: UCRAuthority) -> None:
        """Adding anchor with duplicate force/obj pair raises UCRError."""
        ucr = UCR(authority=authority)
        a1 = UCRAnchor(
            index=1, force="F", obj="O", canonical="c1", coords=(0, 0, 0, 0)
        )
        a2 = UCRAnchor(
            index=2, force="F", obj="O", canonical="c2", coords=(1, 1, 1, 1)
        )
        ucr.add_anchor(a1)
        with pytest.raises(UCRError, match="exists"):
            ucr.add_anchor(a2)

    # -- get_by_index --------------------------------------------------------

    def test_get_by_index_found(self, base_ucr: UCR) -> None:
        """get_by_index returns anchor for known index."""
        anchor = base_ucr.get_by_index(0x0031)
        assert anchor is not None
        assert anchor.force == "Request"
        assert anchor.obj == "Plan"

    def test_get_by_index_missing(self, base_ucr: UCR) -> None:
        """get_by_index returns None for unknown index."""
        assert base_ucr.get_by_index(0xFFFF) is None

    # -- get_by_force_obj ----------------------------------------------------

    def test_get_by_force_obj_found(self, base_ucr: UCR) -> None:
        """get_by_force_obj returns anchor for known pair."""
        anchor = base_ucr.get_by_force_obj("Request", "Plan")
        assert anchor is not None
        assert anchor.index == 0x0031

    def test_get_by_force_obj_missing(self, base_ucr: UCR) -> None:
        """get_by_force_obj returns None for unknown pair."""
        assert base_ucr.get_by_force_obj("NonExistent", "Pair") is None

    # -- find_nearest --------------------------------------------------------

    def test_find_nearest_exact_match(self, base_ucr: UCR) -> None:
        """find_nearest returns exact match when coords match an anchor."""
        # Request Plan has coords (3, 4, 1, 4)
        result = base_ucr.find_nearest((3, 4, 1, 4))
        assert result.force == "Request"
        assert result.obj == "Plan"

    def test_find_nearest_close_coords(self, base_ucr: UCR) -> None:
        """find_nearest returns closest anchor by Manhattan distance."""
        # (3, 4, 1, 5) is 1 away from Request Plan (3, 4, 1, 4)
        result = base_ucr.find_nearest((3, 4, 1, 5))
        # Should be close to Request Plan
        assert result is not None
        assert isinstance(result, UCRAnchor)

    def test_find_nearest_empty_raises(self, authority: UCRAuthority) -> None:
        """find_nearest raises UCRError on an empty registry."""
        ucr = UCR(authority=authority)
        with pytest.raises(UCRError, match="No anchors"):
            ucr.find_nearest((0, 0, 0, 0))

    def test_find_nearest_skips_deprecated(self, authority: UCRAuthority) -> None:
        """find_nearest ignores deprecated anchors."""
        ucr = UCR(authority=authority)
        deprecated = UCRAnchor(
            index=1,
            force="Old",
            obj="Thing",
            canonical="old",
            coords=(0, 0, 0, 0),
            state=AnchorState.DEPRECATED,
        )
        active = UCRAnchor(
            index=2,
            force="New",
            obj="Thing",
            canonical="new",
            coords=(7, 7, 7, 7),
        )
        ucr.add_anchor(deprecated)
        ucr.add_anchor(active)
        # Even though (0,0,0,0) is exact match for deprecated, it should
        # return the active anchor instead.
        result = ucr.find_nearest((0, 0, 0, 0))
        assert result.force == "New"

    def test_find_nearest_manhattan_distance(
        self, authority: UCRAuthority
    ) -> None:
        """find_nearest uses Manhattan (L1) distance, not Euclidean."""
        ucr = UCR(authority=authority)
        a = UCRAnchor(
            index=1, force="A", obj="A", canonical="a", coords=(0, 0, 0, 0)
        )
        b = UCRAnchor(
            index=2, force="B", obj="B", canonical="b", coords=(1, 1, 1, 1)
        )
        ucr.add_anchor(a)
        ucr.add_anchor(b)
        # Manhattan distance to (0,0,0,1): A=1, B=3 -> A wins
        result = ucr.find_nearest((0, 0, 0, 1))
        assert result.force == "A"

    # -- next_extension_index ------------------------------------------------

    def test_next_extension_index_no_extensions(
        self, base_ucr: UCR
    ) -> None:
        """With only core anchors, next extension index is CORE_RANGE_END."""
        assert base_ucr.next_extension_index() == CORE_RANGE_END

    def test_next_extension_index_with_extensions(
        self, base_ucr: UCR
    ) -> None:
        """After adding extension, next index increments."""
        base_ucr.propose_extension(
            force="Ext",
            obj="First",
            canonical="first extension",
            coords=(0, 0, 0, 0),
        )
        assert base_ucr.next_extension_index() == CORE_RANGE_END + 1

    # -- propose_extension ---------------------------------------------------

    def test_propose_extension(self, base_ucr: UCR) -> None:
        """propose_extension creates a DRAFT, non-core anchor."""
        before_len = len(base_ucr)
        ext = base_ucr.propose_extension(
            force="Deploy",
            obj="Container",
            canonical="Deploy a container",
            coords=(5, 6, 5, 4),
            created_by="devops-agent",
        )
        assert ext.is_core is False
        assert ext.state is AnchorState.DRAFT
        assert ext.created_by == "devops-agent"
        assert ext.index >= CORE_RANGE_END
        assert len(base_ucr) == before_len + 1
        # Retrievable by force/obj
        assert base_ucr.get_by_force_obj("Deploy", "Container") is ext

    # -- content_hash --------------------------------------------------------

    def test_content_hash_deterministic(self, base_ucr: UCR) -> None:
        """Same content produces same hash every time."""
        h1 = base_ucr.content_hash()
        h2 = base_ucr.content_hash()
        assert h1 == h2

    def test_content_hash_same_content(self) -> None:
        """Two registries with identical anchors produce the same hash."""
        ucr1 = create_base_ucr()
        ucr2 = create_base_ucr()
        assert ucr1.content_hash() == ucr2.content_hash()

    def test_content_hash_changes_on_modification(
        self, base_ucr: UCR
    ) -> None:
        """Adding an anchor changes the content hash."""
        h_before = base_ucr.content_hash()
        base_ucr.propose_extension(
            force="New", obj="Anchor", canonical="new", coords=(0, 0, 0, 0)
        )
        h_after = base_ucr.content_hash()
        assert h_before != h_after

    def test_content_hash_is_string(self, base_ucr: UCR) -> None:
        """content_hash returns a non-empty string."""
        h = base_ucr.content_hash()
        assert isinstance(h, str)
        assert len(h) > 0

    # -- save / load ---------------------------------------------------------

    def test_save_load_roundtrip(self, base_ucr: UCR, tmp_path: Path) -> None:
        """save then load produces equivalent registry."""
        filepath = tmp_path / "ucr.json"
        base_ucr.save(filepath)
        loaded = UCR.load(filepath)

        assert len(loaded) == len(base_ucr)
        assert loaded.authority.authority_id == base_ucr.authority.authority_id
        assert loaded.authority.ucr_version == base_ucr.authority.ucr_version
        assert loaded.authority.ucr_hash == base_ucr.content_hash()

        # Spot-check an anchor
        orig = base_ucr.get_by_index(0x0031)
        rest = loaded.get_by_index(0x0031)
        assert orig is not None
        assert rest is not None
        assert orig.force == rest.force
        assert orig.obj == rest.obj
        assert orig.coords == rest.coords
        assert orig.state == rest.state

    def test_save_load_roundtrip_with_extensions(
        self, base_ucr: UCR, tmp_path: Path
    ) -> None:
        """Extensions survive save/load roundtrip."""
        base_ucr.propose_extension(
            force="Custom",
            obj="Widget",
            canonical="custom widget",
            coords=(1, 2, 3, 4),
            created_by="test-agent",
        )
        filepath = tmp_path / "ucr_ext.json"
        base_ucr.save(filepath)
        loaded = UCR.load(filepath)

        ext = loaded.get_by_force_obj("Custom", "Widget")
        assert ext is not None
        assert ext.is_core is False
        assert ext.state is AnchorState.DRAFT
        assert ext.created_by == "test-agent"

    # -- __len__ and __iter__ ------------------------------------------------

    def test_len(self, base_ucr: UCR) -> None:
        """__len__ returns number of anchors."""
        assert len(base_ucr) == 45

    def test_len_empty(self, authority: UCRAuthority) -> None:
        """Empty registry has length 0."""
        ucr = UCR(authority=authority)
        assert len(ucr) == 0

    def test_iter(self, base_ucr: UCR) -> None:
        """__iter__ yields all anchors."""
        anchors = list(base_ucr)
        assert len(anchors) == 45
        assert all(isinstance(a, UCRAnchor) for a in anchors)

    def test_iter_empty(self, authority: UCRAuthority) -> None:
        """Iterating empty registry yields nothing."""
        ucr = UCR(authority=authority)
        assert list(ucr) == []


# ---------------------------------------------------------------------------
# create_base_ucr
# ---------------------------------------------------------------------------


class TestCreateBaseUCR:
    """Tests for the create_base_ucr factory function."""

    def test_returns_ucr(self) -> None:
        """Factory returns a UCR instance."""
        ucr = create_base_ucr()
        assert isinstance(ucr, UCR)

    def test_has_45_anchors(self, base_ucr: UCR) -> None:
        """Base UCR contains exactly 45 core anchors."""
        assert len(base_ucr) == 45

    def test_all_anchors_are_core(self, base_ucr: UCR) -> None:
        """Every anchor in the base UCR is marked core."""
        for anchor in base_ucr:
            assert anchor.is_core is True, (
                f"Anchor {anchor.mnemonic} is not core"
            )

    def test_has_known_anchor_request_plan(self, base_ucr: UCR) -> None:
        """Base UCR contains the well-known (Request, Plan) anchor."""
        anchor = base_ucr.get_by_force_obj("Request", "Plan")
        assert anchor is not None
        assert anchor.canonical == "Request plan creation"
        assert anchor.index == 0x0031

    def test_has_known_anchor_meta_ack(self, base_ucr: UCR) -> None:
        """Base UCR contains the well-known (Meta, Ack) anchor."""
        anchor = base_ucr.get_by_force_obj("Meta", "Ack")
        assert anchor is not None
        assert anchor.index == 0x0070

    def test_content_hash_is_non_empty_string(self, base_ucr: UCR) -> None:
        """Base UCR has a pre-computed non-empty content hash."""
        assert isinstance(base_ucr.authority.ucr_hash, str)
        assert len(base_ucr.authority.ucr_hash) > 0

    def test_authority_defaults(self, base_ucr: UCR) -> None:
        """Base UCR has expected default authority values."""
        assert base_ucr.authority.authority_id == "slipcore-default"
        assert base_ucr.authority.ucr_version == "3.0.0"

    def test_custom_authority_id(self) -> None:
        """Factory accepts a custom authority_id."""
        ucr = create_base_ucr(authority_id="my-org")
        assert ucr.authority.authority_id == "my-org"
        # Still has all 45 anchors
        assert len(ucr) == 45
