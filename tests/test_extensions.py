"""Comprehensive tests for slipcore.extensions -- FallbackEvent, FallbackTracker, ExtensionManager."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from slipcore.events import ExtensionProposed, clear_sinks, register_sink
from slipcore.extensions import ExtensionManager, FallbackEvent, FallbackTracker
from slipcore.ucr import CORE_RANGE_END, AnchorState, UCR, UCRAuthority, create_base_ucr


# ---------------------------------------------------------------------------
# Autouse fixture: prevent event sink leaks between tests
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _clean_event_sinks():
    """Clear all event sinks before and after every test."""
    clear_sinks()
    yield
    clear_sinks()


# ===========================================================================
# FallbackEvent
# ===========================================================================

class TestFallbackEvent:
    """Tests for the FallbackEvent dataclass."""

    def test_construction_all_fields(self):
        event = FallbackEvent(
            thought="deploy the new model",
            src="alice",
            dst="bob",
            timestamp=12345.6,
            nearest_force="Request",
            nearest_obj="Task",
            nearest_score=0.82,
        )
        assert event.thought == "deploy the new model"
        assert event.src == "alice"
        assert event.dst == "bob"
        assert event.timestamp == 12345.6
        assert event.nearest_force == "Request"
        assert event.nearest_obj == "Task"
        assert event.nearest_score == 0.82

    def test_default_values(self):
        event = FallbackEvent(thought="hello", src="a", dst="b")
        assert event.timestamp == 0.0
        assert event.nearest_force is None
        assert event.nearest_obj is None
        assert event.nearest_score == 0.0

    def test_required_fields_only(self):
        event = FallbackEvent(thought="x", src="s", dst="d")
        assert event.thought == "x"
        assert event.src == "s"
        assert event.dst == "d"

    def test_partial_optional_fields(self):
        event = FallbackEvent(
            thought="partial",
            src="a",
            dst="b",
            nearest_force="Observe",
        )
        assert event.nearest_force == "Observe"
        assert event.nearest_obj is None
        assert event.nearest_score == 0.0


# ===========================================================================
# FallbackTracker
# ===========================================================================

class TestFallbackTracker:
    """Tests for FallbackTracker pattern extraction and event management."""

    # -- Initialization ---

    def test_init_defaults(self):
        tracker = FallbackTracker()
        assert tracker.max_events == 10000
        assert tracker.events == []
        assert len(tracker._pattern_counts) == 0

    def test_init_custom_max(self):
        tracker = FallbackTracker(max_events=50)
        assert tracker.max_events == 50

    # -- record() ---

    def test_record_appends_event(self):
        tracker = FallbackTracker()
        event = FallbackEvent(thought="a b c", src="s", dst="d")
        tracker.record(event)
        assert len(tracker.events) == 1
        assert tracker.events[0] is event

    def test_record_multiple_events(self):
        tracker = FallbackTracker()
        for i in range(5):
            tracker.record(FallbackEvent(thought=f"word{i} extra", src="s", dst="d"))
        assert len(tracker.events) == 5

    # -- _extract_patterns() ---

    def test_extract_bigrams(self):
        tracker = FallbackTracker()
        event = FallbackEvent(thought="deploy the model", src="s", dst="d")
        tracker.record(event)
        patterns = dict(tracker.get_top_patterns(100))
        assert "deploy the" in patterns
        assert "the model" in patterns

    def test_extract_trigrams(self):
        tracker = FallbackTracker()
        event = FallbackEvent(thought="deploy the model", src="s", dst="d")
        tracker.record(event)
        patterns = dict(tracker.get_top_patterns(100))
        assert "deploy the model" in patterns

    def test_extract_patterns_lowercased(self):
        tracker = FallbackTracker()
        event = FallbackEvent(thought="Deploy The Model", src="s", dst="d")
        tracker.record(event)
        patterns = dict(tracker.get_top_patterns(100))
        assert "deploy the model" in patterns
        assert "Deploy The Model" not in patterns

    def test_extract_no_patterns_single_word(self):
        tracker = FallbackTracker()
        event = FallbackEvent(thought="deploy", src="s", dst="d")
        tracker.record(event)
        assert len(tracker._pattern_counts) == 0

    def test_extract_patterns_two_words_only_bigram(self):
        tracker = FallbackTracker()
        event = FallbackEvent(thought="deploy now", src="s", dst="d")
        tracker.record(event)
        patterns = dict(tracker.get_top_patterns(100))
        assert "deploy now" in patterns
        # No trigrams possible with only two words
        trigrams = [p for p in patterns if p.count(" ") == 2]
        assert trigrams == []

    def test_extract_patterns_accumulates_counts(self):
        tracker = FallbackTracker()
        for _ in range(3):
            tracker.record(FallbackEvent(thought="run the tests", src="s", dst="d"))
        patterns = dict(tracker.get_top_patterns(100))
        assert patterns["run the"] == 3
        assert patterns["the tests"] == 3
        assert patterns["run the tests"] == 3

    def test_extract_patterns_empty_thought(self):
        tracker = FallbackTracker()
        event = FallbackEvent(thought="", src="s", dst="d")
        tracker.record(event)
        assert len(tracker._pattern_counts) == 0

    # -- max_events truncation ---

    def test_max_events_truncation(self):
        tracker = FallbackTracker(max_events=5)
        for i in range(10):
            tracker.record(FallbackEvent(thought=f"event {i}", src="s", dst="d"))
        assert len(tracker.events) == 5
        # Keeps the last 5
        assert tracker.events[0].thought == "event 5"
        assert tracker.events[-1].thought == "event 9"

    def test_max_events_at_boundary(self):
        tracker = FallbackTracker(max_events=3)
        for i in range(3):
            tracker.record(FallbackEvent(thought=f"event {i}", src="s", dst="d"))
        assert len(tracker.events) == 3
        # No truncation yet -- exactly at limit
        assert tracker.events[0].thought == "event 0"

    def test_max_events_one_over_triggers_truncation(self):
        tracker = FallbackTracker(max_events=3)
        for i in range(4):
            tracker.record(FallbackEvent(thought=f"event {i}", src="s", dst="d"))
        assert len(tracker.events) == 3
        assert tracker.events[0].thought == "event 1"

    def test_patterns_survive_truncation(self):
        """Pattern counts are not reset when events are truncated."""
        tracker = FallbackTracker(max_events=2)
        for _ in range(5):
            tracker.record(FallbackEvent(thought="alpha beta gamma", src="s", dst="d"))
        assert len(tracker.events) == 2
        patterns = dict(tracker.get_top_patterns(100))
        # All 5 recordings should contribute to pattern counts
        assert patterns["alpha beta"] == 5

    # -- get_top_patterns() ---

    def test_get_top_patterns_respects_n(self):
        tracker = FallbackTracker()
        # Generate many distinct patterns
        for i in range(10):
            tracker.record(FallbackEvent(thought=f"word{i} next{i}", src="s", dst="d"))
        top3 = tracker.get_top_patterns(3)
        assert len(top3) == 3

    def test_get_top_patterns_empty_tracker(self):
        tracker = FallbackTracker()
        assert tracker.get_top_patterns(10) == []

    def test_get_top_patterns_returns_sorted_by_count(self):
        tracker = FallbackTracker()
        # "alpha beta" appears 5 times, "gamma delta" appears 2 times
        for _ in range(5):
            tracker.record(FallbackEvent(thought="alpha beta", src="s", dst="d"))
        for _ in range(2):
            tracker.record(FallbackEvent(thought="gamma delta", src="s", dst="d"))
        top = tracker.get_top_patterns(2)
        assert top[0][0] == "alpha beta"
        assert top[0][1] == 5
        assert top[1][0] == "gamma delta"
        assert top[1][1] == 2

    # -- get_stats() ---

    def test_get_stats_empty(self):
        tracker = FallbackTracker()
        stats = tracker.get_stats()
        assert stats["total_events"] == 0
        assert stats["unique_patterns"] == 0
        assert stats["top_patterns"] == []

    def test_get_stats_with_events(self):
        tracker = FallbackTracker()
        tracker.record(FallbackEvent(thought="run the tests now", src="s", dst="d"))
        stats = tracker.get_stats()
        assert stats["total_events"] == 1
        assert stats["unique_patterns"] > 0
        assert isinstance(stats["top_patterns"], list)

    def test_get_stats_structure(self):
        tracker = FallbackTracker()
        tracker.record(FallbackEvent(thought="a b c", src="s", dst="d"))
        stats = tracker.get_stats()
        assert set(stats.keys()) == {"total_events", "unique_patterns", "top_patterns"}

    # -- clear() ---

    def test_clear_empties_events(self):
        tracker = FallbackTracker()
        for i in range(5):
            tracker.record(FallbackEvent(thought=f"word{i} next{i}", src="s", dst="d"))
        tracker.clear()
        assert tracker.events == []

    def test_clear_empties_patterns(self):
        tracker = FallbackTracker()
        for i in range(5):
            tracker.record(FallbackEvent(thought=f"word{i} next{i}", src="s", dst="d"))
        tracker.clear()
        assert len(tracker._pattern_counts) == 0

    def test_clear_then_record_works(self):
        tracker = FallbackTracker()
        tracker.record(FallbackEvent(thought="before clear happened", src="s", dst="d"))
        tracker.clear()
        tracker.record(FallbackEvent(thought="after clear happened", src="s", dst="d"))
        assert len(tracker.events) == 1
        assert tracker.events[0].thought == "after clear happened"
        patterns = dict(tracker.get_top_patterns(100))
        assert "before clear" not in patterns
        assert "after clear" in patterns


# ===========================================================================
# ExtensionManager
# ===========================================================================

class TestExtensionManager:
    """Tests for ExtensionManager integration with UCR and event system."""

    # -- Initialization ---

    def test_init_default_ucr(self):
        mgr = ExtensionManager()
        assert mgr.ucr is not None
        assert isinstance(mgr.ucr, UCR)
        assert mgr.ucr.authority.ucr_version == "3.0.0"

    def test_init_creates_default_with_core_anchors(self):
        mgr = ExtensionManager()
        core = mgr.ucr.core_anchors()
        assert len(core) > 0

    def test_init_custom_ucr(self, base_ucr):
        mgr = ExtensionManager(ucr=base_ucr)
        assert mgr.ucr is base_ucr

    def test_init_fallback_tracker_created(self):
        mgr = ExtensionManager()
        assert isinstance(mgr.fallback_tracker, FallbackTracker)

    # -- add_extension() ---

    def test_add_extension_returns_anchor(self):
        mgr = ExtensionManager()
        anchor = mgr.add_extension(
            force="Deploy",
            obj="Container",
            canonical="Deploy container image",
            coords=(3, 5, 0, 4),
        )
        assert anchor.force == "Deploy"
        assert anchor.obj == "Container"
        assert anchor.canonical == "Deploy container image"
        assert anchor.coords == (3, 5, 0, 4)

    def test_add_extension_in_extension_range(self):
        mgr = ExtensionManager()
        anchor = mgr.add_extension(
            force="Deploy",
            obj="Container",
            canonical="Deploy container image",
            coords=(3, 5, 0, 4),
        )
        assert anchor.index >= CORE_RANGE_END  # >= 0x8000

    def test_add_extension_not_core(self):
        mgr = ExtensionManager()
        anchor = mgr.add_extension(
            force="Deploy",
            obj="Container",
            canonical="Deploy container image",
            coords=(3, 5, 0, 4),
        )
        assert anchor.is_core is False

    def test_add_extension_state_is_draft(self):
        mgr = ExtensionManager()
        anchor = mgr.add_extension(
            force="Deploy",
            obj="Container",
            canonical="Deploy container image",
            coords=(3, 5, 0, 4),
        )
        assert anchor.state == AnchorState.DRAFT

    def test_add_extension_default_created_by(self):
        mgr = ExtensionManager()
        anchor = mgr.add_extension(
            force="Deploy",
            obj="Container",
            canonical="Deploy container image",
            coords=(3, 5, 0, 4),
        )
        assert anchor.created_by == "swarm"

    def test_add_extension_custom_created_by(self):
        mgr = ExtensionManager()
        anchor = mgr.add_extension(
            force="Deploy",
            obj="Container",
            canonical="Deploy container image",
            coords=(3, 5, 0, 4),
            created_by="agent-007",
        )
        assert anchor.created_by == "agent-007"

    def test_add_extension_registers_in_ucr(self):
        mgr = ExtensionManager()
        anchor = mgr.add_extension(
            force="Deploy",
            obj="Container",
            canonical="Deploy container image",
            coords=(3, 5, 0, 4),
        )
        found = mgr.ucr.get_by_force_obj("Deploy", "Container")
        assert found is not None
        assert found.index == anchor.index

    def test_add_extension_lookup_by_index(self):
        mgr = ExtensionManager()
        anchor = mgr.add_extension(
            force="Deploy",
            obj="Container",
            canonical="Deploy container image",
            coords=(3, 5, 0, 4),
        )
        found = mgr.ucr.get_by_index(anchor.index)
        assert found is not None
        assert found.force == "Deploy"

    def test_add_multiple_extensions_increment_index(self):
        mgr = ExtensionManager()
        a1 = mgr.add_extension(
            force="Deploy", obj="Container",
            canonical="Deploy container", coords=(3, 5, 0, 4),
        )
        a2 = mgr.add_extension(
            force="Scale", obj="Service",
            canonical="Scale service instances", coords=(3, 5, 5, 4),
        )
        assert a2.index == a1.index + 1
        assert a1.index >= CORE_RANGE_END
        assert a2.index >= CORE_RANGE_END

    # -- add_extension emits ExtensionProposed event ---

    def test_add_extension_emits_event(self):
        captured = []
        register_sink(captured.append)
        mgr = ExtensionManager()
        mgr.add_extension(
            force="Deploy",
            obj="Container",
            canonical="Deploy container image",
            coords=(3, 5, 0, 4),
            created_by="tester",
        )
        assert len(captured) == 1
        event = captured[0]
        assert isinstance(event, ExtensionProposed)
        assert event.force == "Deploy"
        assert event.obj == "Container"
        assert event.created_by == "tester"
        assert event.event_type == "extension.proposed"

    def test_add_extension_emits_event_for_each_call(self):
        captured = []
        register_sink(captured.append)
        mgr = ExtensionManager()
        mgr.add_extension(
            force="Deploy", obj="Container",
            canonical="c1", coords=(3, 5, 0, 4),
        )
        mgr.add_extension(
            force="Scale", obj="Service",
            canonical="c2", coords=(3, 5, 5, 4),
        )
        assert len(captured) == 2
        assert captured[0].force == "Deploy"
        assert captured[1].force == "Scale"

    def test_add_extension_event_has_timestamp(self):
        captured = []
        register_sink(captured.append)
        mgr = ExtensionManager()
        mgr.add_extension(
            force="Deploy", obj="Container",
            canonical="c1", coords=(3, 5, 0, 4),
        )
        assert captured[0].timestamp > 0

    # -- record_fallback() ---

    def test_record_fallback_delegates_to_tracker(self):
        mgr = ExtensionManager()
        mgr.record_fallback(
            thought="please deploy the new container image",
            src="alice",
            dst="bob",
            nearest_force="Request",
            nearest_obj="Task",
            nearest_score=0.65,
        )
        assert len(mgr.fallback_tracker.events) == 1
        event = mgr.fallback_tracker.events[0]
        assert event.thought == "please deploy the new container image"
        assert event.src == "alice"
        assert event.dst == "bob"
        assert event.nearest_force == "Request"
        assert event.nearest_obj == "Task"
        assert event.nearest_score == 0.65

    def test_record_fallback_defaults(self):
        mgr = ExtensionManager()
        mgr.record_fallback(thought="unknown intent", src="a", dst="b")
        event = mgr.fallback_tracker.events[0]
        assert event.nearest_force is None
        assert event.nearest_obj is None
        assert event.nearest_score == 0.0

    def test_record_fallback_extracts_patterns(self):
        mgr = ExtensionManager()
        mgr.record_fallback(thought="deploy the container", src="a", dst="b")
        patterns = dict(mgr.fallback_tracker.get_top_patterns(100))
        assert "deploy the" in patterns
        assert "the container" in patterns
        assert "deploy the container" in patterns

    def test_record_multiple_fallbacks(self):
        mgr = ExtensionManager()
        for i in range(10):
            mgr.record_fallback(thought=f"intent {i} description", src="a", dst="b")
        assert len(mgr.fallback_tracker.events) == 10

    # -- suggest_extensions() ---

    def test_suggest_extensions_empty(self):
        mgr = ExtensionManager()
        suggestions = mgr.suggest_extensions()
        assert suggestions == []

    def test_suggest_extensions_below_threshold(self):
        mgr = ExtensionManager()
        for _ in range(4):
            mgr.record_fallback(thought="deploy the container", src="a", dst="b")
        # Default min_count=5, so 4 occurrences should not meet the threshold
        suggestions = mgr.suggest_extensions(min_count=5)
        assert suggestions == []

    def test_suggest_extensions_at_threshold(self):
        mgr = ExtensionManager()
        for _ in range(5):
            mgr.record_fallback(thought="deploy the container", src="a", dst="b")
        suggestions = mgr.suggest_extensions(min_count=5)
        assert len(suggestions) > 0
        assert all(s.startswith("Agent intent: ") for s in suggestions)

    def test_suggest_extensions_above_threshold(self):
        mgr = ExtensionManager()
        for _ in range(10):
            mgr.record_fallback(thought="scale the service up", src="a", dst="b")
        suggestions = mgr.suggest_extensions(min_count=5)
        assert len(suggestions) > 0

    def test_suggest_extensions_mixed_patterns(self):
        mgr = ExtensionManager()
        # Pattern A appears 7 times
        for _ in range(7):
            mgr.record_fallback(thought="restart the worker", src="a", dst="b")
        # Pattern B appears 3 times (below threshold)
        for _ in range(3):
            mgr.record_fallback(thought="flush the cache", src="a", dst="b")
        suggestions = mgr.suggest_extensions(min_count=5)
        suggestion_texts = " ".join(suggestions)
        assert "restart the" in suggestion_texts
        assert "flush the" not in suggestion_texts

    def test_suggest_extensions_custom_min_count(self):
        mgr = ExtensionManager()
        for _ in range(2):
            mgr.record_fallback(thought="drain the node", src="a", dst="b")
        # With min_count=2, should appear
        suggestions = mgr.suggest_extensions(min_count=2)
        assert len(suggestions) > 0

    def test_suggest_extensions_format(self):
        mgr = ExtensionManager()
        for _ in range(5):
            mgr.record_fallback(thought="deploy the container", src="a", dst="b")
        suggestions = mgr.suggest_extensions(min_count=5)
        for suggestion in suggestions:
            assert suggestion.startswith("Agent intent: ")
            # Strip prefix and check there is content after it
            pattern = suggestion[len("Agent intent: "):]
            assert len(pattern) > 0

    # -- export_extensions() ---

    def test_export_extensions_creates_file(self, tmp_path):
        mgr = ExtensionManager()
        out = tmp_path / "extensions.json"
        mgr.export_extensions(out)
        assert out.exists()

    def test_export_extensions_valid_json(self, tmp_path):
        mgr = ExtensionManager()
        out = tmp_path / "extensions.json"
        mgr.export_extensions(out)
        data = json.loads(out.read_text(encoding="utf-8"))
        assert isinstance(data, dict)

    def test_export_extensions_structure(self, tmp_path):
        mgr = ExtensionManager()
        out = tmp_path / "extensions.json"
        mgr.export_extensions(out)
        data = json.loads(out.read_text(encoding="utf-8"))
        assert "ucr_version" in data
        assert "extensions" in data
        assert "fallback_stats" in data
        assert data["ucr_version"] == "3.0.0"

    def test_export_extensions_empty_no_extensions(self, tmp_path):
        mgr = ExtensionManager()
        out = tmp_path / "extensions.json"
        mgr.export_extensions(out)
        data = json.loads(out.read_text(encoding="utf-8"))
        assert data["extensions"] == []

    def test_export_extensions_includes_added_extensions(self, tmp_path):
        mgr = ExtensionManager()
        mgr.add_extension(
            force="Deploy", obj="Container",
            canonical="Deploy container image", coords=(3, 5, 0, 4),
            created_by="test-agent",
        )
        out = tmp_path / "extensions.json"
        mgr.export_extensions(out)
        data = json.loads(out.read_text(encoding="utf-8"))
        assert len(data["extensions"]) == 1
        ext = data["extensions"][0]
        assert ext["force"] == "Deploy"
        assert ext["obj"] == "Container"
        assert ext["canonical"] == "Deploy container image"
        assert ext["coords"] == [3, 5, 0, 4]
        assert ext["is_core"] is False
        assert ext["created_by"] == "test-agent"
        assert ext["index"] >= CORE_RANGE_END

    def test_export_extensions_includes_fallback_stats(self, tmp_path):
        mgr = ExtensionManager()
        for _ in range(3):
            mgr.record_fallback(thought="run the tests", src="a", dst="b")
        out = tmp_path / "extensions.json"
        mgr.export_extensions(out)
        data = json.loads(out.read_text(encoding="utf-8"))
        stats = data["fallback_stats"]
        assert stats["total_events"] == 3
        assert stats["unique_patterns"] > 0

    def test_export_extensions_multiple_anchors(self, tmp_path):
        mgr = ExtensionManager()
        mgr.add_extension(
            force="Deploy", obj="Container",
            canonical="c1", coords=(3, 5, 0, 4),
        )
        mgr.add_extension(
            force="Scale", obj="Service",
            canonical="c2", coords=(3, 5, 5, 4),
        )
        out = tmp_path / "extensions.json"
        mgr.export_extensions(out)
        data = json.loads(out.read_text(encoding="utf-8"))
        assert len(data["extensions"]) == 2

    def test_export_extensions_excludes_core_anchors(self, tmp_path):
        mgr = ExtensionManager()
        out = tmp_path / "extensions.json"
        mgr.export_extensions(out)
        data = json.loads(out.read_text(encoding="utf-8"))
        # Core anchors exist in the UCR but should not appear in the export
        assert data["extensions"] == []
        assert len(mgr.ucr.core_anchors()) > 0

    def test_export_overwrite_existing_file(self, tmp_path):
        mgr = ExtensionManager()
        out = tmp_path / "extensions.json"
        out.write_text("{}", encoding="utf-8")
        mgr.add_extension(
            force="Deploy", obj="Container",
            canonical="c1", coords=(3, 5, 0, 4),
        )
        mgr.export_extensions(out)
        data = json.loads(out.read_text(encoding="utf-8"))
        assert len(data["extensions"]) == 1

    # -- get_stats() ---

    def test_get_stats_structure(self):
        mgr = ExtensionManager()
        stats = mgr.get_stats()
        assert "core_anchors" in stats
        assert "extension_anchors" in stats
        assert "fallback_stats" in stats

    def test_get_stats_core_count(self):
        mgr = ExtensionManager()
        stats = mgr.get_stats()
        assert stats["core_anchors"] == len(mgr.ucr.core_anchors())
        assert stats["core_anchors"] > 0

    def test_get_stats_extension_count_zero_initially(self):
        mgr = ExtensionManager()
        stats = mgr.get_stats()
        assert stats["extension_anchors"] == 0

    def test_get_stats_extension_count_after_add(self):
        mgr = ExtensionManager()
        mgr.add_extension(
            force="Deploy", obj="Container",
            canonical="c1", coords=(3, 5, 0, 4),
        )
        stats = mgr.get_stats()
        assert stats["extension_anchors"] == 1

    def test_get_stats_fallback_stats_nested(self):
        mgr = ExtensionManager()
        stats = mgr.get_stats()
        fb = stats["fallback_stats"]
        assert "total_events" in fb
        assert "unique_patterns" in fb
        assert "top_patterns" in fb

    def test_get_stats_fallback_reflects_recordings(self):
        mgr = ExtensionManager()
        for _ in range(4):
            mgr.record_fallback(thought="check the logs", src="a", dst="b")
        stats = mgr.get_stats()
        assert stats["fallback_stats"]["total_events"] == 4

    def test_get_stats_combined(self):
        mgr = ExtensionManager()
        mgr.add_extension(
            force="Deploy", obj="Container",
            canonical="c1", coords=(3, 5, 0, 4),
        )
        mgr.add_extension(
            force="Scale", obj="Service",
            canonical="c2", coords=(3, 5, 5, 4),
        )
        for _ in range(3):
            mgr.record_fallback(thought="run the tests", src="a", dst="b")
        stats = mgr.get_stats()
        assert stats["extension_anchors"] == 2
        assert stats["fallback_stats"]["total_events"] == 3
        assert stats["core_anchors"] > 0

    # -- Integration: custom UCR ---

    def test_with_empty_ucr(self):
        """ExtensionManager works with a UCR that has no core anchors."""
        ucr = UCR(authority=UCRAuthority(authority_id="empty", ucr_version="3.0.0"))
        mgr = ExtensionManager(ucr=ucr)
        anchor = mgr.add_extension(
            force="Custom", obj="Action",
            canonical="Custom action", coords=(0, 0, 0, 0),
        )
        assert anchor.index == CORE_RANGE_END  # First extension starts at 0x8000
        stats = mgr.get_stats()
        assert stats["core_anchors"] == 0
        assert stats["extension_anchors"] == 1

    def test_event_sinks_isolated_between_tests(self):
        """Verify the autouse fixture prevents event leaks.

        This test registers a sink and checks it receives events.
        Thanks to the autouse fixture, this sink will be cleaned up
        before the next test runs.
        """
        captured = []
        register_sink(captured.append)
        mgr = ExtensionManager()
        mgr.add_extension(
            force="Test", obj="Isolation",
            canonical="test", coords=(0, 0, 0, 0),
        )
        assert len(captured) == 1

    def test_event_sinks_isolated_second_check(self):
        """Complement to the isolation test -- no residual sink from above."""
        captured = []
        register_sink(captured.append)
        mgr = ExtensionManager()
        mgr.add_extension(
            force="Test", obj="Isolation2",
            canonical="test2", coords=(1, 1, 1, 1),
        )
        # Should be exactly 1, not 2 (which would indicate the previous
        # test's sink leaked)
        assert len(captured) == 1
