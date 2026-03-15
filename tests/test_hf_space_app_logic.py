from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_app_logic_module():
    path = Path(__file__).resolve().parents[1] / "hf-space" / "app_logic.py"
    spec = importlib.util.spec_from_file_location("hf_space_app_logic", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Unable to load hf-space/app_logic.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_build_ucr_rows_returns_core_anchor_rows() -> None:
    logic = _load_app_logic_module()

    rows = logic.build_ucr_rows()

    assert len(rows) == 45
    assert rows[0]["force"] == "Observe"
    assert rows[0]["object"] == "State"
    assert rows[0]["index"] == "0x0001"


def test_build_ucr_rows_can_filter_by_force_and_search() -> None:
    logic = _load_app_logic_module()

    rows = logic.build_ucr_rows(force_filter="Request", search="review")

    assert len(rows) == 1
    assert rows[0]["force"] == "Request"
    assert rows[0]["object"] == "Review"


def test_analyze_wire_returns_valid_summary_for_good_message() -> None:
    logic = _load_app_logic_module()

    result = logic.analyze_wire("SLIP v3 planner reviewer Request Review auth")

    assert result["status"] == "valid"
    assert result["issues"] == []
    assert "Request Review" in result["human"]
    assert result["fields"]["force"] == "Request"
    assert result["fields"]["object"] == "Review"


def test_analyze_wire_returns_issues_for_invalid_message() -> None:
    logic = _load_app_logic_module()

    result = logic.analyze_wire("NOPE v3 planner reviewer Request Review auth")

    assert result["status"] == "invalid"
    assert any("Bad prefix" in issue for issue in result["issues"])
    assert result["human"] == ""


def test_get_langgraph_snippet_returns_adapter_example() -> None:
    logic = _load_app_logic_module()

    snippet = logic.get_langgraph_snippet("Boundary Encode/Decode")

    assert "LangGraphSlipstreamAdapter" in snippet
    assert "make_encode_node" in snippet
    assert "make_decode_node" in snippet


def test_get_training_guidance_marks_training_as_optional() -> None:
    logic = _load_app_logic_module()

    guidance = logic.get_training_guidance("When should I train?")

    assert "optional" in guidance.lower()
    assert "fallback" in guidance.lower()


def test_load_example_wires_falls_back_when_vectors_are_missing(
    monkeypatch, tmp_path
) -> None:
    logic = _load_app_logic_module()

    monkeypatch.setattr(logic, "VALID_VECTORS", tmp_path / "missing-valid.jsonl")
    monkeypatch.setattr(logic, "INVALID_VECTORS", tmp_path / "missing-invalid.jsonl")

    valid = logic.load_example_wires("Valid")
    invalid = logic.load_example_wires("Invalid")

    assert valid == logic.FALLBACK_VALID_WIRES
    assert invalid == logic.FALLBACK_INVALID_WIRES


def test_get_head_html_points_to_custom_space_assets() -> None:
    logic = _load_app_logic_module()

    head_html = logic.get_head_html()

    assert 'property="og:title"' in head_html
    assert "Slipstream Lab" in head_html
    assert 'name="twitter:image"' in head_html
    assert "slipstream-social-card.svg" in head_html
    assert 'rel="icon"' in head_html
    assert "slipstream-mark.svg" in head_html


def test_get_overview_markdown_guides_adoption() -> None:
    logic = _load_app_logic_module()

    overview = logic.get_overview_markdown()

    assert "You do not need to train a model" in overview
    assert "Inspect the protocol surface" in overview
    assert "LangGraph" in overview


def test_brand_assets_exist() -> None:
    assets_dir = Path(__file__).resolve().parents[1] / "hf-space" / "assets"

    assert (assets_dir / "slipstream-mark.svg").exists()
    assert (assets_dir / "slipstream-social-card.svg").exists()
