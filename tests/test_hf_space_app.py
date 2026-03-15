from __future__ import annotations

from pathlib import Path


def test_custom_css_forces_light_text_and_non_default_fonts() -> None:
    app_source = (Path(__file__).resolve().parents[1] / "hf-space" / "app.py").read_text(
        encoding="utf-8"
    )

    assert ".hero-shell," in app_source
    assert ".gradio-container .prose" in app_source
    assert "Georgia" in app_source
    assert "Trebuchet MS" in app_source
