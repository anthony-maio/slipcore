from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _load_app_module():
    path = Path(__file__).resolve().parents[1] / "hf-space" / "app.py"
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location("hf_space_app", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Unable to load hf-space/app.py")
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.pop(0)


def test_custom_css_forces_light_text_and_non_default_fonts() -> None:
    app = _load_app_module()

    css = app.CUSTOM_CSS

    assert ".hero-shell," in css
    assert ".gradio-container .prose" in css
    assert "Georgia" in css
    assert "Trebuchet MS" in css
