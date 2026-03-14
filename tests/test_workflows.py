from pathlib import Path

import pytest

WORKFLOWS = (
    Path(".github/workflows/ci.yml"),
    Path(".github/workflows/publish.yml"),
)


@pytest.mark.parametrize("workflow_path", WORKFLOWS)
def test_workflow_uses_node24_ready_checkout(workflow_path: Path) -> None:
    text = workflow_path.read_text(encoding="utf-8")
    assert "actions/checkout@v5" in text


@pytest.mark.parametrize("workflow_path", WORKFLOWS)
def test_workflow_uses_node24_ready_setup_python(workflow_path: Path) -> None:
    text = workflow_path.read_text(encoding="utf-8")
    assert "actions/setup-python@v6" in text
