"""Tests for slipcore.finetune template-based dataset generator."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from slipcore.finetune import (
    TrainingExample,
    generate_dataset,
    generate_training_examples,
    to_alpaca_format,
    to_chat_format,
    to_sharegpt_format,
    to_sharegpt_with_semantics,
    to_sharegpt_with_thought,
)


class TestGenerateTrainingExamples:
    def test_returns_requested_count(self):
        examples = generate_training_examples(num_examples=20)
        assert len(examples) == 20

    def test_returns_all_if_fewer_than_requested(self):
        examples = generate_training_examples(num_examples=10_000)
        assert len(examples) > 0
        assert len(examples) <= 10_000

    def test_all_examples_have_valid_wire(self):
        examples = generate_training_examples(num_examples=50)
        for ex in examples:
            assert ex.response.startswith("SLIP v3 "), f"Bad wire: {ex.response}"

    def test_all_examples_have_instruction(self):
        examples = generate_training_examples(num_examples=50)
        for ex in examples:
            assert ex.instruction, "Empty instruction"

    def test_all_examples_have_thought(self):
        examples = generate_training_examples(num_examples=50)
        for ex in examples:
            assert ex.thought, "Empty thought"

    def test_all_examples_have_force_and_obj(self):
        examples = generate_training_examples(num_examples=50)
        for ex in examples:
            assert ex.force, "Empty force"
            assert ex.obj, "Empty obj"

    def test_seeded_generation_is_deterministic(self):
        a = generate_training_examples(num_examples=30, seed=123)
        b = generate_training_examples(num_examples=30, seed=123)
        assert [e.response for e in a] == [e.response for e in b]

    def test_different_seeds_produce_different_output(self):
        a = generate_training_examples(num_examples=30, seed=1)
        b = generate_training_examples(num_examples=30, seed=2)
        assert [e.response for e in a] != [e.response for e in b]

    def test_includes_fallbacks_by_default(self):
        examples = generate_training_examples(num_examples=500)
        fallbacks = [e for e in examples if e.is_fallback]
        assert len(fallbacks) > 0

    def test_can_exclude_fallbacks(self):
        examples = generate_training_examples(num_examples=500, include_fallbacks=False)
        fallbacks = [e for e in examples if e.is_fallback]
        assert len(fallbacks) == 0

    def test_fallback_refs_are_not_placeholder(self):
        examples = generate_training_examples(num_examples=500)
        fallbacks = [e for e in examples if e.is_fallback]
        for ex in fallbacks:
            assert "refXXXX" not in ex.response, f"Placeholder ref in: {ex.response}"
            assert "ref" in ex.response

    def test_diverse_forces(self):
        examples = generate_training_examples(num_examples=200)
        forces = {e.force for e in examples}
        assert len(forces) >= 4, f"Only {forces} forces generated"


class TestOutputFormats:
    @pytest.fixture
    def example(self) -> TrainingExample:
        return TrainingExample(
            instruction="Review the auth module",
            thought="User wants code review",
            response="SLIP v3 dev reviewer Request Review auth",
            force="Request",
            obj="Review",
            dst="reviewer",
        )

    def test_sharegpt_format(self, example: TrainingExample):
        out = to_sharegpt_format(example)
        assert "conversations" in out
        convos = out["conversations"]
        assert len(convos) == 3
        assert convos[0]["from"] == "system"
        assert convos[1]["from"] == "human"
        assert convos[2]["from"] == "gpt"
        assert convos[1]["value"] == example.instruction
        assert convos[2]["value"] == example.response

    def test_sharegpt_with_thought(self, example: TrainingExample):
        out = to_sharegpt_with_thought(example)
        gpt_value = out["conversations"][2]["value"]
        assert gpt_value.startswith("THOUGHT:")
        assert "SLIP:" in gpt_value
        assert example.response in gpt_value

    def test_sharegpt_with_semantics(self, example: TrainingExample):
        out = to_sharegpt_with_semantics(example)
        gpt_value = out["conversations"][2]["value"]
        assert "THOUGHT:" in gpt_value
        assert "QUANTIZE:" in gpt_value
        assert "Force=Request" in gpt_value
        assert "Object=Review" in gpt_value
        assert "SLIP:" in gpt_value

    def test_chat_format(self, example: TrainingExample):
        out = to_chat_format(example)
        assert "messages" in out
        msgs = out["messages"]
        assert len(msgs) == 3
        assert msgs[0]["role"] == "system"
        assert msgs[1]["role"] == "user"
        assert msgs[2]["role"] == "assistant"

    def test_alpaca_format(self, example: TrainingExample):
        out = to_alpaca_format(example)
        assert "instruction" in out
        assert "input" in out
        assert "output" in out
        assert out["output"] == example.response


class TestGenerateDataset:
    @pytest.mark.parametrize("fmt", [
        "sharegpt",
        "sharegpt_thought",
        "sharegpt_semantics",
        "chat",
        "alpaca",
    ])
    def test_writes_valid_jsonl(self, tmp_path: Path, fmt: str):
        out = tmp_path / "train.jsonl"
        count = generate_dataset(out, num_examples=10, format=fmt)
        assert count == 10
        assert out.exists()

        lines = out.read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) == 10
        for line in lines:
            parsed = json.loads(line)
            assert isinstance(parsed, dict)

    def test_default_format_is_sharegpt_thought(self, tmp_path: Path):
        out = tmp_path / "train.jsonl"
        generate_dataset(out, num_examples=5)
        line = json.loads(out.read_text(encoding="utf-8").split("\n")[0])
        gpt_value = line["conversations"][2]["value"]
        assert gpt_value.startswith("THOUGHT:")

    def test_returns_example_count(self, tmp_path: Path):
        out = tmp_path / "train.jsonl"
        count = generate_dataset(out, num_examples=15)
        assert count == 15
