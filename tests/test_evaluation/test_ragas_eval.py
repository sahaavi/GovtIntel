"""Tests for the optional RAGAS evaluation adapter."""

from __future__ import annotations

from typing import Any

import pytest

from govintel.evaluation import ragas_eval


def test_run_ragas_evaluation_rejects_mismatched_input_lengths() -> None:
    with pytest.raises(ValueError, match="same length"):
        ragas_eval.run_ragas_evaluation(
            questions=["Who won?"],
            answers=[],
            contexts=[["context"]],
            ground_truths=["answer"],
        )


def test_run_ragas_evaluation_raises_clear_error_when_optional_dependencies_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unavailable() -> Any:
        raise ragas_eval.RagasUnavailableError("Install with pip install -e \".[eval]\"")

    monkeypatch.setattr(ragas_eval, "_load_ragas_dependencies", unavailable)

    with pytest.raises(ragas_eval.RagasUnavailableError, match=r"\.\[eval\]"):
        ragas_eval.run_ragas_evaluation(
            questions=["Who are the top DHS cyber contractors?"],
            answers=["Booz Allen Hamilton leads."],
            contexts=[["Booz Allen Hamilton contract evidence"]],
            ground_truths=["Booz Allen Hamilton leads DHS cyber awards."],
        )


def test_run_ragas_evaluation_builds_dataset_and_returns_plain_dict(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    class FakeDataset:
        @staticmethod
        def from_dict(payload: dict[str, Any]) -> tuple[str, dict[str, Any]]:
            captured["payload"] = payload
            return ("dataset", payload)

    class FakeFrame:
        def to_dict(self) -> dict[str, list[float]]:
            return {"faithfulness": [0.9], "answer_relevancy": [0.8]}

    class FakeResult:
        def to_pandas(self) -> FakeFrame:
            return FakeFrame()

    def fake_evaluate(*, dataset: Any, metrics: list[Any]) -> FakeResult:
        captured["dataset"] = dataset
        captured["metrics"] = metrics
        return FakeResult()

    monkeypatch.setattr(
        ragas_eval,
        "_load_ragas_dependencies",
        lambda: (
            fake_evaluate,
            FakeDataset,
            ["faithfulness", "answer_relevancy", "context_precision", "context_recall"],
        ),
    )

    result = ragas_eval.run_ragas_evaluation(
        questions=["Who are the top DHS cyber contractors?"],
        answers=["Booz Allen Hamilton leads."],
        contexts=[["Booz Allen Hamilton contract evidence"]],
        ground_truths=["Booz Allen Hamilton leads DHS cyber awards."],
    )

    assert captured["payload"] == {
        "question": ["Who are the top DHS cyber contractors?"],
        "answer": ["Booz Allen Hamilton leads."],
        "contexts": [["Booz Allen Hamilton contract evidence"]],
        "ground_truth": ["Booz Allen Hamilton leads DHS cyber awards."],
    }
    assert captured["dataset"] == ("dataset", captured["payload"])
    assert captured["metrics"] == [
        "faithfulness",
        "answer_relevancy",
        "context_precision",
        "context_recall",
    ]
    assert result == {"faithfulness": [0.9], "answer_relevancy": [0.8]}
