"""Optional RAGAS adapter for offline RAG evaluation."""

from __future__ import annotations

import importlib
from collections.abc import Callable, Sequence
from typing import Any, cast


class RagasUnavailableError(RuntimeError):
    """Raised when optional RAGAS evaluation dependencies are unavailable."""


def run_ragas_evaluation(
    questions: Sequence[str],
    answers: Sequence[str],
    contexts: Sequence[Sequence[str]],
    ground_truths: Sequence[str],
) -> dict[str, object]:
    """Run RAGAS metrics and return a plain dictionary of scores."""

    _validate_equal_lengths(questions, answers, contexts, ground_truths)
    evaluate, dataset_cls, metrics = _load_ragas_dependencies()
    dataset = dataset_cls.from_dict(
        {
            "question": list(questions),
            "answer": list(answers),
            "contexts": [list(context) for context in contexts],
            "ground_truth": list(ground_truths),
        }
    )
    result = evaluate(dataset=dataset, metrics=metrics)
    return _result_to_dict(result)


def _validate_equal_lengths(*values: Sequence[object]) -> None:
    """Require all input sequences to have the same length."""

    lengths = {len(value) for value in values}
    if len(lengths) > 1:
        raise ValueError("questions, answers, contexts, and ground_truths must have same length")


def _load_ragas_dependencies() -> tuple[Callable[..., Any], Any, list[Any]]:
    """Load optional RAGAS dependencies lazily."""

    try:
        datasets_module = importlib.import_module("datasets")
        ragas_module = importlib.import_module("ragas")
        metrics_module = importlib.import_module("ragas.metrics")
    except ImportError as exc:
        raise RagasUnavailableError(
            "RAGAS evaluation requires optional dependencies. "
            'Install with pip install -e ".[eval]".'
        ) from exc

    return (
        getattr(ragas_module, "evaluate"),
        getattr(datasets_module, "Dataset"),
        [
            getattr(metrics_module, "faithfulness"),
            getattr(metrics_module, "answer_relevancy"),
            getattr(metrics_module, "context_precision"),
            getattr(metrics_module, "context_recall"),
        ],
    )


def _result_to_dict(result: Any) -> dict[str, object]:
    """Normalize common RAGAS result shapes to a plain dict."""

    if isinstance(result, dict):
        return cast(dict[str, object], result)

    to_pandas = getattr(result, "to_pandas", None)
    if callable(to_pandas):
        frame = to_pandas()
        to_dict = getattr(frame, "to_dict", None)
        if callable(to_dict):
            try:
                return cast(dict[str, object], to_dict(orient="list"))
            except TypeError:
                return cast(dict[str, object], to_dict())

    scores = getattr(result, "scores", None)
    if isinstance(scores, dict):
        return cast(dict[str, object], scores)

    return {"result": result}


__all__ = ["RagasUnavailableError", "run_ragas_evaluation"]
