"""Offline ablation runner for procurement RAG evaluation."""

from __future__ import annotations

import inspect
import json
import re
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path
from time import perf_counter
from typing import TypeVar, cast

from govintel.evaluation.metrics import citation_correctness, dollar_accuracy, entity_accuracy
from govintel.evaluation.ragas_eval import RagasUnavailableError

CONTRACT_ID_PATTERN = re.compile(r"\b[A-Z]{2,}[A-Z0-9_-]*\d[A-Z0-9_-]*\b")
_TABLE_HEADER = (
    "| Config | Faithfulness | Answer Relevancy | Context Precision | Context Recall | "
    "Entity Accuracy | Dollar Accuracy | Citation Correctness | Avg Latency (ms) | "
    "Avg Tokens |"
)
_TABLE_SEPARATOR = "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"


@dataclass(frozen=True)
class EvalConfig:
    """A single model/retrieval/prompt configuration to evaluate."""

    name: str
    use_rag: bool
    model: str
    prompt_strategy: str


@dataclass(frozen=True)
class EvalCase:
    """Gold evaluation case loaded from checked-in JSON fixtures."""

    id: str
    question: str
    agency_filter: str
    expected_contractors: tuple[str, ...]
    expected_dollar_ranges: dict[str, tuple[float, float]]
    expected_trends: str
    valid_contract_ids: tuple[str, ...]
    quality_rating: int
    gold_answer: str

    def expected_dollar_amounts(self) -> dict[str, float]:
        """Return midpoint dollar amounts for deterministic point-estimate scoring."""

        return {
            contractor: (low + high) / 2
            for contractor, (low, high) in self.expected_dollar_ranges.items()
        }


@dataclass(frozen=True)
class EvalPrediction:
    """Prediction and trace data returned by an injected evaluation pipeline."""

    answer: str
    contexts: list[str] = field(default_factory=list)
    citations: list[str] = field(default_factory=list)
    latency_ms: float = 0.0
    tokens: int = 0


@dataclass(frozen=True)
class EvalResult:
    """Aggregate metrics for one evaluation configuration."""

    config_name: str
    faithfulness: float
    answer_relevancy: float
    context_precision: float
    context_recall: float
    entity_accuracy: float
    dollar_accuracy: float
    citation_correctness: float
    avg_latency_ms: float
    avg_tokens: int


PredictionFn = Callable[[EvalConfig, EvalCase], EvalPrediction | Awaitable[EvalPrediction]]
RagasRunner = Callable[
    [list[str], list[str], list[list[str]], list[str]],
    Mapping[str, object],
]
T = TypeVar("T")


class AblationRunner:
    """Run injected ablation configurations against checked-in evaluation cases."""

    def __init__(
        self,
        configs: Sequence[EvalConfig],
        test_queries_path: str | Path,
        *,
        gold_answers_path: str | Path | None = None,
        prediction_fn: PredictionFn | None = None,
        ragas_runner: RagasRunner | None = None,
    ) -> None:
        self._configs = list(configs)
        self._test_queries_path = Path(test_queries_path)
        self._gold_answers_path = (
            Path(gold_answers_path)
            if gold_answers_path is not None
            else self._test_queries_path.with_name("gold_answers.json")
        )
        self._prediction_fn = prediction_fn
        self._ragas_runner = ragas_runner

    def load_cases(self) -> list[EvalCase]:
        """Load evaluation cases and gold answers from JSON fixtures."""

        raw_cases = _load_json(self._test_queries_path)
        raw_gold_answers = _load_json(self._gold_answers_path)
        if not isinstance(raw_cases, list):
            raise ValueError("test query fixture must be a list")
        if not isinstance(raw_gold_answers, dict):
            raise ValueError("gold answers fixture must be an object keyed by case id")

        return [
            _eval_case_from_mapping(raw_case, cast(Mapping[str, object], raw_gold_answers))
            for raw_case in raw_cases
        ]

    async def run_all(self) -> list[EvalResult]:
        """Run all configurations against all test queries."""

        if self._prediction_fn is None:
            raise RuntimeError("AblationRunner requires an injected prediction_fn")

        cases = self.load_cases()
        results: list[EvalResult] = []
        for config in self._configs:
            predictions: list[EvalPrediction] = []
            entity_scores: list[float] = []
            dollar_scores: list[float] = []
            citation_scores: list[float] = []
            latencies: list[float] = []
            token_counts: list[int] = []

            for case in cases:
                start = perf_counter()
                prediction = await _maybe_await(self._prediction_fn(config, case))
                measured_latency_ms = (perf_counter() - start) * 1000
                predictions.append(prediction)
                entity_scores.append(entity_accuracy(case.expected_contractors, prediction.answer))
                dollar_scores.append(
                    dollar_accuracy(case.expected_dollar_amounts(), prediction.answer)
                )
                citation_scores.append(
                    citation_correctness(
                        case.valid_contract_ids,
                        prediction.citations or _extract_contract_ids(prediction.answer),
                    )
                )
                latencies.append(prediction.latency_ms or measured_latency_ms)
                token_counts.append(prediction.tokens)

            ragas_scores = self._run_ragas(cases, predictions)
            results.append(
                EvalResult(
                    config_name=config.name,
                    faithfulness=_mean_metric(ragas_scores, "faithfulness"),
                    answer_relevancy=_mean_metric(ragas_scores, "answer_relevancy"),
                    context_precision=_mean_metric(ragas_scores, "context_precision"),
                    context_recall=_mean_metric(ragas_scores, "context_recall"),
                    entity_accuracy=_mean(entity_scores),
                    dollar_accuracy=_mean(dollar_scores),
                    citation_correctness=_mean(citation_scores),
                    avg_latency_ms=_mean(latencies),
                    avg_tokens=round(_mean([float(count) for count in token_counts])),
                )
            )
        return results

    def to_markdown_table(self, results: Sequence[EvalResult]) -> str:
        """Render evaluation results as a deterministic markdown table."""

        lines = [
            _TABLE_HEADER,
            _TABLE_SEPARATOR,
        ]
        for result in results:
            lines.append(
                "| "
                + " | ".join(
                    [
                        result.config_name,
                        _format_decimal(result.faithfulness, places=3),
                        _format_decimal(result.answer_relevancy, places=3),
                        _format_decimal(result.context_precision, places=3),
                        _format_decimal(result.context_recall, places=3),
                        _format_decimal(result.entity_accuracy, places=3),
                        _format_decimal(result.dollar_accuracy, places=3),
                        _format_decimal(result.citation_correctness, places=3),
                        _format_decimal(result.avg_latency_ms, places=1),
                        str(result.avg_tokens),
                    ]
                )
                + " |"
            )
        return "\n".join(lines)

    def _run_ragas(
        self,
        cases: Sequence[EvalCase],
        predictions: Sequence[EvalPrediction],
    ) -> Mapping[str, object]:
        """Run optional RAGAS scoring if a runner was injected."""

        if self._ragas_runner is None or not predictions:
            return {}

        try:
            return self._ragas_runner(
                [case.question for case in cases],
                [prediction.answer for prediction in predictions],
                [prediction.contexts for prediction in predictions],
                [case.gold_answer for case in cases],
            )
        except RagasUnavailableError:
            return {}


def _load_json(path: Path) -> object:
    """Load JSON from disk."""

    with path.open() as handle:
        return json.load(handle)


def _eval_case_from_mapping(raw_case: object, gold_answers: Mapping[str, object]) -> EvalCase:
    """Parse an EvalCase from untyped fixture JSON."""

    if not isinstance(raw_case, Mapping):
        raise ValueError("each test query must be an object")

    case_id = _required_str(raw_case, "id")
    gold_answer = gold_answers.get(case_id)
    if not isinstance(gold_answer, str) or not gold_answer:
        raise ValueError(f"missing gold answer for {case_id}")

    return EvalCase(
        id=case_id,
        question=_required_str(raw_case, "question"),
        agency_filter=_required_str(raw_case, "agency_filter"),
        expected_contractors=tuple(_required_str_list(raw_case, "expected_contractors")),
        expected_dollar_ranges=_dollar_ranges(raw_case.get("expected_dollar_ranges", {})),
        expected_trends=_required_str(raw_case, "expected_trends"),
        valid_contract_ids=tuple(_required_str_list(raw_case, "valid_contract_ids")),
        quality_rating=_required_int(raw_case, "quality_rating"),
        gold_answer=gold_answer,
    )


def _required_str(raw_case: Mapping[object, object], key: str) -> str:
    """Read a required non-empty string from a JSON object."""

    value = raw_case.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{key} must be a non-empty string")
    return value


def _required_int(raw_case: Mapping[object, object], key: str) -> int:
    """Read a required integer from a JSON object."""

    value = raw_case.get(key)
    if not isinstance(value, int):
        raise ValueError(f"{key} must be an integer")
    return value


def _required_str_list(raw_case: Mapping[object, object], key: str) -> list[str]:
    """Read a required list of non-empty strings from a JSON object."""

    value = raw_case.get(key)
    if not isinstance(value, list) or not value:
        raise ValueError(f"{key} must be a non-empty list")
    if not all(isinstance(item, str) and item for item in value):
        raise ValueError(f"{key} must contain only non-empty strings")
    return cast(list[str], value)


def _dollar_ranges(value: object) -> dict[str, tuple[float, float]]:
    """Parse expected dollar ranges from fixture JSON."""

    if not isinstance(value, Mapping):
        raise ValueError("expected_dollar_ranges must be an object")

    parsed: dict[str, tuple[float, float]] = {}
    for contractor, bounds in value.items():
        if not isinstance(contractor, str):
            raise ValueError("expected_dollar_ranges keys must be contractor names")
        if (
            not isinstance(bounds, list)
            or len(bounds) != 2
            or not isinstance(bounds[0], int | float)
            or not isinstance(bounds[1], int | float)
        ):
            raise ValueError("expected_dollar_ranges values must be [low, high]")
        parsed[contractor] = (float(bounds[0]), float(bounds[1]))
    return parsed


def _extract_contract_ids(text: str) -> list[str]:
    """Extract contract-like IDs from answer text."""

    return list(dict.fromkeys(CONTRACT_ID_PATTERN.findall(text)))


def _mean_metric(scores: Mapping[str, object], name: str) -> float:
    """Return the mean numeric value for a metric name from a RAGAS result dict."""

    return _mean(_numeric_values(scores.get(name)))


def _numeric_values(value: object) -> list[float]:
    """Flatten common scalar/list/dict numeric score shapes."""

    if isinstance(value, bool) or value is None:
        return []
    if isinstance(value, int | float):
        return [float(value)]
    if isinstance(value, Mapping):
        return [
            number
            for nested_value in value.values()
            for number in _numeric_values(nested_value)
        ]
    if isinstance(value, list | tuple):
        return [
            number
            for nested_value in value
            for number in _numeric_values(nested_value)
        ]
    return []


def _mean(values: Sequence[float]) -> float:
    """Return arithmetic mean or zero for empty inputs."""

    if not values:
        return 0.0
    return sum(values) / len(values)


def _format_decimal(value: float, *, places: int) -> str:
    """Format decimal values with deterministic half-up rounding."""

    quantizer = Decimal("1").scaleb(-places)
    rounded = Decimal(str(value)).quantize(quantizer, rounding=ROUND_HALF_UP)
    return f"{rounded:.{places}f}"


async def _maybe_await(value: T | Awaitable[T]) -> T:
    """Await only when an injected callable returns an awaitable."""

    if inspect.isawaitable(value):
        return await value
    return value


__all__ = [
    "AblationRunner",
    "EvalCase",
    "EvalConfig",
    "EvalPrediction",
    "EvalResult",
]
