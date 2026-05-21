"""Helpers for fine-tuned model decoding experiments."""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass

from govintel.evaluation.metrics import entity_accuracy

DEFAULT_TEMPERATURES = (0.0, 0.3, 0.7)
DEFAULT_TOP_KS = (50, 100)
DEFAULT_TOP_PS = (0.9, 0.95)
_CONTRACT_ID_PATTERN = re.compile(r"\b[A-Z]{2,}[A-Z0-9_-]*\d[A-Z0-9_-]*\b")
_TOKEN_PATTERN = re.compile(r"[a-z0-9]+")
_STOPWORDS = {
    "and",
    "are",
    "for",
    "from",
    "has",
    "have",
    "into",
    "not",
    "the",
    "this",
    "that",
    "through",
    "with",
    "without",
}


@dataclass(frozen=True)
class DecodingConfig:
    """One decoding parameter combination for experiment scoring."""

    temperature: float
    top_k: int
    top_p: float

    @property
    def name(self) -> str:
        """Return a stable label for result tables."""

        return f"temp={self.temperature}|top_k={self.top_k}|top_p={self.top_p}"


@dataclass(frozen=True)
class DecodingResult:
    """Scores for one decoded model answer."""

    config: DecodingConfig
    answer: str
    entity_accuracy: float
    faithfulness: float

    @property
    def overall_score(self) -> float:
        """Return the simple average used to rank decoding settings."""

        return (self.entity_accuracy + self.faithfulness) / 2


def decoding_grid(
    *,
    temperatures: Sequence[float] = DEFAULT_TEMPERATURES,
    top_ks: Sequence[int] = DEFAULT_TOP_KS,
    top_ps: Sequence[float] = DEFAULT_TOP_PS,
) -> list[DecodingConfig]:
    """Expand temperature, top-k, and top-p values into deterministic configs."""

    return [
        DecodingConfig(temperature=temperature, top_k=top_k, top_p=top_p)
        for temperature in temperatures
        for top_k in top_ks
        for top_p in top_ps
    ]


def score_decoding_output(
    config: DecodingConfig,
    *,
    answer: str,
    expected_contractors: Sequence[str],
    contexts: Sequence[str],
    valid_contract_ids: Sequence[str],
) -> DecodingResult:
    """Score one decoded answer for entity coverage and context faithfulness."""

    return DecodingResult(
        config=config,
        answer=answer,
        entity_accuracy=entity_accuracy(expected_contractors, answer),
        faithfulness=score_faithfulness(
            answer,
            contexts=contexts,
            valid_contract_ids=valid_contract_ids,
        ),
    )


def score_faithfulness(
    answer: str,
    *,
    contexts: Sequence[str],
    valid_contract_ids: Sequence[str],
) -> float:
    """Return a lightweight faithfulness proxy for decoding sweeps."""

    answer_tokens = _content_tokens(answer)
    context_tokens = _content_tokens(" ".join(contexts))
    if not answer_tokens or not context_tokens:
        return 0.0

    overlap_score = len(answer_tokens & context_tokens) / len(answer_tokens)
    answer_ids = _extract_contract_ids(answer)
    if not answer_ids:
        return overlap_score * 0.5

    normalized_valid = set(valid_contract_ids)
    supported_ids = sum(1 for contract_id in answer_ids if contract_id in normalized_valid)
    citation_score = supported_ids / len(answer_ids)
    return (overlap_score + citation_score) / 2


def select_best_config(results: Sequence[DecodingResult]) -> DecodingResult:
    """Return the highest-ranked decoding result."""

    if not results:
        raise ValueError("at least one decoding result is required")
    return max(
        results,
        key=lambda result: (
            result.overall_score,
            result.faithfulness,
            result.entity_accuracy,
            -result.config.temperature,
        ),
    )


def _extract_contract_ids(text: str) -> list[str]:
    """Extract unique contract-like IDs in first-seen order."""

    return list(dict.fromkeys(_CONTRACT_ID_PATTERN.findall(text)))


def _content_tokens(text: str) -> set[str]:
    """Return lowercased content tokens used for overlap scoring."""

    return {
        token
        for token in _TOKEN_PATTERN.findall(text.casefold())
        if len(token) > 2 and token not in _STOPWORDS
    }


__all__ = [
    "DecodingConfig",
    "DecodingResult",
    "decoding_grid",
    "score_decoding_output",
    "score_faithfulness",
    "select_best_config",
]

