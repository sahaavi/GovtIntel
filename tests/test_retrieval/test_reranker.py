"""Tests for cross-encoder reranking."""

from __future__ import annotations

from typing import Any

import pytest

from govintel.retrieval import reranker
from govintel.retrieval.reranker import CrossEncoderReranker


class FakeCrossEncoder:
    """Deterministic stand-in for sentence-transformers CrossEncoder."""

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name

    def predict(self, pairs: list[tuple[str, str]]) -> list[float]:
        score_by_text = {
            "generic IT support": 0.1,
            "DHS cybersecurity monitoring": 0.9,
            "cloud migration services": 0.4,
        }
        return [score_by_text[text] for _, text in pairs]


def test_cross_encoder_reranker_scores_and_sorts_candidates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(reranker, "CrossEncoder", FakeCrossEncoder)

    ranker = CrossEncoderReranker()

    results = ranker.rerank(
        "DHS cybersecurity contracts",
        [
            "generic IT support",
            "DHS cybersecurity monitoring",
            "cloud migration services",
        ],
        top_k=2,
    )

    assert [result.text for result in results] == [
        "DHS cybersecurity monitoring",
        "cloud migration services",
    ]
    assert [result.score for result in results] == [0.9, 0.4]
    assert results[0].doc_id == "1"
    assert results[0].metadata == {
        "candidate_index": 1,
        "reranker_model": "cross-encoder/ms-marco-MiniLM-L-6-v2",
    }


def test_cross_encoder_reranker_handles_blank_query_without_loading_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_if_loaded(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("CrossEncoder should not load for blank queries")

    monkeypatch.setattr(reranker, "CrossEncoder", fail_if_loaded)

    ranker = CrossEncoderReranker()

    assert ranker.rerank("   ", ["DHS cybersecurity monitoring"], top_k=5) == []


def test_cross_encoder_reranker_respects_top_k(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(reranker, "CrossEncoder", FakeCrossEncoder)

    ranker = CrossEncoderReranker()

    results = ranker.rerank(
        "DHS cybersecurity contracts",
        [
            "generic IT support",
            "DHS cybersecurity monitoring",
            "cloud migration services",
        ],
        top_k=1,
    )

    assert len(results) == 1
    assert results[0].text == "DHS cybersecurity monitoring"
