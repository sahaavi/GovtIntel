"""Tests for deterministic procurement evaluation metrics."""

from __future__ import annotations

import pytest

from govintel.evaluation.metrics import (
    citation_correctness,
    dollar_accuracy,
    entity_accuracy,
)


def test_entity_accuracy_counts_expected_contractors_in_generated_text() -> None:
    expected_contractors = ["Booz Allen Hamilton", "Leidos", "SAIC"]
    generated_text = (
        "Booz Allen-Hamilton leads with $50M. "
        "LEIDOS follows at $30M. Accenture Federal appears at $20M."
    )

    score = entity_accuracy(expected_contractors, generated_text)

    assert score == pytest.approx(2 / 3)


def test_entity_accuracy_rejects_partial_word_matches() -> None:
    generated_text = "The mosaic platform mentions saico systems, not the expected firm."

    score = entity_accuracy(["SAIC"], generated_text)

    assert score == 0.0


def test_entity_accuracy_returns_zero_for_empty_expectations() -> None:
    assert entity_accuracy([], "Booz Allen Hamilton appears in the answer.") == 0.0


def test_dollar_accuracy_counts_amounts_within_tolerance() -> None:
    expected_amounts = {
        "Booz Allen Hamilton": 50_000_000,
        "Leidos": 30_000_000,
        "SAIC": 25_000_000,
    }
    generated_text = (
        "Booz Allen Hamilton: $48.5M in awards. "
        "Leidos received $30,000,000. "
        "SAIC is mentioned, but no amount is provided."
    )

    score = dollar_accuracy(expected_amounts, generated_text)

    assert score == pytest.approx(2 / 3)


def test_dollar_accuracy_supports_k_m_b_suffixes_and_raw_amounts() -> None:
    generated_text = (
        "Alpha Corp received $750K. "
        "Beta LLC received 1.25 million. "
        "Gamma Inc received $2B."
    )

    assert dollar_accuracy({"Alpha Corp": 750_000}, generated_text) == 1.0
    assert dollar_accuracy({"Beta LLC": 1_250_000}, generated_text) == 1.0
    assert dollar_accuracy({"Gamma Inc": 2_000_000_000}, generated_text) == 1.0


def test_dollar_accuracy_handles_empty_zero_and_negative_expected_values() -> None:
    assert dollar_accuracy({}, "No awards.") == 0.0
    assert dollar_accuracy({"No Spend LLC": 0}, "No Spend LLC received $0.") == 1.0
    assert dollar_accuracy({"Refund Corp": -1_000}, "Refund Corp received -$1,000.") == 0.0


def test_citation_correctness_scores_unique_supported_citations() -> None:
    valid_contract_ids = {"CONT_001", "CONT_002", "CONT_003"}
    citations = ["CONT_001", "CONT_001", "CONT_002:chunk:0", "CONT_999"]

    score = citation_correctness(valid_contract_ids, citations)

    assert score == pytest.approx(2 / 3)


def test_citation_correctness_returns_zero_without_citations() -> None:
    assert citation_correctness({"CONT_001"}, []) == 0.0
