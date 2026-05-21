"""Tests for decoding experiment helpers."""

from __future__ import annotations

import pytest

from govintel.training.decoding import (
    DecodingConfig,
    decoding_grid,
    score_decoding_output,
    select_best_config,
)


def test_decoding_grid_expands_required_temperature_top_k_top_p_matrix() -> None:
    configs = decoding_grid()

    assert len(configs) == 12
    assert configs[0] == DecodingConfig(temperature=0.0, top_k=50, top_p=0.9)
    assert configs[-1] == DecodingConfig(temperature=0.7, top_k=100, top_p=0.95)
    assert {config.temperature for config in configs} == {0.0, 0.3, 0.7}
    assert {config.top_k for config in configs} == {50, 100}
    assert {config.top_p for config in configs} == {0.9, 0.95}


def test_score_decoding_output_combines_entity_accuracy_and_faithfulness() -> None:
    result = score_decoding_output(
        DecodingConfig(temperature=0.3, top_k=50, top_p=0.9),
        answer=(
            "Alpha Analytics leads DHS cyber work through DHS-ALPHA-001. "
            "Beta Systems is not in the retrieved context."
        ),
        expected_contractors=["Alpha Analytics", "Beta Systems"],
        contexts=["DHS-ALPHA-001 Alpha Analytics Department of Homeland Security cyber work"],
        valid_contract_ids=["DHS-ALPHA-001"],
    )

    assert result.entity_accuracy == 1.0
    assert result.faithfulness < 1.0
    assert result.overall_score == pytest.approx((result.entity_accuracy + result.faithfulness) / 2)


def test_select_best_config_prefers_highest_overall_score() -> None:
    lower = score_decoding_output(
        DecodingConfig(temperature=0.7, top_k=100, top_p=0.95),
        answer="Alpha Analytics appears without source support.",
        expected_contractors=["Alpha Analytics"],
        contexts=["DHS-ALPHA-001 Alpha Analytics source context"],
        valid_contract_ids=["DHS-ALPHA-001"],
    )
    higher = score_decoding_output(
        DecodingConfig(temperature=0.0, top_k=50, top_p=0.9),
        answer="Alpha Analytics is grounded in DHS-ALPHA-001.",
        expected_contractors=["Alpha Analytics"],
        contexts=["DHS-ALPHA-001 Alpha Analytics source context"],
        valid_contract_ids=["DHS-ALPHA-001"],
    )

    assert select_best_config([lower, higher]) == higher

