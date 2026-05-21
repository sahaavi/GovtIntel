"""Tests for the offline ablation evaluation runner."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from govintel.evaluation.runner import (
    AblationRunner,
    EvalConfig,
    EvalPrediction,
    EvalResult,
)


def test_markdown_table_has_stable_columns_and_numeric_formatting() -> None:
    runner = AblationRunner(configs=[], test_queries_path="unused.json")
    results = [
        EvalResult(
            config_name="gemini-flash-rag",
            faithfulness=0.81234,
            answer_relevancy=0.7234,
            context_precision=0.9345,
            context_recall=0.6456,
            entity_accuracy=1.0,
            dollar_accuracy=0.5,
            citation_correctness=0.75,
            avg_latency_ms=123.456,
            avg_tokens=789,
        )
    ]

    table = runner.to_markdown_table(results)
    expected_header = (
        "| Config | Faithfulness | Answer Relevancy | Context Precision | Context Recall | "
        "Entity Accuracy | Dollar Accuracy | Citation Correctness | Avg Latency (ms) | "
        "Avg Tokens |"
    )

    assert table.splitlines() == [
        expected_header,
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        (
            "| gemini-flash-rag | 0.812 | 0.723 | 0.935 | 0.646 | 1.000 | "
            "0.500 | 0.750 | 123.5 | 789 |"
        ),
    ]


@pytest.mark.asyncio
async def test_runner_aggregates_custom_and_ragas_metrics_with_injected_predictions(
    tmp_path: Path,
) -> None:
    test_queries_path, gold_answers_path = _write_eval_files(tmp_path)
    config = EvalConfig(
        name="gemini-flash-rag",
        use_rag=True,
        model="gemini-flash",
        prompt_strategy="few_shot",
    )

    async def predict(config: EvalConfig, case: Any) -> EvalPrediction:
        assert config.name == "gemini-flash-rag"
        assert case.id == "q001"
        return EvalPrediction(
            answer=(
                "Booz Allen Hamilton leads with $50M in DHS cyber awards. "
                "Leidos follows with $30M. Citations: CONT_001, CONT_002."
            ),
            contexts=["CONT_001 Booz Allen Hamilton DHS cyber award evidence"],
            citations=["CONT_001", "CONT_002"],
            latency_ms=20.0,
            tokens=100,
        )

    def run_ragas(
        questions: list[str],
        answers: list[str],
        contexts: list[list[str]],
        ground_truths: list[str],
    ) -> dict[str, list[float]]:
        assert questions == ["Who are the top DHS cybersecurity contractors?"]
        assert len(answers) == len(contexts) == len(ground_truths) == 1
        return {
            "faithfulness": [0.9],
            "answer_relevancy": [0.8],
            "context_precision": [0.7],
            "context_recall": [0.6],
        }

    runner = AblationRunner(
        configs=[config],
        test_queries_path=test_queries_path,
        gold_answers_path=gold_answers_path,
        prediction_fn=predict,
        ragas_runner=run_ragas,
    )

    results = await runner.run_all()

    assert results == [
        EvalResult(
            config_name="gemini-flash-rag",
            faithfulness=0.9,
            answer_relevancy=0.8,
            context_precision=0.7,
            context_recall=0.6,
            entity_accuracy=1.0,
            dollar_accuracy=1.0,
            citation_correctness=1.0,
            avg_latency_ms=20.0,
            avg_tokens=100,
        )
    ]


@pytest.mark.asyncio
async def test_runner_can_disable_ragas_and_still_return_custom_metrics(tmp_path: Path) -> None:
    test_queries_path, gold_answers_path = _write_eval_files(tmp_path)

    async def predict(config: EvalConfig, case: Any) -> EvalPrediction:
        return EvalPrediction(
            answer="Booz Allen Hamilton leads with $50M. Citations: CONT_001.",
            contexts=[],
            citations=["CONT_001"],
            latency_ms=5.0,
            tokens=20,
        )

    runner = AblationRunner(
        configs=[
            EvalConfig(
                name="gemini-flash-no-rag",
                use_rag=False,
                model="gemini-flash",
                prompt_strategy="zero_shot",
            )
        ],
        test_queries_path=test_queries_path,
        gold_answers_path=gold_answers_path,
        prediction_fn=predict,
        ragas_runner=None,
    )

    result = (await runner.run_all())[0]

    assert result.faithfulness == 0.0
    assert result.context_precision == 0.0
    assert result.entity_accuracy == pytest.approx(0.5)
    assert result.dollar_accuracy == pytest.approx(0.5)
    assert result.citation_correctness == 1.0


def _write_eval_files(tmp_path: Path) -> tuple[Path, Path]:
    test_queries_path = tmp_path / "test_queries.json"
    gold_answers_path = tmp_path / "gold_answers.json"
    test_queries_path.write_text(
        json.dumps(
            [
                {
                    "id": "q001",
                    "question": "Who are the top DHS cybersecurity contractors?",
                    "agency_filter": "Department of Homeland Security",
                    "expected_contractors": ["Booz Allen Hamilton", "Leidos"],
                    "expected_dollar_ranges": {
                        "Booz Allen Hamilton": [45_000_000, 55_000_000],
                        "Leidos": [25_000_000, 35_000_000],
                    },
                    "expected_trends": "increasing",
                    "valid_contract_ids": ["CONT_001", "CONT_002"],
                    "quality_rating": 5,
                    "gold_answer_id": "q001",
                }
            ]
        )
    )
    gold_answers_path.write_text(
        json.dumps(
            {
                "q001": (
                    "Booz Allen Hamilton and Leidos are the top DHS cybersecurity "
                    "contractors in the gold answer."
                )
            }
        )
    )
    return test_queries_path, gold_answers_path
