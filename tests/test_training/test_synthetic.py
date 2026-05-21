"""Tests for synthetic fine-tuning data generation."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from datetime import date
from pathlib import Path
from typing import Any

import pytest
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

from govintel.ingestion.loader import contracts_table, metadata
from govintel.models import ContractAward
from govintel.training.synthetic import (
    ContractGroup,
    TrainingExample,
    build_training_example,
    fetch_contract_groups,
    generate_training_examples,
    validate_curated_jsonl,
    write_jsonl,
)


class FakeGenerator:
    """Deterministic stand-in for a Gemini-compatible generator."""

    def __init__(self, responses: list[str]) -> None:
        self.responses = responses
        self.calls: list[dict[str, str]] = []

    async def generate(self, system: str, user_message: str) -> str:
        self.calls.append({"system": system, "user_message": user_message})
        return self.responses.pop(0)


@pytest.fixture
async def contract_engine() -> AsyncIterator[AsyncEngine]:
    """Database with multiple agency/contractor groups."""

    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(metadata.create_all)
        await conn.execute(
            contracts_table.insert(),
            [
                _contract_row(
                    award_id="DHS-ALPHA-001",
                    recipient_name="Alpha Analytics",
                    awarding_agency="Department of Homeland Security",
                    award_amount=5_000_000.0,
                    start_date=date(2024, 1, 15),
                    description="Zero trust cybersecurity monitoring",
                ),
                _contract_row(
                    award_id="DHS-ALPHA-002",
                    recipient_name="Alpha Analytics",
                    awarding_agency="Department of Homeland Security",
                    award_amount=7_500_000.0,
                    start_date=date(2024, 5, 1),
                    description="SOC engineering and threat hunting",
                ),
                _contract_row(
                    award_id="DOD-BETA-001",
                    recipient_name="Beta Systems",
                    awarding_agency="Department of Defense",
                    award_amount=9_000_000.0,
                    start_date=date(2024, 3, 20),
                    description="Cloud migration support",
                ),
            ],
        )

    try:
        yield engine
    finally:
        await engine.dispose()


def test_build_training_example_formats_context_and_jsonl_record() -> None:
    group = ContractGroup(
        agency="Department of Homeland Security",
        contractor="Alpha Analytics",
        awards=(
            ContractAward(
                award_id="DHS-ALPHA-001",
                recipient_name="Alpha Analytics",
                awarding_agency="Department of Homeland Security",
                award_amount=5_000_000.0,
                start_date=date(2024, 1, 15),
                end_date=date(2025, 1, 14),
                naics_code="541512",
                description="Zero trust cybersecurity monitoring",
                place_of_performance_state="VA",
                award_type="Definitive Contract",
            ),
            ContractAward(
                award_id="DHS-ALPHA-002",
                recipient_name="Alpha Analytics",
                awarding_agency="Department of Homeland Security",
                award_amount=7_500_000.0,
                start_date=date(2024, 5, 1),
                end_date=None,
                naics_code="541512",
                description="SOC engineering and threat hunting",
                place_of_performance_state="MD",
                award_type="Delivery Order",
            ),
        ),
    )

    example = build_training_example(
        group,
        response=(
            "Alpha Analytics is expanding its DHS cyber footprint through "
            "DHS-ALPHA-001 and DHS-ALPHA-002."
        ),
    )

    assert example.instruction == (
        "Write a grounded GovCon intelligence report snippet for the contractor "
        "and agency in the provided contract context."
    )
    assert "Agency: Department of Homeland Security" in example.context
    assert "Contractor: Alpha Analytics" in example.context
    assert "Award count: 2" in example.context
    assert "Total obligated value: $12,500,000.00" in example.context
    assert "Date range: 2024-01-15 to 2024-05-01" in example.context
    assert "NAICS codes: 541512" in example.context
    assert "DHS-ALPHA-001" in example.context
    assert "DHS-ALPHA-002" in example.context
    assert example.to_json_record() == {
        "instruction": example.instruction,
        "context": example.context,
        "response": example.response,
    }


@pytest.mark.asyncio
async def test_fetch_contract_groups_groups_records_by_agency_and_contractor(
    contract_engine: AsyncEngine,
) -> None:
    groups = await fetch_contract_groups(
        contract_engine,
        group_limit=10,
        contracts_per_group=2,
    )

    assert [(group.agency, group.contractor) for group in groups] == [
        ("Department of Defense", "Beta Systems"),
        ("Department of Homeland Security", "Alpha Analytics"),
    ]
    assert [award.award_id for award in groups[1].awards] == [
        "DHS-ALPHA-002",
        "DHS-ALPHA-001",
    ]


@pytest.mark.asyncio
async def test_generate_training_examples_calls_generator_and_rejects_ungrounded_responses(
    contract_engine: AsyncEngine,
) -> None:
    groups = await fetch_contract_groups(contract_engine, group_limit=2, contracts_per_group=2)
    generator = FakeGenerator(
        [
            "Beta Systems response without a source award id.",
            "Alpha Analytics has grounded DHS growth through DHS-ALPHA-001.",
        ]
    )

    examples = await generate_training_examples(groups, generator)

    assert len(examples) == 1
    assert examples[0].response == "Alpha Analytics has grounded DHS growth through DHS-ALPHA-001."
    assert len(generator.calls) == 2
    assert "You generate factual procurement intelligence snippets" in generator.calls[0]["system"]
    assert "Contract context:" in generator.calls[0]["user_message"]


def test_write_jsonl_writes_round_trippable_training_records(tmp_path: Path) -> None:
    output_path = tmp_path / "training.jsonl"
    example = TrainingExample(
        instruction="Write a grounded snippet.",
        context="Award ID: DHS-ALPHA-001",
        response="Alpha Analytics is supported by DHS-ALPHA-001.",
    )

    written = write_jsonl([example], output_path)

    assert written == 1
    rows = [json.loads(line) for line in output_path.read_text().splitlines()]
    assert rows == [
        {
            "instruction": "Write a grounded snippet.",
            "context": "Award ID: DHS-ALPHA-001",
            "response": "Alpha Analytics is supported by DHS-ALPHA-001.",
        }
    ]


def test_validate_curated_jsonl_requires_fifty_high_quality_examples(tmp_path: Path) -> None:
    curated_path = tmp_path / "curated.jsonl"
    write_jsonl(
        [
            TrainingExample(
                instruction="Write a grounded snippet.",
                context=f"Award ID: DHS-ALPHA-{index:03d}",
                response=f"Alpha Analytics is supported by DHS-ALPHA-{index:03d}.",
            )
            for index in range(49)
        ],
        curated_path,
    )

    with pytest.raises(ValueError, match="at least 50"):
        validate_curated_jsonl(curated_path)

    write_jsonl(
        [
            TrainingExample(
                instruction="Write a grounded snippet.",
                context=f"Award ID: DHS-ALPHA-{index:03d}",
                response=f"Alpha Analytics is supported by DHS-ALPHA-{index:03d}.",
            )
            for index in range(50)
        ],
        curated_path,
    )

    assert validate_curated_jsonl(curated_path) == 50


def _contract_row(
    *,
    award_id: str,
    recipient_name: str,
    awarding_agency: str,
    award_amount: float,
    start_date: date,
    description: str,
) -> dict[str, Any]:
    return {
        "award_id": award_id,
        "recipient_name": recipient_name,
        "awarding_agency": awarding_agency,
        "award_amount": award_amount,
        "start_date": start_date,
        "end_date": None,
        "naics_code": "541512",
        "description": description,
        "place_of_performance_state": "VA",
        "award_type": "Definitive Contract",
    }

