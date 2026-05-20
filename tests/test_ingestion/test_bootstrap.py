"""Tests for the minimal USAspending-to-Postgres bootstrap command."""

from __future__ import annotations

from datetime import date
from typing import Any

import pytest

from govintel.config import Settings
from govintel.ingestion import bootstrap
from govintel.models import ContractAward


class FakeClient:
    def __init__(self, awards: list[ContractAward]) -> None:
        self.awards = awards
        self.fetch_calls: list[dict[str, Any]] = []
        self.close_calls = 0

    async def fetch_all_awards(
        self,
        *,
        filters: dict[str, Any],
        max_pages: int,
        page_size: int,
    ) -> list[ContractAward]:
        self.fetch_calls.append(
            {
                "filters": filters,
                "max_pages": max_pages,
                "page_size": page_size,
            }
        )
        return self.awards

    async def close(self) -> None:
        self.close_calls += 1


class FakeLoader:
    def __init__(self) -> None:
        self.create_tables_calls = 0
        self.loaded_awards: list[ContractAward] = []
        self.close_calls = 0

    async def create_tables(self) -> None:
        self.create_tables_calls += 1

    async def load_awards(self, awards: list[ContractAward]) -> int:
        self.loaded_awards.extend(awards)
        return len(awards)

    async def close(self) -> None:
        self.close_calls += 1


def _award() -> ContractAward:
    return ContractAward(
        award_id="CONT_AWD_001",
        recipient_name="Example Corp",
        awarding_agency="Department of Homeland Security",
        award_amount=1_250_000.0,
        start_date=date(2024, 1, 1),
        end_date=date(2025, 1, 1),
        naics_code="541512",
        description="Cybersecurity services",
        place_of_performance_state="VA",
        award_type="Definitive Contract",
    )


@pytest.mark.asyncio
async def test_bootstrap_contract_data_fetches_creates_tables_and_loads_awards() -> None:
    client = FakeClient([_award()])
    loader = FakeLoader()

    result = await bootstrap.bootstrap_contract_data(
        settings=Settings(ingestion_max_pages=2, ingestion_page_size=25),
        client=client,
        loader=loader,
        naics_code="541512",
    )

    assert result.awards_fetched == 1
    assert result.awards_loaded == 1
    assert loader.create_tables_calls == 1
    assert loader.loaded_awards == client.awards
    assert client.fetch_calls == [
        {
            "filters": {
                "award_type_codes": ["A", "B", "C", "D"],
                "naics_codes": {"require": ["541512"]},
            },
            "max_pages": 2,
            "page_size": 25,
        }
    ]
    assert client.close_calls == 0
    assert loader.close_calls == 0


@pytest.mark.asyncio
async def test_bootstrap_contract_data_closes_owned_client_and_loader(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    created_client = FakeClient([])
    created_loader = FakeLoader()

    monkeypatch.setattr(bootstrap, "USAspendingClient", lambda settings: created_client)
    monkeypatch.setattr(bootstrap, "PostgresLoader", lambda settings: created_loader)

    result = await bootstrap.bootstrap_contract_data(settings=Settings())

    assert result.awards_fetched == 0
    assert result.awards_loaded == 0
    assert created_loader.create_tables_calls == 1
    assert created_client.close_calls == 1
    assert created_loader.close_calls == 1
