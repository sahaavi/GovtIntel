"""Tests for the PostgreSQL contract loader."""

from __future__ import annotations

from datetime import date
from typing import Any, cast

import pytest

from govintel.config import Settings
from govintel.ingestion.loader import PostgresLoader
from govintel.models import ContractAward


class FakeScalarResult:
    def __init__(self, value: int) -> None:
        self._value = value

    def scalar_one(self) -> int:
        return self._value


class FakeConnection:
    def __init__(self) -> None:
        self.executions: list[tuple[Any, dict[str, Any] | None]] = []
        self.run_sync_calls: list[Any] = []

    async def execute(
        self,
        statement: Any,
        params: dict[str, Any] | None = None,
    ) -> FakeScalarResult:
        self.executions.append((statement, params))
        return FakeScalarResult(7)

    async def run_sync(self, callable_: Any) -> None:
        self.run_sync_calls.append(callable_)


class FakeConnectionContext:
    def __init__(self, connection: FakeConnection) -> None:
        self.connection = connection

    async def __aenter__(self) -> FakeConnection:
        return self.connection

    async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        return None


class FakeEngine:
    def __init__(self) -> None:
        self.connection = FakeConnection()
        self.begin_calls = 0
        self.connect_calls = 0
        self.dispose_calls = 0

    def begin(self) -> FakeConnectionContext:
        self.begin_calls += 1
        return FakeConnectionContext(self.connection)

    def connect(self) -> FakeConnectionContext:
        self.connect_calls += 1
        return FakeConnectionContext(self.connection)

    async def dispose(self) -> None:
        self.dispose_calls += 1


def _loader_with_fake_engine() -> tuple[PostgresLoader, FakeEngine]:
    loader = PostgresLoader(Settings())
    engine = FakeEngine()
    cast(Any, loader)._engine = engine
    return loader, engine


@pytest.mark.asyncio
async def test_load_awards_returns_zero_without_opening_transaction_for_empty_input() -> None:
    loader, engine = _loader_with_fake_engine()

    loaded = await loader.load_awards([])

    assert loaded == 0
    assert engine.begin_calls == 0


@pytest.mark.asyncio
async def test_create_tables_runs_metadata_create_all() -> None:
    loader, engine = _loader_with_fake_engine()

    await loader.create_tables()

    assert engine.begin_calls == 1
    assert len(engine.connection.run_sync_calls) == 1


@pytest.mark.asyncio
async def test_load_awards_executes_upsert_for_each_contract() -> None:
    loader, engine = _loader_with_fake_engine()
    award = ContractAward(
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

    loaded = await loader.load_awards([award])

    assert loaded == 1
    assert engine.begin_calls == 1
    statement, params = engine.connection.executions[0]
    assert "INSERT INTO contracts" in str(statement)
    assert "ON CONFLICT" in str(statement)
    assert params is not None
    assert params["award_id"] == "CONT_AWD_001"
    assert params["description"] == "Cybersecurity services"


@pytest.mark.asyncio
async def test_count_awards_returns_scalar_count_as_int() -> None:
    loader, engine = _loader_with_fake_engine()

    count = await loader.count_awards()

    assert count == 7
    assert engine.connect_calls == 1
