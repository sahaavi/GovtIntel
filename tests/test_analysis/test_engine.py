"""Tests for SQL-based procurement analytics."""

from __future__ import annotations

from collections.abc import AsyncIterator
from datetime import date

import pytest
from sqlalchemy.ext.asyncio import create_async_engine

from govintel.analysis.engine import AnalyticsEngine
from govintel.ingestion.loader import contracts_table, metadata


@pytest.fixture
async def analytics_engine() -> AsyncIterator[AnalyticsEngine]:
    """Analytics engine backed by an isolated async SQLAlchemy database."""

    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(metadata.create_all)
        await conn.execute(
            contracts_table.insert(),
            [
                {
                    "award_id": "DHS-001",
                    "recipient_name": "Alpha Analytics",
                    "awarding_agency": "Department of Homeland Security",
                    "award_amount": 75.0,
                    "start_date": date(2024, 1, 15),
                    "end_date": date(2025, 1, 14),
                    "naics_code": "541512",
                    "description": "DHS cybersecurity monitoring",
                    "place_of_performance_state": "VA",
                    "award_type": "Definitive Contract",
                },
                {
                    "award_id": "DHS-002",
                    "recipient_name": "Beta Systems",
                    "awarding_agency": "Department of Homeland Security",
                    "award_amount": 25.0,
                    "start_date": date(2024, 4, 2),
                    "end_date": date(2025, 4, 1),
                    "naics_code": "541512",
                    "description": "DHS cloud security modernization",
                    "place_of_performance_state": "MD",
                    "award_type": "Delivery Order",
                },
                {
                    "award_id": "DHS-003",
                    "recipient_name": "Alpha Analytics",
                    "awarding_agency": "Department of Homeland Security",
                    "award_amount": 50.0,
                    "start_date": date(2024, 7, 20),
                    "end_date": date(2025, 7, 19),
                    "naics_code": "541512",
                    "description": "DHS SOC engineering",
                    "place_of_performance_state": "VA",
                    "award_type": "Task Order",
                },
                {
                    "award_id": "DOD-001",
                    "recipient_name": "Defense Prime",
                    "awarding_agency": "Department of Defense",
                    "award_amount": 500.0,
                    "start_date": date(2024, 1, 10),
                    "end_date": date(2025, 1, 9),
                    "naics_code": "541512",
                    "description": "DoD cybersecurity support",
                    "place_of_performance_state": "VA",
                    "award_type": "Definitive Contract",
                },
                {
                    "award_id": "DHS-OLD",
                    "recipient_name": "Legacy Inc",
                    "awarding_agency": "Department of Homeland Security",
                    "award_amount": 999.0,
                    "start_date": date(2010, 1, 1),
                    "end_date": date(2011, 1, 1),
                    "naics_code": "541512",
                    "description": "Old DHS contract outside window",
                    "place_of_performance_state": "VA",
                    "award_type": "Definitive Contract",
                },
            ],
        )

    try:
        yield AnalyticsEngine(engine=engine, today=date(2026, 1, 1))
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_top_contractors_returns_ranked_totals_for_matching_agency_and_year_window(
    analytics_engine: AnalyticsEngine,
) -> None:
    top = await analytics_engine.top_contractors(
        agency="Department of Homeland Security",
        years=3,
        limit=5,
    )

    assert [contractor.name for contractor in top] == ["Alpha Analytics", "Beta Systems"]
    assert top[0].total_award_value == 125.0
    assert top[0].contract_count == 2
    assert top[0].avg_contract_value == 62.5
    assert top[0].primary_agencies == ["Department of Homeland Security"]
    assert top[1].total_award_value == 25.0


@pytest.mark.asyncio
async def test_top_contractors_respects_limit(analytics_engine: AnalyticsEngine) -> None:
    top = await analytics_engine.top_contractors(
        agency=None,
        years=3,
        limit=2,
    )

    assert len(top) == 2
    assert [contractor.name for contractor in top] == ["Defense Prime", "Alpha Analytics"]


@pytest.mark.asyncio
async def test_spend_trend_returns_chronological_quarter_totals(
    analytics_engine: AnalyticsEngine,
) -> None:
    trend = await analytics_engine.spend_trend(
        agency="Department of Homeland Security",
        quarters=8,
    )

    assert [(point.quarter, point.total_spend) for point in trend] == [
        ("2024-Q1", 75.0),
        ("2024-Q2", 25.0),
        ("2024-Q3", 50.0),
    ]
    assert [point.award_count for point in trend] == [1, 1, 1]


@pytest.mark.asyncio
async def test_spend_trend_filters_by_agency(analytics_engine: AnalyticsEngine) -> None:
    trend = await analytics_engine.spend_trend(
        agency="Department of Defense",
        quarters=8,
    )

    assert [(point.quarter, point.total_spend) for point in trend] == [("2024-Q1", 500.0)]


@pytest.mark.asyncio
async def test_spend_trend_returns_no_more_than_requested_quarters() -> None:
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(metadata.create_all)
        await conn.execute(
            contracts_table.insert(),
            [
                {
                    "award_id": f"DHS-Q{index}",
                    "recipient_name": "Alpha Analytics",
                    "awarding_agency": "Department of Homeland Security",
                    "award_amount": 10.0,
                    "start_date": start_date,
                    "end_date": None,
                    "naics_code": "541512",
                    "description": "Quarterly cyber support",
                    "place_of_performance_state": "VA",
                    "award_type": "Definitive Contract",
                }
                for index, start_date in enumerate(
                    [
                        date(2024, 1, 1),
                        date(2024, 4, 1),
                        date(2024, 7, 1),
                        date(2024, 10, 1),
                        date(2025, 1, 1),
                        date(2025, 4, 1),
                        date(2025, 7, 1),
                        date(2025, 10, 1),
                        date(2026, 1, 1),
                    ],
                    start=1,
                )
            ],
        )

    try:
        analytics = AnalyticsEngine(engine=engine, today=date(2026, 1, 1))

        trend = await analytics.spend_trend(
            agency="Department of Homeland Security",
            quarters=8,
        )

        assert len(trend) == 8
        assert trend[0].quarter == "2024-Q2"
        assert trend[-1].quarter == "2026-Q1"
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_market_hhi_returns_sum_of_squared_market_shares(
    analytics_engine: AnalyticsEngine,
) -> None:
    hhi = await analytics_engine.market_hhi(
        agency="Department of Homeland Security",
        years=3,
    )

    assert hhi == pytest.approx(7_222.2222)


@pytest.mark.asyncio
async def test_market_hhi_returns_zero_when_no_contracts_match(
    analytics_engine: AnalyticsEngine,
) -> None:
    hhi = await analytics_engine.market_hhi(
        agency="General Services Administration",
        years=3,
    )

    assert hhi == 0.0


@pytest.mark.asyncio
async def test_agency_filter_escapes_like_wildcards(analytics_engine: AnalyticsEngine) -> None:
    top = await analytics_engine.top_contractors(agency="%", years=3, limit=5)

    assert top == []
