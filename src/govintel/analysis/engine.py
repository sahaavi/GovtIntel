"""SQL-backed procurement analytics for structured contract awards."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any

from sqlalchemy import Select, func, select
from sqlalchemy.ext.asyncio import AsyncEngine

from govintel.ingestion.loader import contracts_table
from govintel.models import ContractorSummary

AGENCY_ALIASES = {
    "DHS": "Department of Homeland Security",
    "DOD": "Department of Defense",
    "DoD": "Department of Defense",
    "GSA": "General Services Administration",
}


@dataclass(frozen=True)
class SpendTrendPoint:
    """Quarterly spend aggregate for contract awards."""

    quarter: str
    total_spend: float
    award_count: int


class AnalyticsEngine:
    """Run award-level procurement analytics against the contracts table."""

    def __init__(self, *, engine: AsyncEngine, today: date | None = None) -> None:
        self._engine = engine
        self._today = today or date.today()

    async def top_contractors(
        self,
        *,
        agency: str | None,
        years: int,
        limit: int,
        naics_code: str | None = None,
    ) -> list[ContractorSummary]:
        """Return contractors ranked by total obligated award value."""

        if years <= 0 or limit <= 0:
            return []

        start_date = _years_ago(self._today, years)
        statement = (
            select(
                contracts_table.c.recipient_name.label("name"),
                func.sum(contracts_table.c.award_amount).label("total_award_value"),
                func.count(contracts_table.c.award_id).label("contract_count"),
                func.avg(contracts_table.c.award_amount).label("avg_contract_value"),
            )
            .where(contracts_table.c.start_date >= start_date)
            .group_by(contracts_table.c.recipient_name)
            .order_by(func.sum(contracts_table.c.award_amount).desc())
            .limit(limit)
        )
        statement = apply_agency_filter(statement, agency)
        statement = _apply_naics_filter(statement, naics_code)

        async with self._engine.connect() as conn:
            rows = (await conn.execute(statement)).mappings().all()
            total_spend = await _total_spend(
                conn,
                agency=agency,
                start_date=start_date,
                naics_code=naics_code,
            )

        return [
            ContractorSummary(
                name=str(row["name"]),
                total_award_value=float(row["total_award_value"] or 0.0),
                contract_count=int(row["contract_count"] or 0),
                win_rate=_award_share(float(row["total_award_value"] or 0.0), total_spend),
                primary_agencies=[_normalize_agency(agency)] if agency else [],
                avg_contract_value=float(row["avg_contract_value"] or 0.0),
            )
            for row in rows
        ]

    async def spend_trend(
        self,
        *,
        agency: str | None,
        quarters: int,
        naics_code: str | None = None,
    ) -> list[SpendTrendPoint]:
        """Return quarter-over-quarter spend in chronological order."""

        if quarters <= 0:
            return []

        start_date = _subtract_quarters(_quarter_start(self._today), quarters)
        statement = select(
            contracts_table.c.start_date,
            contracts_table.c.award_amount,
            contracts_table.c.award_id,
        ).where(contracts_table.c.start_date >= start_date)
        statement = apply_agency_filter(statement, agency)
        statement = _apply_naics_filter(statement, naics_code)

        async with self._engine.connect() as conn:
            rows = (await conn.execute(statement)).mappings().all()

        totals: dict[str, tuple[float, int]] = {}
        for row in rows:
            quarter = _quarter_label(row["start_date"])
            current_total, current_count = totals.get(quarter, (0.0, 0))
            totals[quarter] = (
                current_total + float(row["award_amount"] or 0.0),
                current_count + 1,
            )

        points = [
            SpendTrendPoint(
                quarter=quarter,
                total_spend=total,
                award_count=count,
            )
            for quarter, (total, count) in sorted(totals.items())
        ]
        return points[-quarters:]

    async def market_hhi(
        self,
        *,
        agency: str | None,
        years: int,
        naics_code: str | None = None,
    ) -> float:
        """Return Herfindahl-Hirschman Index for contractor concentration."""

        contractors = await self.top_contractors(
            agency=agency,
            years=years,
            limit=10_000,
            naics_code=naics_code,
        )
        total = sum(contractor.total_award_value for contractor in contractors)
        if total <= 0:
            return 0.0

        return sum((contractor.total_award_value / total * 100) ** 2 for contractor in contractors)


def _normalize_agency(agency: str | None) -> str:
    """Map common agency acronyms to stored agency names."""

    if not agency:
        return ""
    return AGENCY_ALIASES.get(agency, agency)


def apply_agency_filter(statement: Select[Any], agency: str | None) -> Select[Any]:
    """Apply a case-insensitive agency filter to a SQLAlchemy statement."""

    normalized = _normalize_agency(agency)
    if not normalized:
        return statement

    escaped = _escape_like(normalized.lower())
    return statement.where(
        func.lower(contracts_table.c.awarding_agency).like(f"%{escaped}%", escape="\\")
    )


def _apply_naics_filter(statement: Select[Any], naics_code: str | None) -> Select[Any]:
    """Apply exact NAICS filtering when supplied."""

    if not naics_code:
        return statement
    return statement.where(contracts_table.c.naics_code == naics_code)


def _escape_like(value: str) -> str:
    """Escape user-supplied LIKE wildcards."""

    return value.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


async def _total_spend(
    conn: Any,
    *,
    agency: str | None,
    start_date: date,
    naics_code: str | None,
) -> float:
    """Return total filtered spend for award-share calculations."""

    statement = select(func.sum(contracts_table.c.award_amount)).where(
        contracts_table.c.start_date >= start_date
    )
    statement = apply_agency_filter(statement, agency)
    statement = _apply_naics_filter(statement, naics_code)
    return float((await conn.execute(statement)).scalar_one() or 0.0)


def _award_share(total_award_value: float, total_spend: float) -> float:
    """Return a bounded award-share fraction for the existing win_rate field."""

    if total_spend <= 0:
        return 0.0
    return max(0.0, min(total_award_value / total_spend, 1.0))


def _years_ago(value: date, years: int) -> date:
    """Subtract whole years while handling leap-day edge cases."""

    try:
        return value.replace(year=value.year - years)
    except ValueError:
        return value.replace(year=value.year - years, day=28)


def _quarter_start(value: date) -> date:
    """Return the first day of the quarter containing the given date."""

    month = ((value.month - 1) // 3) * 3 + 1
    return date(value.year, month, 1)


def _subtract_quarters(value: date, quarters: int) -> date:
    """Subtract whole quarters from a quarter-start date."""

    month_index = value.year * 12 + value.month - 1 - quarters * 3
    year, month_zero_based = divmod(month_index, 12)
    return date(year, month_zero_based + 1, 1)


def _quarter_label(value: date) -> str:
    """Render a date as an ISO-like quarter label."""

    quarter = (value.month - 1) // 3 + 1
    return f"{value.year}-Q{quarter}"


__all__ = ["AnalyticsEngine", "SpendTrendPoint", "apply_agency_filter"]
