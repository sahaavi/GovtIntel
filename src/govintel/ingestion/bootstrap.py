"""Minimal USAspending-to-PostgreSQL bootstrap for local data availability."""

from __future__ import annotations

import argparse
import asyncio
from dataclasses import dataclass
from typing import Any, Protocol

from govintel.config import Settings, get_settings
from govintel.ingestion.loader import PostgresLoader
from govintel.ingestion.usaspending import USAspendingClient, build_contract_award_filters
from govintel.models import ContractAward


class AwardClient(Protocol):
    async def fetch_all_awards(
        self,
        *,
        filters: dict[str, Any],
        max_pages: int,
        page_size: int,
    ) -> list[ContractAward]:
        """Fetch parsed contract awards."""

    async def close(self) -> None:
        """Release client resources."""


class AwardLoader(Protocol):
    async def create_tables(self) -> None:
        """Create required tables."""

    async def load_awards(self, awards: list[ContractAward]) -> int:
        """Persist contract awards."""

    async def close(self) -> None:
        """Release loader resources."""


@dataclass(frozen=True)
class BootstrapResult:
    """Counts from one contract data bootstrap run."""

    awards_fetched: int
    awards_loaded: int


async def bootstrap_contract_data(
    *,
    settings: Settings | None = None,
    client: AwardClient | None = None,
    loader: AwardLoader | None = None,
    naics_code: str | None = None,
    agency: str | None = None,
    max_pages: int | None = None,
    page_size: int | None = None,
) -> BootstrapResult:
    """Fetch a bounded USAspending contract slice and load it into PostgreSQL."""

    resolved_settings = settings or get_settings()
    own_client = client is None
    own_loader = loader is None
    award_client = client or USAspendingClient(resolved_settings)
    award_loader = loader or PostgresLoader(resolved_settings)

    filters = build_contract_award_filters(
        naics_code=naics_code or resolved_settings.ingestion_naics_code,
        agency=agency,
    )

    try:
        await award_loader.create_tables()
        awards = await award_client.fetch_all_awards(
            filters=filters,
            max_pages=max_pages or resolved_settings.ingestion_max_pages,
            page_size=page_size or resolved_settings.ingestion_page_size,
        )
        loaded = await award_loader.load_awards(awards)
        return BootstrapResult(awards_fetched=len(awards), awards_loaded=loaded)
    finally:
        if own_client:
            await award_client.close()
        if own_loader:
            await award_loader.close()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fetch a bounded USAspending contract slice into PostgreSQL."
    )
    parser.add_argument("--naics-code", help="Restrict import to one NAICS code.")
    parser.add_argument("--agency", help="Restrict import to one awarding toptier agency.")
    parser.add_argument("--max-pages", type=int, help="Maximum USAspending pages to fetch.")
    parser.add_argument("--page-size", type=int, help="USAspending page size.")
    return parser


def main(argv: list[str] | None = None) -> None:
    """Run the local bootstrap command."""

    args = _build_parser().parse_args(argv)
    result = asyncio.run(
        bootstrap_contract_data(
            naics_code=args.naics_code,
            agency=args.agency,
            max_pages=args.max_pages,
            page_size=args.page_size,
        )
    )
    print(
        "Loaded "
        f"{result.awards_loaded} of {result.awards_fetched} fetched awards into PostgreSQL."
    )


if __name__ == "__main__":
    main()
