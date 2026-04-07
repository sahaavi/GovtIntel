"""PostgreSQL loader for structured contract data."""

from __future__ import annotations

import logging

from sqlalchemy import Column, Date, Float, MetaData, String, Table, text
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

from govintel.config import Settings
from govintel.models import ContractAward

logger = logging.getLogger(__name__)

metadata = MetaData()

contracts_table = Table(
    "contracts",
    metadata,
    Column("award_id", String, primary_key=True),
    Column("recipient_name", String, nullable=False),
    Column("awarding_agency", String, nullable=False),
    Column("award_amount", Float, nullable=False),
    Column("start_date", Date, nullable=False),
    Column("end_date", Date, nullable=True),
    Column("naics_code", String, default=""),
    Column("description", String, default=""),
    Column("place_of_performance_state", String, default=""),
    Column("award_type", String, default=""),
)


class PostgresLoader:
    """Loads contract awards into PostgreSQL."""

    def __init__(self, settings: Settings) -> None:
        self._engine: AsyncEngine = create_async_engine(settings.database_url, echo=False)

    async def create_tables(self) -> None:
        """Create the contracts table if it doesn't exist."""
        async with self._engine.begin() as conn:
            await conn.run_sync(metadata.create_all)
        logger.info("Database tables created")

    async def load_awards(self, awards: list[ContractAward]) -> int:
        """Upsert contract awards into the database.

        Args:
            awards: List of ContractAward models to load.

        Returns:
            Number of awards loaded.
        """
        if not awards:
            return 0

        async with self._engine.begin() as conn:
            for award in awards:
                row = award.model_dump()
                # Use INSERT ... ON CONFLICT for upsert
                await conn.execute(
                    text("""
                        INSERT INTO contracts (
                            award_id, recipient_name, awarding_agency, award_amount,
                            start_date, end_date, naics_code, description,
                            place_of_performance_state, award_type
                        ) VALUES (
                            :award_id, :recipient_name, :awarding_agency, :award_amount,
                            :start_date, :end_date, :naics_code, :description,
                            :place_of_performance_state, :award_type
                        )
                        ON CONFLICT (award_id) DO UPDATE SET
                            recipient_name = EXCLUDED.recipient_name,
                            awarding_agency = EXCLUDED.awarding_agency,
                            award_amount = EXCLUDED.award_amount,
                            start_date = EXCLUDED.start_date,
                            end_date = EXCLUDED.end_date,
                            naics_code = EXCLUDED.naics_code,
                            description = EXCLUDED.description,
                            place_of_performance_state = EXCLUDED.place_of_performance_state,
                            award_type = EXCLUDED.award_type
                    """),
                    row,
                )

        logger.info("Loaded %d awards into PostgreSQL", len(awards))
        return len(awards)

    async def count_awards(self) -> int:
        """Return total number of awards in the database."""
        async with self._engine.connect() as conn:
            result = await conn.execute(text("SELECT COUNT(*) FROM contracts"))
            row = result.scalar_one()
            return int(row)

    async def close(self) -> None:
        """Dispose of the database engine."""
        await self._engine.dispose()
