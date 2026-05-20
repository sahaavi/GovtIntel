"""USAspending.gov API client with pagination and rate limiting."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import httpx

from govintel.config import Settings
from govintel.models import ContractAward

logger = logging.getLogger(__name__)

DEFAULT_PAGE_SIZE = 100
RATE_LIMIT_DELAY_SECONDS = 0.5
CONTRACT_AWARD_TYPE_CODES = ["A", "B", "C", "D"]
DEFAULT_AWARD_FIELDS = [
    "generated_internal_id",
    "Award ID",
    "Recipient Name",
    "Awarding Agency",
    "Award Amount",
    "Start Date",
    "End Date",
    "NAICS",
    "Description",
    "Place of Performance State Code",
    "Contract Award Type",
]


def build_contract_award_filters(
    *,
    naics_code: str | None = None,
    agency: str | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
) -> dict[str, Any]:
    """Build a small USAspending filter object for contract awards."""

    filters: dict[str, Any] = {"award_type_codes": list(CONTRACT_AWARD_TYPE_CODES)}
    if naics_code:
        filters["naics_codes"] = {"require": [naics_code]}
    if agency:
        filters["agencies"] = [
            {
                "type": "awarding",
                "tier": "toptier",
                "name": agency,
            }
        ]
    if start_date and end_date:
        filters["time_period"] = [{"start_date": start_date, "end_date": end_date}]
    return filters


class USAspendingClient:
    """Async client for the USAspending.gov API."""

    def __init__(self, settings: Settings) -> None:
        self._base_url = settings.usaspending_base_url.rstrip("/")
        self._client: httpx.AsyncClient | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=self._base_url,
                timeout=httpx.Timeout(30.0),
                headers={"Content-Type": "application/json"},
            )
        return self._client

    async def close(self) -> None:
        """Close the underlying HTTP client."""
        if self._client is not None and not self._client.is_closed:
            await self._client.aclose()
            self._client = None

    async def search_awards(
        self,
        *,
        filters: dict[str, Any],
        page: int = 1,
        limit: int = DEFAULT_PAGE_SIZE,
        fields: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Search contract awards with the given filters.

        Args:
            filters: USAspending API filter object.
            page: Page number (1-indexed).
            limit: Results per page.

        Returns:
            List of raw award dicts from the API response.
        """
        results, _ = await self._search_awards_page(
            filters=filters,
            page=page,
            limit=limit,
            fields=fields,
        )
        return results

    async def _search_awards_page(
        self,
        *,
        filters: dict[str, Any],
        page: int,
        limit: int,
        fields: list[str] | None = None,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """Search one award page and return rows plus pagination metadata."""
        client = await self._get_client()
        payload = {
            "subawards": False,
            "filters": filters,
            "fields": fields or DEFAULT_AWARD_FIELDS,
            "page": page,
            "limit": limit,
            "sort": "Award Amount",
            "order": "desc",
        }

        response = await client.post("/search/spending_by_award/", json=payload)
        response.raise_for_status()

        data = response.json()
        return data.get("results", []), data.get("page_metadata", {})

    async def fetch_all_awards(
        self,
        *,
        filters: dict[str, Any],
        max_pages: int = 10,
        page_size: int = DEFAULT_PAGE_SIZE,
    ) -> list[ContractAward]:
        """Fetch multiple pages of awards with rate limiting.

        Args:
            filters: USAspending API filter object.
            max_pages: Maximum number of pages to fetch.
            page_size: Results per page.

        Returns:
            List of parsed ContractAward models.
        """
        awards: list[ContractAward] = []

        for page in range(1, max_pages + 1):
            logger.info("Fetching page %d (limit=%d)", page, page_size)

            results, page_metadata = await self._search_awards_page(
                filters=filters,
                page=page,
                limit=page_size,
            )

            if not results:
                logger.info("No more results at page %d, stopping", page)
                break

            for raw in results:
                try:
                    awards.append(ContractAward.from_usaspending(raw))
                except Exception:
                    logger.warning("Failed to parse award: %s", raw.get("id", "unknown"))

            has_next = bool(page_metadata.get("hasNext", len(results) == page_size))
            if not has_next:
                break

            await asyncio.sleep(RATE_LIMIT_DELAY_SECONDS)

        logger.info("Fetched %d awards total", len(awards))
        return awards


__all__ = [
    "CONTRACT_AWARD_TYPE_CODES",
    "DEFAULT_AWARD_FIELDS",
    "USAspendingClient",
    "build_contract_award_filters",
]
