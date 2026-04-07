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
    ) -> list[dict[str, Any]]:
        """Search contract awards with the given filters.

        Args:
            filters: USAspending API filter object.
            page: Page number (1-indexed).
            limit: Results per page.

        Returns:
            List of raw award dicts from the API response.
        """
        client = await self._get_client()
        payload = {
            "filters": filters,
            "page": page,
            "limit": limit,
            "sort": "Award Amount",
            "order": "desc",
        }

        response = await client.post("/search/spending_by_award/", json=payload)
        response.raise_for_status()

        data = response.json()
        return data.get("results", [])

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

            results = await self.search_awards(filters=filters, page=page, limit=page_size)

            if not results:
                logger.info("No more results at page %d, stopping", page)
                break

            for raw in results:
                try:
                    awards.append(ContractAward.from_usaspending(raw))
                except Exception:
                    logger.warning("Failed to parse award: %s", raw.get("id", "unknown"))

            if len(results) < page_size:
                break

            await asyncio.sleep(RATE_LIMIT_DELAY_SECONDS)

        logger.info("Fetched %d awards total", len(awards))
        return awards
